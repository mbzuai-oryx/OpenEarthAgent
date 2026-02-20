"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import torch
import uvicorn
from functools import partial

from enum import IntEnum

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303
    TIMEOUT_ERROR = 40304

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("tool_worker", f"base_tool_worker_{worker_id}.log")
#model_semaphore = None

class BaseToolWorker:
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = None, 
                 model_base = None, 
                 model_name = None,
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 wait_timeout = 120.0,
                 task_timeout = 30.0,
                 args = None,
                 ):
        self.args = args
        self.controller_addr = controller_addr
        assert port is not None, "Port must be specified"
        if worker_addr == "auto":
            node_name = os.getenv("SLURMD_NODENAME", "Unknown")
            print(f"SLURM Node Name: {node_name}")
            if node_name == "Unknown":
                node_name = "localhost"
            self.worker_addr = f"http://{node_name}:{port}"
        else:
            self.worker_addr = worker_addr
            
        self.model_path = model_path
        self.model_base = model_base
        if model_name is None:
            self.model_name = model_path.split("/")[-1]
        else:
            self.model_name = model_name
        
        if model_semaphore is not None:
            self.model_semaphore = model_semaphore
        else:
            self.model_semaphore = None
        
        self.worker_id = worker_id
        self.no_register = no_register
        
       
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        self.limit_model_concurrency = limit_model_concurrency
        self.host = host
        self.port = port
        
        self.global_counter = 0
        # self.wait_timeout = wait_timeout
        self.task_timeout = task_timeout
        self.semaphore_timeout = wait_timeout
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=self.heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
            
        # Set up the routes
        self.app = FastAPI()
        self.setup_routes()
        self.init_model()
        
        

    ## HTTP Methods    
    def heart_beat_worker(self, controller):
        while True:
            time.sleep(WORKER_HEART_BEAT_INTERVAL)
            controller.send_heart_beat()

        
    def release_model_semaphore(self, fn=None):
        if self.model_semaphore:
            self.model_semaphore.release()
            if fn is not None:
                fn()
    
    async def acquire_model_semaphore(self):
        self.global_counter += 1
        if self.model_semaphore is None:
            # Use a threading semaphore instead of asyncio semaphore for thread safety
            self.model_semaphore = threading.Semaphore(self.limit_model_concurrency)
        # For thread-safe acquisition in async context with timeout
        loop = asyncio.get_event_loop()
        
        future = loop.run_in_executor(None, lambda: self.model_semaphore.acquire(timeout=self.semaphore_timeout))
        try:
            return await asyncio.wait_for(future, timeout=self.semaphore_timeout)
        except asyncio.TimeoutError:
            logger.error(f"Semaphore acquisition timed out after {self.semaphore_timeout}s")
            raise TimeoutError("Model is busy, request timed out while waiting in queue")
                
    def setup_routes(self):
        @self.app.post("/worker_generate")
        async def api_generate(request: Request):
            params = await request.json()
            try:
                await self.acquire_model_semaphore()
                output = self.generate_gate(params)
                return JSONResponse(output)
            except TimeoutError:
                return JSONResponse({
                    "text": "Request timed out while waiting in queue",
                    "error_code": ErrorCode.TIMEOUT_ERROR
                }, status_code=503)
            finally:
                # Only release if we actually acquired it
                if self.model_semaphore._value < self.limit_model_concurrency:
                    self.release_model_semaphore()
        

        @self.app.post("/worker_get_status")
        async def get_status(request: Request):
            return self.get_status()
        
        @self.app.post("/model_details")
        async def model_details(request: Request):
            pass
        
        @self.app.post("/tool_instruction")
        async def tool_instruction(request: Request):
            params = await request.json()
            try:
                tool_instruction = self.get_tool_instruction()
                return JSONResponse({
                    "tool_instruction": tool_instruction,
                    "error_code": 0
                })
            except Exception as e:
                logger.error(f"Error getting tool instruction: {e}")
                return JSONResponse({
                    "text": SERVER_ERROR_MSG,
                    "error_code": ErrorCode.INTERNAL_ERROR
                }, status_code=500)
    
    def generate_gate(self, params):
        try:
            ret = {"text": "", "error_code": 0}
            ret = self.generate(params)
            # ret = asyncio.get_event_loop().run_until_complete(self.async_generate(params))
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret
    
    
    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200, f"Failed to register to controller: {r.text}"

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(self.model_semaphore)}. "
                    f"global_counter: {self.global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if self.model_semaphore is None:
            return 0
        value = getattr(self.model_semaphore, "_value", self.limit_model_concurrency)
        used = self.limit_model_concurrency - value
        return max(0, used)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    # Launch method
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info",log_config=None)
    
    # abstract methods
    def init_model(self):
        pass
    
    @torch.inference_mode()
    def generate(self, params):
        pass

    def get_tool_instruction(self):
        pass

    

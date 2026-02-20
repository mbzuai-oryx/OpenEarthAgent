"""
A model worker executes the Change Detection model (TEOChat).
"""

import argparse
import traceback
import uuid
import numpy as np

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

# worker id and logger
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")


class ChangeDetectionWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="ChangeDetection",
                 model_base="jirvin16/TEOChat",
                 device="cuda",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 args=None):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,   # model path/name
            model_base,
            model_name,
            False,   # load_8bit handled in load_model
            False,   # load_4bit not used here
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            args=args
        )
        self.device = device

    def init_model(self):
        logger.info(f"Initializing ChangeDetection model: {self.model_base}")
        try:
            from videollava.eval.eval import load_model
            self.tokenizer, self.model, self.processor = load_model(
                model_path=self.model_base,
                model_base=None,
                load_8bit=True,
                device=self.device
            )
            print(f"[DEBUG] ChangeDetection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing ChangeDetection model: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate(self, params):
        required_keys = ("pre_image", "post_image", "text")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        pre_image = params.get("pre_image")
        post_image = params.get("post_image")
        query_text = params.get("text")

        for key in ["pre_image", "post_image"]:
            val = locals()[key]
            if not (os.path.exists(val)):
                txt_e = f"Image not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}

        try:
            from videollava.eval.inference import run_inference_single

            pre_image = params["pre_image"].strip()
            post_image = params["post_image"].strip()
            query = "These are two satellite images in chronological order: <video> " + query_text

            txt = run_inference_single(
                self.model, self.processor, self.tokenizer, query, [pre_image, post_image]
            )
            return {"text": txt, "error_code": 0}
        
        except Exception as e:
            txt_e = f"Error in ChangeDetection: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "ChangeDetection",
                "description": "Detect and describe changes between two chronological satellite images.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Natural language query, e.g., 'Identify the damaged buildings.'"
                        },
                        "pre_image": {
                            "type": "string",
                            "description": "Path of the pre-change image."
                        },
                        "post_image": {
                            "type": "string",
                            "description": "Path of the post-change image."
                        }
                    },
                    "required": ["text", "pre_image", "post_image"]
                }
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20029)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://127.0.0.1:20001")
    parser.add_argument("--model_base", type=str, default="jirvin16/TEOChat")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ChangeDetectionWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        model_base=args.model_base,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
        device=args.device,
        args=args,
    )
    worker.run()

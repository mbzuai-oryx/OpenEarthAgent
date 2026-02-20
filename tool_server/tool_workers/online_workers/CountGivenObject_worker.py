import torch
import numpy as np
from PIL import Image
import base64
import argparse
import uuid
import os
import re
from io import BytesIO
import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"CountGivenObject_worker_{worker_id}.log")

class CountGivenObject(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="CountGivenObject",
                 device="cuda",
                 BaseModel_server="BaseModel", 
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        self.BaseModel_server = BaseModel_server
        self.BaseModel_server_addr = None
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def _resolve_base_model(self):
        """Resolve BaseModel worker address via controller."""
        if self.BaseModel_server_addr:
            return self.BaseModel_server_addr
        try:
            ret = requests.post(
                self.controller_addr + "/get_worker_address",
                json={"model": self.BaseModel_server},
                timeout=5,
            )
            self.BaseModel_server_addr = (ret.json().get("address") or "").strip()
            if self.BaseModel_server_addr:
                logger.info(f"Resolved BaseModel worker address: {self.BaseModel_server_addr}")
            else:
                logger.warning("Controller did not return a valid BaseModel address.")
        except Exception as e:
            logger.warning(f"Could not resolve BaseModel worker address: {e}")
            self.BaseModel_server_addr = None
        return self.BaseModel_server_addr
    
    def init_model(self):
        logger.info(f"Initializing {self.model_name} worker...")
        self._resolve_base_model()  

    def _load_image(self, image_path: str):
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        else:
            return Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")

    @torch.inference_mode()
    def generate(self, params):
        required_keys = ("image", "text")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        image = params.get("image")
        text = params.get("text")
        bbox = params.get("bbox")  # expected format: (x1, y1, x2, y2)
        
        if not os.path.exists(image):
                txt_e = f"Image not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
        
        # Ensure bbox is a string in the form (x1,y1,x2,y2)
        if bbox is not None:
            if not isinstance(bbox, str) or not re.match(r'^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)$', bbox):
                txt_e = f"Invalid bbox format: {bbox}. Expected format '(x1,y1,x2,y2)'."
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}
        
        if not self.BaseModel_server_addr:
            if not self._resolve_base_model():
                txt_e = "BaseModel server not available"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

        try:
            img = self._load_image(image)
            buf = BytesIO()
            if bbox:
                x1, y1, x2, y2 = [int(float(c)) for c in bbox.strip("()").split(",")]
                
                # check if bbox is inside image bounds
                W, H = img.size
                if not (0 <= x1 < W and 0 <= x2 <= W and 0 <= y1 < H and 0 <= y2 <= H):
                    txt_e = f"Invalid bbox coordinates: {(x1,y1,x2,y2)} outside image size {(W,H)}"
                    logger.error(txt_e)
                    return {"text": txt_e, "error_code": 1}

                # check if bbox defines a valid region
                if not (x2 > x1 and y2 > y1):
                    txt_e = f"Invalid bbox region: {(x1,y1,x2,y2)} must satisfy x2>x1 and y2>y1"
                    logger.error(txt_e)
                    return {"text": txt_e, "error_code": 4}
                # crop and encode region
                img.crop((x1, y1, x2, y2)).save(buf, format="PNG")
            else:
                # encode full image to base64 if no bbox
                img.save(buf, format="PNG")
            
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            payload = {"image": img_b64, "query": f"Count the number of {text} in this aerial image. Answer ONLY with a single integer."}
            headers = {"User-Agent": "CountGivenObjectWorker"}
            resp = requests.post(
                self.BaseModel_server_addr + "/worker_generate",
                headers=headers,
                json=payload,
                timeout=30,
            ).json()

            if resp.get("error_code", 1) != 0:
                logger.error(f"BaseModel failed: {resp}")
                return {"text": resp, "error_code": 4}

            txt = resp["text"]
            return {"text": txt, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in CountGivenObject: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}
        
    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "CountGivenObject",
                "description": "Generate a detailed natural language description of the given image using BaseModel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier or path of the image to be described."
                        },
                        "text": {
                            "type": "string",
                            "description": "The object to count, described in English. "
                        },
                        'bbox' : {
                            "type": "string",
                            "description": (
                                "Optional bounding box specifying the region to search in, in the format '(x1,y1,x2,y2)'"
                            )
                        },
                    },
                    "required": ["image", "text"],
                    "optional": ["bbox"]
                }
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20004)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="CountGivenObject")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--BaseModel-server", type=str, default="BaseModel")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = CountGivenObject(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        device=args.device,
        BaseModel_server=args.BaseModel_server,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

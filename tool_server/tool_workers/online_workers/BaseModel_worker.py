import torch
import argparse
import uuid
import os

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger
from transformers import AutoProcessor, AutoModelForImageTextToText

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"BaseModel_worker_{worker_id}.log")


class BaseModelWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="BaseModel",
                 model_path ="Qwen/Qwen3-VL-8B-Instruct",
                 device="cuda",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
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
        self.device = device

    def init_model(self):
        logger.info(f"Initializing model {self.model_name} ...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype=torch.float16, device_map="auto",trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        try:
            model_device = next(self.model.parameters()).device
            logger.info(f"{self.model_name} loaded successfully on {model_device}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        except Exception as e:
            logger.error(f"Could not determine device: {e}")

    @torch.inference_mode()
    def generate(self, params):
        required_keys = ("image", "query")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}
        
        image = params.get("image")
        query = params.get("query")
        
        try:
            image_str = params["image"]
            if os.path.exists(image_str):
                image_content = {"type": "image", "path": image_str}
            else:
                image_content = {"type": "image", "base64": image_str}

            messages = [{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": query},
                ],
            }]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = inputs.to(device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            txt = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return {"text": txt, "error_code": 0}

        
        except Exception as e:
            txt_e = f"Error in BaseModel: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}


    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "BaseModel",
                "description": "General vision-language worker. Provide an image and a query to get an answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Image path"},
                        "query": {"type": "string", "description": "Question about the image"},
                    },
                    "required": ["image"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20100)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    logger.info(f"args: {args}")

    worker = BaseModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_path=args.model_path,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

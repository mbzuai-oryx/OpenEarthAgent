import torch
import numpy as np
from PIL import Image
import base64
import argparse
import uuid
import os
from io import BytesIO
import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SegmentObjectPixels_worker_{worker_id}.log")

class SegmentObjectPixelsWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="SegmentObjectPixels",
                 device="cuda",
                 BaseModel_server="TextToBbox",
                 sam_model_path="sam2.1_hiera_large.pt",
                 sam_model_config="sam2.1_hiera_l.yaml",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        self.BaseModel_server = BaseModel_server
        self.BaseModel_server_addr = None
        self.sam_model_path = sam_model_path
        self.sam_model_config = sam_model_config
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
                logger.info(f"Resolved BaseModel worker: {self.BaseModel_server_addr}")
            else:
                logger.warning("Controller did not return a valid BaseModel address.")
        except Exception as e:
            logger.warning(f"Could not resolve BaseModel worker address: {e}")
            self.BaseModel_server_addr = None
        return self.BaseModel_server_addr

    def init_model(self):
        logger.info(f"Initializing {self.model_name} worker...")
        self._resolve_base_model()
        # load SAM
        # ✅ validate SAM model checkpoint
        if not os.path.exists(self.sam_model_path):
            txt_e = f"SAM model checkpoint not found: {self.sam_model_path}"
            logger.error(txt_e)
            raise FileNotFoundError(txt_e)
        if not os.path.exists(self.sam_model_config):
            txt_e = f"SAM model config not found: {self.sam_model_config}"
            logger.error(txt_e)
            raise FileNotFoundError(txt_e)
        try:
            self.sam_predictor = SAM2ImagePredictor(build_sam2(self.sam_model_config, self.sam_model_path))
            logger.info("SAM model loaded successfully.")
        except Exception as e:
            txt_e = f"Failed to load SAM model from {self.sam_model_path}: {e}"
            logger.error(txt_e)
            raise RuntimeError(txt_e)
        
    def _load_image(self, image: str):
        if os.path.exists(image):
            return Image.open(image).convert("RGB")
        else:
            return Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    
    def _segment_with_sam(self, image_np, boxes):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_predictor.set_image(image_np)
            masks, scores, _ =  self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False,
                )
        masks = masks.squeeze(1) if masks.ndim == 4 else masks
        return masks

    @torch.inference_mode()
    def generate(self, params):
        required_keys = ("image", "text")

        missing = [k for k in required_keys if k not in params]
        if missing:
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}
        
        image = params.get("image")
        text = params.get("text")
        flag = params.get("flag",True)  # optional 

        if not os.path.exists(image):
                txt_e = f"Image not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
        
        if not isinstance(flag, bool):
            txt_e = f"Invalid flag type: {type(flag)}. Must be boolean."
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 4}
            
        if not self.BaseModel_server_addr:
            if not self._resolve_base_model():
                txt_e = "BaseModel server not available"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

        try:
            # send to BaseModel (TextToBbox)
            payload = {"image": image,"text": text}
            headers = {"User-Agent": "SegmentObjectPixelsWorker"}
            resp = requests.post(
                self.BaseModel_server_addr + "/worker_generate",
                headers=headers,
                json=payload,
                timeout=60,
            ).json()

            if resp.get("error_code", 1) != 0:
                logger.error(f"BaseModel failed: {resp}")
                return {"text": resp, "error_code": 4}

            detections = resp["text"].split("\n")
            boxes = []
            for det in detections:
                coords, conf = det.split("),")
                x1, y1, x2, y2 = [int(c) for c in coords.strip("()").split(",")]
                boxes.append([x1, y1, x2, y2])
            if len(boxes) == 0:
                txt_e = f"No objects found for given object: {text}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}
            
            boxes = torch.tensor(boxes, dtype=torch.float32, device=self.device)
            img = self._load_image(image)
            img_np = np.array(img)
            masks = self._segment_with_sam(img_np, boxes)
            ###################################
            # Combine all masks into one
            for i, mask in enumerate(masks):
                if hasattr(mask, "detach"):
                    mask = mask.detach().cpu().numpy()
                print(mask.shape) 
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L") \
                    .save(f"mask_{i}.png")
            ###########################################
            pixel_counts = [int(mask.sum()) for mask in masks]
            if not flag:
                pixel_counts = sum(pixel_counts)
            
            return {"text": str(pixel_counts), "error_code": 0}

        except Exception as e:
            txt_e = f"Error in SegmentObjectPixels: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "SegmentObjectPixels",
                "description": "Segment the specified object in the input image, and return the segmentated object's pixel count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Image path or base64 string"},
                        "text": {"type": "string", "description": "Object name/description in English"},
                        "flag": {
                            "type": "boolean",
                            "description": "If true, return the list of pixel counts per segmented object. If false, return the total pixel count across all detections. Default : true"
                        }
                    },
                    "required": ["image", "text"],
                    "optional": ["flag"]
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20005)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SegmentObjectPixels")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--BaseModel-server", type=str, default="TextToBbox")
    parser.add_argument("--sam_model_path", type=str, default="sam2.1_hiera_large.pt")
    parser.add_argument("--sam_model_config", type=str, default="sam2.1_hiera_l.yaml")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SegmentObjectPixelsWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        device=args.device,
        BaseModel_server=args.BaseModel_server,
        sam_model_path=args.sam_model_path,
        sam_model_config=args.sam_model_config,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

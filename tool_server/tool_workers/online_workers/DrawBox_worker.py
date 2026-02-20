import argparse
import uuid
import os
import re
import base64
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


def parse_multi_float(input_str: str, number: Optional[int] = None) -> Tuple[float, ...]:
    pattern = r'([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, input_str)

    if number is not None and len(matches) != number:
        raise ValueError(f'Expected {number} numbers, got {input_str}.')
    else:
        return tuple(float(num) for num in matches)


# --- worker setup ---
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"draw_box_worker_{worker_id}.log")


class DrawBoxWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="DrawBox",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 save_path = None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        self.save_path = save_path
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,  # no model path needed
            None,
            model_name,
            False,  # load_8bit
            False,  # load_4bit
            "cpu",  # always CPU
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info(f"DrawBoxWorker does not need a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")
    
    def _load_image(self, image_path: str):
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        else:
            return Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")

    def generate(self, params):
        try:
            required_keys = ("image", "bbox")

            if any(k not in params for k in required_keys):
                missing = [k for k in required_keys if k not in params]
                txt_e = f"Missing required parameter(s): {', '.join(missing)}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 2}

            image = params.get("image")
            bbox = params.get("bbox")
            annotation = params.get("annotation", None)
            
            if not os.path.exists(image):
                txt_e = f"Image not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
            
            # Ensure bbox is a string in the form (x1,y1,x2,y2)
            if not isinstance(bbox, str) or not re.match(r'^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)$', bbox):
                txt_e = f"Invalid bbox format: {bbox}. Expected format '(x1,y1,x2,y2)'."
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}
            
            img = self._load_image(image).convert("RGBA")
            x1, y1, x2, y2 = [int(float(c)) for c in bbox.strip("()").split(",")]
            
            # check if bbox is inside image bounds
            W, H = img.size
            if not (0 <= x1 < W and 0 <= x2 <= W and 0 <= y1 < H and 0 <= y2 <= H):
                txt_e = f"Invalid bbox coordinates: {(x1,y1,x2,y2)} outside image size {(W,H)}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            # check if bbox defines a valid region
            if not (x2 > x1 and y2 > y1):
                txt_e = f"Invalid bbox region: {(x1,y1,x2,y2)} must satisfy x2>x1 and y2>y1"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}
            
            canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)

            # Estimate font size
            fontsize = int(img.size[1] / 336 * 18)
            try:
                font = ImageFont.truetype("arial.ttf", fontsize)
            except:
                font = ImageFont.load_default()

            draw.rectangle((x1, y1, x2, y2),
                           fill=(255, 0, 0, 64),
                           outline=(255, 0, 0, 255))
            if annotation:
                draw.text((x1, max(0, y1 - 5)),
                          annotation,
                          fill=(255, 0, 0, 255),
                          anchor="lb",
                          font=font)

            result_img = Image.alpha_composite(img, canvas).convert("RGB")

            # Save result
            image_name = os.path.basename(os.path.splitext(image)[0])
            new_filename = f"{image_name}_boxdrawn.png"
            if self.save_path and os.path.isdir(self.save_path):
                save_path = os.path.join(self.save_path, new_filename)
            else:
                if not os.path.isdir(self.save_path):
                    logger.warning(f"Save path '{self.save_path}' is not a valid directory. "
                                f"Falling back to default ./tools_output/")
                save_dir = os.path.join(os.getcwd(), "tools_output")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, new_filename)
            
            result_img.save(save_path)

            txt = f"Annotated image saved to {new_filename}"
            return {"text": txt, "image":save_path, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in DrawBox: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "DrawBox",
                "description": "Draws a bounding box with optional annotation on an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Path to the image."
                        },
                        "bbox": {
                            "type": "string",
                            "description": "Bounding box in format '(x1, y1, x2, y2)'."
                        },
                        "annotation": {
                            "type": "string",
                            "description": "Optional text to display near the box."
                        }
                    },
                    "required": ["image", "bbox"],
                    "optional": ["annotation"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20004)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="DrawBox")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = DrawBoxWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        save_path=args.save_path,
        host=args.host,
        port=args.port,
    )
    worker.run()

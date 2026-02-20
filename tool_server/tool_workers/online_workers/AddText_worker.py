import argparse
import uuid
import os
import re
import base64
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(
    BASE_DIR,
    "fonts",
    "NotoSansCJK-Regular.ttc"
)
def parse_multi_float(input_str: str, number: Optional[int] = None) -> Tuple[float, ...]:
    pattern = r'([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, input_str)
    if number is not None and len(matches) != number:
        raise ValueError(f'Expected {number} numbers, got {input_str}.')
    else:
        return tuple(float(num) for num in matches)


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"add_text_worker_{worker_id}.log")


class AddTextWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="AddText",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 save_path=None,
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
        logger.info("AddTextWorker does not need a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

    def _load_image_simple(self, image_path: str) -> Image.Image:
        """
        Load non-GeoTIFF images (JPG/PNG or base64) using PIL.
        """
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            # treat as base64-encoded image
            return Image.open(BytesIO(base64.b64decode(image_path)))

    def generate(self, params):
        try:
            required_keys = ("image", "text", "position")

            if any(k not in params for k in required_keys):
                missing = [k for k in required_keys if k not in params]
                txt_e = f"Missing required parameter(s): {', '.join(missing)}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 2}

            image = params["image"]
            text = params["text"]
            position = params["position"]
            color = params.get("color", "red")
            font_size = params.get("font_size")
            # For path-based images, ensure file exists
            is_path = os.path.exists(image)
            if is_path is False:
                # If it's not a path, we still allow base64 for non-tif
                # but for GeoTIFF behavior (preserving metadata) we require a path.
                logger.info("Image not found on disk; treating 'image' as base64 for non-tif formats.")
            
            # Determine extension (if path-based)
            ext = ".png"
            if is_path:
                ext = os.path.splitext(image)[1].lower()
            else:
                # Treat base64 inputs as PNG-like for output naming
                ext = ".png"

            pil_img = None
            profile = None
            raster_data = None

            ###################################################################################
            # CASE A: GeoTIFF (.tif/.tiff) with path – use rasterio to preserve geo information
            ###################################################################################
            if is_path and ext in [".tif", ".tiff"]:
                try:
                    with rasterio.open(image) as src:
                        profile = src.profile.copy()
                        raster_data = src.read()

                    # Convert raster data to an image array (H, W, bands)
                    img_arr = reshape_as_image(raster_data)
                    pil_img = Image.fromarray(img_arr)
                except Exception as e:
                    txt_e = f"Error reading GeoTIFF with rasterio: {e}"
                    logger.error(txt_e)
                    return {"text": txt_e, "error_code": 1}

            ####################################################################
            # CASE B: JPG/PNG/other (or base64) – fast PIL path
            ####################################################################
            else:
                try:
                    pil_img = self._load_image_simple(image)
                except Exception as e:
                    txt_e = f"Error loading image: {e}"
                    logger.error(txt_e)
                    return {"text": txt_e, "error_code": 1}

            draw = ImageDraw.Draw(pil_img)

            # -------- FONT SIZE --------
            w, h = pil_img.size
            if not font_size:
                fontsize = max(20, int(h * 0.025))
            else:
                fontsize = font_size
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except Exception:
                font = ImageFont.load_default()

            # -------- POSITIONING --------
            w, h = pil_img.size
            m = w // 20
            POS = {
                'lt': (m, m), 'lm': (m, h // 2), 'lb': (m, h - m),
                'mt': (w // 2, m), 'mm': (w // 2, h // 2), 'mb': (w // 2, h - m),
                'rt': (w - m, m), 'rm': (w - m, h // 2), 'rb': (w - m, h - m),
            }

            if position in POS:
                xy = POS[position]
                anchor = position
            else:
                try:
                    x, y = parse_multi_float(position, 2)
                    xy = (x, y)
                    anchor = None  # explicit coords, no anchor
                except ValueError:
                    msg = "Invalid position string."
                    logger.error(msg)
                    return {"text": msg, "error_code": 1}

            # -------- DRAW TEXT WITH TRANSPARENT WHITE BACKGROUND --------
            try:
                bg_padding = 4   # pixels padding around text
                opacity = 255    # 0–255 (higher = more opaque)

                is_multiline = "\n" in text

                # For multiline, we avoid using anchor because some Pillow versions
                # do not support anchor for multiline_text / multiline_textbbox.
                use_anchor = (anchor is not None) and (not is_multiline)

                # 1) Compute text bounding box using same xy + (maybe) anchor
                if is_multiline:
                    # Try multiline_textbbox if available; fallback to simple estimate
                    if hasattr(draw, "multiline_textbbox"):
                        bbox = draw.multiline_textbbox(
                            xy, text, font=font
                        )
                    else:
                        # Fallback: approximate bounding box manually
                        lines = text.splitlines()
                        widths = []
                        heights = []
                        for line in lines:
                            l0, t0, r0, b0 = font.getbbox(line)
                            widths.append(r0 - l0)
                            heights.append(b0 - t0)
                        text_w = max(widths) if widths else 0
                        line_h = max(heights) if heights else fontsize
                        text_h = line_h * max(len(lines), 1)
                        # treat xy as top-left
                        left, top = xy
                        bbox = (left, top, left + text_w, top + text_h)
                else:
                    # single-line text
                    if use_anchor and hasattr(draw, "textbbox"):
                        bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
                    elif hasattr(draw, "textbbox"):
                        bbox = draw.textbbox(xy, text, font=font)
                    else:
                        l0, t0, r0, b0 = font.getbbox(text)
                        left, top = xy
                        bbox = (left, top, left + (r0 - l0), top + (b0 - t0))

                left, top, right, bottom = bbox
                bg_box = (
                    left - bg_padding,
                    top - bg_padding,
                    right + bg_padding,
                    bottom + bg_padding,
                )

                # 2) Create transparent overlay, draw white box on it
                pil_img = pil_img.convert("RGBA")
                overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
                ov_draw = ImageDraw.Draw(overlay)

                ov_draw.rectangle(bg_box, fill=(255, 255, 255, opacity))

                # 3) Composite overlay onto image
                pil_img.alpha_composite(overlay)

                # 4) Draw text on top of the box
                draw = ImageDraw.Draw(pil_img)
                if is_multiline:
                    # No anchor for multiline to avoid compatibility issues
                    draw.multiline_text(xy, text, fill=color, font=font)
                else:
                    if use_anchor:
                        draw.text(xy, text, fill=color, font=font, anchor=anchor)
                    else:
                        draw.text(xy, text, fill=color, font=font)

            except Exception as e:
                txt_e = f"Error in AddText while drawing text: {e}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 1}


            # -------- SAVE RESULT --------
            base_name = "image"
            if is_path:
                base_name = os.path.splitext(os.path.basename(image))[0]

            new_filename = f"{base_name}_AddedText{ext}"

            if self.save_path and os.path.isdir(self.save_path):
                save_path = os.path.join(self.save_path, new_filename)
            else:
                if self.save_path and not os.path.isdir(self.save_path):
                    logger.warning(
                        f"Save path '{self.save_path}' is not a valid directory. "
                        f"Falling back to default ./tools_output/"
                    )
                save_dir = os.path.join(os.getcwd(), "tools_output")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, new_filename)

            ####################################################################
            # Write out
            # - For GeoTIFF: go back to raster & write with rasterio to keep metadata
            # - For others: PIL save with same extension
            ####################################################################
            if is_path and ext in [".tif", ".tiff"] and profile is not None and raster_data is not None:
                try:
                    arr = np.array(pil_img)
                    # reshape back to (bands, height, width)
                    raster_out = reshape_as_raster(arr)

                    # Ensure band count matches profile
                    profile_out = profile.copy()
                    profile_out["count"] = raster_out.shape[0]

                    with rasterio.open(save_path, "w", **profile_out) as dst:
                        dst.write(raster_out)
                except Exception as e:
                    txt_e = f"Error writing GeoTIFF with rasterio: {e}"
                    logger.error(txt_e)
                    return {"text": txt_e, "error_code": 1}
            else:
                # Non-GeoTIFF or base64: just save via PIL
                try:
                    if ext in [".jpg", ".jpeg"] and pil_img.mode == "RGBA":
                        pil_img = pil_img.convert("RGB")
                    pil_img.save(save_path)
                    
                except Exception as e:
                    txt_e = f"Error saving image: {e}"
                    logger.error(txt_e)
                    return {"text": txt_e, "error_code": 1}

            txt = f"Annotated image saved to {new_filename}"
            logger.info(txt)
            return {"text": txt, "image": save_path, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in AddText: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "AddText",
                "description": "Draws text on an image at a specified position.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Path to the image (or base64-encoded image for non-GeoTIFF)."
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to add to the image. Can be multiline using '\\n'."
                        },
                        "position": {
                            "type": "string",
                            "description": "Coordinates `(x,y)` or position keyword like 'lt','mm','rb' "
                                           "which are combinations of ['l'(left), 'm'(middle), 'r'(right)] "
                                           "and ['t'(top), 'm'(middle), 'b'(bottom)]."
                        },
                        "color": {
                            "type": "string",
                            "description": "Text color (default: red)."
                        }
                    },
                    "required": ["image", "text", "position"],
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20005)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="AddText")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = AddTextWorker(
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

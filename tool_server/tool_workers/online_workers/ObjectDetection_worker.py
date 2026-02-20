import torch
import numpy as np
from PIL import Image
import base64
import argparse
import uuid
import os
from io import BytesIO

from mmdet.apis import DetInferencer
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"object_detection_lae_dino_worker_{worker_id}.log")


class ObjectDetectionWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_path=None,          # LAE-DINO checkpoint (.pth)
        model_config=None,        # LAE-DINO config (.py)
        model_name="ObjectDetection",
        load_8bit=False,
        load_4bit=False,
        device="cuda",
        limit_model_concurrency=5,
        host="0.0.0.0",
        port=None,
        model_semaphore=None,
        wait_timeout=120.0,
        task_timeout=30.0,
    ):
        self.model_path = model_path
        self.model_config = model_config

        # Optional: default open-vocab categories for remote sensing
        self.default_categories = [
            "helicopter",
            "roundabout",
            "soccerball field",
            "swimming pool",
            "helipad",
            "airplane",
            "airport",
            "groundtrack field",
            "harbor",
            "baseball field",
            "basketball court",
            "bridge",
            "storage tank",
            "tennis court",
            "small vehicle",
            "large vehicle",
            "trailer",
            "ship",
            "boat",
            "excavator",
            "building",
            "shed",
            "overpass",
            "container",
            "train station",

        ]

        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            None,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    # -------------------------------------------------------
    # Model init: LAE-DINO DetInferencer
    # -------------------------------------------------------
    def init_model(self):
        logger.info(f"Initializing LAE-DINO ObjectDetection model {self.model_name}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")

        if not (self.model_config and self.model_path):
            raise ValueError("model_config and model_path must be provided for LAE-DINO")
        print(self.model_path)
        self.inferencer = DetInferencer(
            model=self.model_config,    # config .py path
            weights=self.model_path,    # checkpoint .pth
            device=self.device,         # e.g. "cuda" or "cuda:0"
            palette="random",
        )

        logger.info("LAE-DINO DetInferencer initialized successfully.")

    # -------------------------------------------------------
    # Image loader (optional, if you need it)
    # -------------------------------------------------------
    def _load_image(self, image_input: str) -> Image.Image:
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        else:
            return Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")

    # -------------------------------------------------------
    # Main generate method
    # -------------------------------------------------------
    @torch.inference_mode()
    def generate(self, params):
        if "image" not in params:
            txt_e = "Missing required parameter: image"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        image_path = params.get("image")
        if not os.path.exists(image_path):
            txt_e = f"Image not found: {image_path}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}

        # Optional parameters
        user_text = params.get("text", None)
        top1 = bool(params.get("top1", False))
        box_threshold = float(params.get("box_threshold", 0.25))

        # If user doesn't give text, use default remote-sensing vocabulary
        if user_text is None:
            texts = " . ".join(self.default_categories)
        else:
            texts = user_text

        logger.info(f"Running LAE-DINO on image={image_path}")
        logger.info(f"texts={texts}")
        logger.info(f"box_threshold={box_threshold}, top1={top1}")

        try:
            result = self.inferencer(
                image_path,
                texts=texts,
                pred_score_thr=box_threshold,
                batch_size=1,
                out_dir="",
                no_save_vis=True,
                no_save_pred=True,
                print_result=False,
            )

            if "predictions" not in result or len(result["predictions"]) == 0:
                txt_e = "No predictions found in LAE-DINO output."
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            pred = result["predictions"][0]
            bboxes = np.array(pred.get("bboxes", []))
            scores = np.array(pred.get("scores", []))
            labels_idx = np.array(pred.get("labels", [])) if "labels" in pred else None

            if bboxes.size == 0:
                txt_e = "no detections found after thresholding"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            # Filter by score threshold
            keep = scores >= box_threshold
            bboxes = bboxes[keep]
            scores = scores[keep]
            if labels_idx is not None and labels_idx.size > 0:
                labels_idx = labels_idx[keep]

            if bboxes.size == 0:
                txt_e = "no detections found after filtering"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            # Prepare label names if possible
            # If `texts` is "cat . dog . car", map integer labels to these names.
            label_names = None
            if isinstance(texts, str):
                label_names = [s.strip() for s in texts.split(".") if s.strip()]

            # Optional: only keep top1 by score
            if top1:
                best_idx = int(scores.argmax())
                bboxes = bboxes[best_idx : best_idx + 1]
                scores = scores[best_idx : best_idx + 1]
                if labels_idx is not None and labels_idx.size > 0:
                    labels_idx = labels_idx[best_idx : best_idx + 1]

            detections_str_lines = []
            for i, (box, score) in enumerate(zip(bboxes, scores)):
                x1, y1, x2, y2 = box
                s = float(score)

                if labels_idx is not None and labels_idx.size > 0 and label_names is not None:
                    li = int(labels_idx[i])
                    if 0 <= li < len(label_names):
                        label = label_names[li]
                    else:
                        label = "object"
                else:
                    label = "object"

                # Format must match parse_tool_output():
                # "(x1, y1, x2, y2), label, score 0.87"
                detections_str_lines.append(
                    f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}), {label}, score {s:.2f}"
                )

            if not detections_str_lines:
                txt_e = "no detections after formatting"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            txt = "\n".join(detections_str_lines)
            return {"text": txt, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in ObjectDetection (LAE-DINO): {e}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 1}

    # -------------------------------------------------------
    # Tool schema (for controller / agents)
    # -------------------------------------------------------
    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "ObjectDetection",
                "description": (
                    "Detects objects in an image using LAE-DINO via MMDetection's DetInferencer. "
                    "Supports open-vocabulary text queries; if no text is provided, a default "
                    "remote-sensing vocabulary is used."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image path on disk.",
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Optional open-vocabulary query. Either a single phrase "
                                "like 'a red car' or a dot-separated list like "
                                "'airplane . ship . vehicle'. If omitted, a default "
                                "remote-sensing vocabulary is used."
                            ),
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, return only the highest-confidence detection.",
                        },
                        "box_threshold": {
                            "type": "number",
                            "description": "Score threshold for filtering detections (default 0.3).",
                        },
                    },
                    "required": ["image"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20007)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to LAE-DINO checkpoint (.pth)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to LAE-DINO config (.py)",
    )
    parser.add_argument("--model-name", type=str, default="ObjectDetection")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ObjectDetectionWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_path=args.model_path,
        model_config=args.model_config,
        model_name=args.model_name,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

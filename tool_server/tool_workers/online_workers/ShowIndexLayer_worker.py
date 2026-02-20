import argparse
import uuid
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ShowIndexLayer_worker_{worker_id}.log")


def ShowIndexLayer(gpkg, layer_name, index_type, out_file=None, cmap=None):
    """
    Generate a colorized PNG or JPEG preview from a raster layer in a GeoPackage.

    Parameters
    ----------
    gpkg : str
        Path to GeoPackage.
    layer_name : str
        Raster layer name (inside GeoPackage).
    index_type : str
        'NDVI', 'NDBI', or 'NBR' (used to choose default colormap).
    out_file : str, optional
        Output image path (.png or .jpg). Defaults to f"{layer_name}_preview.png".
    cmap : str or matplotlib colormap, optional
        Custom colormap name (e.g., 'RdYlGn'). If None, a default is chosen.

    Returns
    -------
    str
        Path to saved PNG/JPEG file.
    """

    # --- Load raster from GeoPackage ---
    ds = gdal.Open(f"{gpkg}|layername={layer_name}")
    if ds is None:
        raise Exception(f"Could not open layer '{layer_name}' from {gpkg}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    dtype_name = gdal.GetDataTypeName(band.DataType)
    already_float = dtype_name.startswith("Float")

    if not already_float:
        arr[arr <= 0] = np.nan  # Remove nodata or invalid pixels
        arr = ((arr - 1) / 254.0) * 2.0 - 1.0
    else:
        arr[arr <= -100] = np.nan

    arr = np.clip(arr, -1, 1)

    # --- Set default colormap based on index type ---
    index_type = index_type.upper()
    if cmap is None:
        if index_type == "NDVI":
            cmap = "RdYlGn"      # Red = low, Green = high
        elif index_type == "NDBI":
            cmap = "RdYlBu_r"    # Red = dense urban, Blue = low urban
        elif index_type == "NBR":
            cmap = "RdYlGn" # Red = burn, Green = recovery
        else:
            cmap = "viridis"

    # --- Normalize to [-1, 1] for visualization ---
    vmin, vmax = -1, 1
    arr = np.clip(arr, vmin, vmax)

    # --- Plot and save ---
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{index_type} Value")
    plt.title(f"{index_type} Layer: {layer_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    return f"Saved colorized {index_type} preview to: {out_file}"


class ShowIndexLayerWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="ShowIndexLayer",
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=5,
        model_semaphore=None,
        save_path = None,
        wait_timeout=120.0,
        task_timeout=120.0,
    ):
        self.save_path = save_path
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
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info("ShowIndexLayerWorker does not need a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

    def generate(self, params):
        required_keys = ("gpkg", "layer_name", "index_type")

        missing = [k for k in required_keys if k not in params]
        if missing:
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        layer_name = params.get("layer_name")
        index_type = params.get("index_type")
        cmap = params.get("cmap", None)
        
        if not os.path.exists(gpkg):
            txt_e = f"GeoPackage not found"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}
        
        try:
            # --- Create output path ---
            new_filename = f"{layer_name}_preview.png"
            if self.save_path and os.path.isdir(self.save_path):
                save_path = os.path.join(self.save_path, new_filename)
            else:
                if not os.path.isdir(self.save_path):
                    logger.warning(f"Save path '{self.save_path}' is not a valid directory. "
                                f"Falling back to default ./tools_output/")
                save_dir = os.path.join(os.getcwd(), "tools_output")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, new_filename)

            _ = ShowIndexLayer(
                gpkg=gpkg,
                layer_name=layer_name,
                index_type=index_type,
                out_file=save_path,
                cmap=cmap,
            )
            txt = f"Saved {index_type.upper()} preview for '{layer_name}' to {new_filename}"
            return {"text": txt, "image": save_path, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in ShowIndexLayer: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "ShowIndexLayer",
                "description": "Render a colorized preview image (PNG/JPEG) for an index raster layer in a GeoPackage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "index_type": {"type": "string", "enum": ["NDVI", "NDBI", "NBR"], "description": "Index type for default colormap."},
                        "layer_name": {"type": "string", "description": "Raster layer name inside the GeoPackage."},
                        "out_file": {"type": "string", "description": "Optional output image path (.png or .jpg)."},
                    },
                    "required": ["gpkg", "layer_name", "index_type"],
                },
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20112)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ShowIndexLayer")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ShowIndexLayerWorker(
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

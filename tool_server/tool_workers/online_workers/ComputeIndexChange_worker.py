import argparse
import uuid
import math, os
import numpy as np
import re

from qgis.core import (
    QgsRasterLayer,
    QgsRasterPipe,
    QgsRasterFileWriter,
)
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry

from osgeo import gdal

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ComputeIndexChange_worker_{worker_id}.log")


def ComputeIndexChange(gpkg, index_type, layer1_name, layer2_name, diff_layer_name=None):
    """
    Compute ΔIndex = layer2 - layer1, classify changes, and report statistics.
    Supports NDVI, NDBI, and NBR.

    Parameters
    ----------
    gpkg : str
        Path to GeoPackage file.
    index_type : str
        NDVI, NDBI or NBR
    layer1_name : str
        Name of first NDVI raster layer (baseline).
    layer2_name : str
        Name of second NDVI raster layer (comparison).
    diff_layer_name : str, optional
        Output layer name. Defaults to f"{index_type}_Change".

    Returns
    -------
    dict
        Dictionary with summary statistics and new layer name.
    """
    if not diff_layer_name:
        diff_layer_name = f"{index_type}_Change"

    layer1 = QgsRasterLayer(f"{gpkg}|layername={layer1_name}", layer1_name)
    layer2 = QgsRasterLayer(f"{gpkg}|layername={layer2_name}", layer2_name)

    if not layer1.isValid() or not layer2.isValid():
        raise Exception("One or both raster layers could not be loaded")

    e1 = QgsRasterCalculatorEntry()
    e1.ref, e1.raster, e1.bandNumber = "layer1@1", layer1, 1
    e2 = QgsRasterCalculatorEntry()
    e2.ref, e2.raster, e2.bandNumber = "layer2@1", layer2, 1
    entries = [e1, e2]

    tmp_tif = os.path.join(os.path.dirname(gpkg), f"{diff_layer_name}.tif")

    formula = "(((layer2@1 - 1) / 254.0 * 2.0 - 1.0) - ((layer1@1 - 1) / 254.0 * 2.0 - 1.0))"
    calc = QgsRasterCalculator(
        formula,
        tmp_tif,
        "GTiff",
        layer1.extent(),
        layer1.width(),
        layer1.height(),
        entries,
    )
    result = calc.processCalculation()
    if result != 0:
        raise Exception(f"RasterCalculator failed with code {result}")

    diff_raster = QgsRasterLayer(tmp_tif, diff_layer_name)

    # ----------------------------------------------------------------------
    # --- Define change classification thresholds ---
    # ----------------------------------------------------------------------
    if index_type.upper() == "NDVI":
        breaks = [-math.inf, -0.3, -0.1, 0.1, 0.3, math.inf]
        class_names = {
            1: "Severe vegetation loss",             # (< -0.3)
            2: "Moderate vegetation loss",          # (-0.3 – -0.1)
            3: "Stable /no significant change",     # (-0.1 – +0.1)
            4: "Moderate vegetation gain",          # (+0.1 – +0.3)
            5: "Strong vegetation gain / regrowth"  # (> +0.3)
        }

    elif index_type.upper() == "NDBI":
        breaks = [-math.inf, -0.3, -0.1, 0.1, 0.3, math.inf]
        class_names = {
            1: "Strong urban decrease",             # (< -0.3)
            2: "Moderate urban decrease",           # (-0.3 – -0.1)
            3: "Stable /no significant change",     # (-0.1 – +0.1)
            4: "Moderate urban growth",             # (+0.1 – +0.3)
            5: "Strong urban growth"                # (> +0.3)
        }

    elif index_type.upper() == "NBR":
        breaks = [-math.inf, -0.66, -0.27, -0.1, 0.1, math.inf]
        class_names = {
            1: "Severe burn Severity",    # (< -0.66)
            2: "Moderate burn Severity",  # (-0.66 – -0.27)
            3: "Low burn Severity",       # (-0.27 – -0.1)
            4: "Unburned",                # (-0.1 – +0.1)
            5: "Enhanced Regrowth"        # (> +0.1)
        }

    else:
        raise ValueError("index_type must be one of: 'NDVI', 'NDBI', or 'NBR'")

    ds = gdal.Open(tmp_tif)
    arr = ds.ReadAsArray().astype(np.float32)

    arr = arr[arr > -100]      # Drop -100 and any lower (e.g., -3.4e+38)
    total = arr.size
    summary = f"{index_type} Change Statistics: \n"
    for i in range(1, len(breaks)):
        count = np.sum((arr >= breaks[i-1]) & (arr < breaks[i]))
        pct = (count / total) * 100 if total > 0 else 0
        summary += f"  {class_names[i]:<45}: {pct:6.2f} % \n"
    #----------------------------------------------------------------
    if not diff_raster.isValid():
        raise Exception("Failed to create transition raster")

    pipe = QgsRasterPipe()
    provider = diff_raster.dataProvider().clone()
    pipe.set(provider)

    writer = QgsRasterFileWriter(f"{gpkg}|layername={diff_layer_name}")
    status = writer.writeRaster(
        pipe,
        diff_raster.width(),
        diff_raster.height(),
        diff_raster.extent(),
        diff_raster.crs()
    )
    gpkg_name = os.path.basename(gpkg)
    if status == QgsRasterFileWriter.NoError:
        msg = f"delta-{index_type} layer saved to {gpkg_name} as '{diff_layer_name}'\n"
    else:
        raise Exception("Error writing Δ raster")

    return msg + summary


class ComputeIndexChangeWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="ComputeIndexChange",
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=5,
        model_semaphore=None,
        wait_timeout=300.0,
        task_timeout=300.0,
    ):
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
        logger.info("ComputeIndexChangeWorker does not need a model. Ready to run.")

    def generate(self, params):
        required_keys = ("gpkg", "index_type", "layer1_name", "layer2_name")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        index_type = params.get("index_type")
        layer1_name = params.get("layer1_name")
        layer2_name = params.get("layer2_name")
        diff_layer_name = params.get("diff_layer_name", None)
        
        if not os.path.exists(gpkg):
                txt_e = f"GeoPackage not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
        
        try:
            txt = ComputeIndexChange(
                gpkg=gpkg,
                index_type=index_type,
                layer1_name=layer1_name,
                layer2_name=layer2_name,
                diff_layer_name=diff_layer_name,
            )
            cleaned_txt = re.sub(r' {2,}', ' ', txt).strip()
            return {"text": cleaned_txt, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in ComputeIndexChange: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "ComputeIndexChange",
                "description": "Compute ΔIndex (layer2 - layer1) for NDVI/NDBI/NBR from two GPKG raster layers and save output into the same GeoPackage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "index_type": {"type": "string", "enum": ["NDVI", "NDBI", "NBR"], "description": "Index type."},
                        "layer1_name": {"type": "string", "description": "Baseline raster layer name."},
                        "layer2_name": {"type": "string", "description": "Comparison raster layer name."},
                        "diff_layer_name": {"type": "string", "description": "Optional output layer name (defaults to '<index_type>_Change')."},
                    },
                    "required": ["gpkg", "index_type", "layer1_name", "layer2_name"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20111)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ComputeIndexChange")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ComputeIndexChangeWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

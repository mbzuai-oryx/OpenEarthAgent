import argparse
import uuid
from datetime import datetime
import os
import ee
import geemap
import geopandas as gpd
import re

# QGIS classes used by your function
from qgis.core import QgsRasterLayer, QgsRasterPipe, QgsRasterFileWriter

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

CRS = "EPSG:4326"

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"AddIndexLayer_worker_{worker_id}.log")

def to_u8_from_neg1_1(img, name):
    """Scale a [-1,1] float image to [1,255] uint8. reserving 0 for nodata"""
    u8 = (img.add(1).divide(2).multiply(254)).add(1).clamp(1, 255).round().toUint8()
    return u8.rename(name)

def AddIndexLayer(gpkg, index_type, layer_name, year, month= None, scale=30):
    """
    Compute Spectral Index : NDVI, NDBI, or NBR and their class percentages for the given year.
    Export a raster layer into a GeoPackage and report summary.

    Parameters
    ----------
    gpkg : str
        Path to GeoPackage containing an 'area_boundary' layer.
    index_type : str
        One of 'NDVI', 'NDBI', or 'NBR'.
    layer_name : str
        Output raster layer name.
    year : int
        Year to process (e.g. 2022).
    scale : int
        Pixel resolution in meters.
    """
    # --- Load AOI from GeoPackage ---
    area_gdf = gpd.read_file(gpkg, layer="area_boundary")
    geom = area_gdf.geometry.iloc[0]
    aoi_geojson = geom.__geo_interface__
    aoi = ee.Geometry(aoi_geojson)
    if month is None:
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
    else:
        # Compute first and last day of month
        start_date = datetime(year, month, 1).strftime("%Y-%m-%d")
        if month == 12:
            end_date = datetime(year + 1, 1, 1).strftime("%Y-%m-%d")
        else:
            end_date = datetime(year, month + 1, 1).strftime("%Y-%m-%d")

    # --- Sentinel-2 SR median composite ---
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
          .filterBounds(aoi)
          .filterDate(start_date, end_date)
          .select(['B4','B8','B11','B12'])
          .median()
          .clip(aoi))

    # --- Compute index ---
    if index_type == "NDVI":
        img = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
        expr = (
            "b(0) < 0.0 ? 1 : "
            "b(0) < 0.1 ? 2 : "
            "b(0) < 0.3 ? 3 : "
            "b(0) < 0.6 ? 4 : "
            "5"
        )
        class_names = {
            1: "Very Low - barren/water (<0.0)",
            2: "Low - bare soil (0.0–0.1)",
            3: "Sparse to no vegetation(0.1–0.3)",
            4: "Moderate vegetation (0.3–0.6)",
            5: "High vegetation (>0.6)"
        }

    elif index_type == "NDBI":
        img = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
        expr = (
            "b(0) < 0.0 ? 1 : "
            "b(0) < 0.2 ? 2 : "
            "b(0) < 0.3 ? 3 : "
            "4"
        )
        class_names = {
            1: "Non-built-up - water/vegetation (<0.0)",
            2: "Transitional - mixed use (0.0–0.2)",
            3: "Moderate built-up (0.2–0.3)",
            4: "Dense built-up (>0.3)"
        }

    elif index_type == "NBR":
        img = s2.normalizedDifference(["B8", "B12"]).rename("NBR")
        expr = (
            "b(0) < -0.5 ? 1 : "
            "b(0) < -0.1 ? 2 : "
            "b(0) < 0.1 ? 3 : "
            "b(0) < 0.5 ? 4 : "
            "5"
        )
        class_names = {
            1: "Very Low (<-0.5)",
            2: "Low (-0.5 – -0.1)",
            3: "Moderate (-0.1 – +0.1)",
            4: "High (0.1 – 0.5)",
            5: "Very High (> 0.5)"
        }

    else:
        raise ValueError("index_type must be one of: 'NDVI', 'NDBI', 'NBR'.")

    # --- Classify ---
    class_img = img.expression(expr, {"b(0)": img}).rename(f"{index_type}_class")

    # --- Compute per-class pixel counts (server-side) ---
    class_count = class_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=scale,
        bestEffort=True,
        maxPixels=1e13
    )

    counts = ee.Dictionary(class_count.get(f"{index_type}_class"))
    total = counts.values().reduce(ee.Reducer.sum())

    # Compute percentages for each class on the server
    percentages = counts.map(lambda k, v: ee.Number(v).divide(total).multiply(100))

    summary = f"\n{index_type} Class Percentage Statistics ({year}):\n"

    # Fetch the server results in one small call (safe: just 5 classes)
    pct_dict = percentages.getInfo()
    for cls, name in class_names.items():
        pct = pct_dict.get(str(cls), 0.0)
        summary += f"  {name:<35}: {pct:6.2f} %\n"
    
    u8_img = to_u8_from_neg1_1(img, f"{index_type}_u8")
    # --- Optional: export raster locally ---
    tmp_tif = os.path.join(os.path.dirname(gpkg), f"{index_type}_{year}_U8.tif")
    
    geemap.ee_export_image(
        u8_img,
        filename=tmp_tif,
        scale=scale,
        region=aoi,
        file_per_band=False,
        crs="EPSG:4326"
    )
    
    raster = QgsRasterLayer(tmp_tif, layer_name)
    if not raster.isValid():
        raise Exception(f"Failed to load raster {tmp_tif}")

    # --- Write raster into GeoPackage as a new layer ---
    pipe = QgsRasterPipe()
    provider = raster.dataProvider().clone()
    pipe.set(provider)

    writer = QgsRasterFileWriter(f"{gpkg}|layername={layer_name}")
    success = writer.writeRaster(
        pipe,
        raster.width(),
        raster.height(),
        raster.extent(),
        raster.crs()
    )

    if success == QgsRasterFileWriter.NoError:
        msg = f"{index_type} raster for {year} saved to {os.path.basename(gpkg)} as layer '{layer_name}'\n"
    else:
        raise Exception("Error writing raster")

    return msg + summary


class AddIndexLayerWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="AddIndexLayer",
        host="0.0.0.0",
        port=None,
        limit_model_concurrency=5,
        model_semaphore=None,
        service_account=None,
        key_file=None,
        wait_timeout=300.0,
        task_timeout=300.0,
    ):
        self.service_account=service_account
        self.key_file=key_file
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
        if not self.service_account or not self.key_file:
            txt_e = "Missing service account or key file path."
            logger.error(txt_e)
            raise ValueError(txt_e)

        if not os.path.exists(self.key_file):
            txt_e = f"Key file not found: {self.key_file}"
            logger.error(txt_e)
            raise FileNotFoundError(txt_e)
        try:
            credentials = ee.ServiceAccountCredentials(self.service_account, self.key_file)
            ee.Initialize(credentials)
            try:
                ee.Number(1).getInfo()  # simple test call
                logger.info("Earth Engine authenticated using service account.")
                logger.info("Initialization successful. Ready to run.")
            except Exception as e:
                txt_e = f"Earth Engine initialized but test call failed: {e}"
                logger.error(txt_e)
                raise RuntimeError(txt_e)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Earth Engine: {e}")
            raise

    def generate(self, params):
        required_keys = ("gpkg", "index_type", "layer_name", "year")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        index_type = params.get("index_type")
        layer_name = params.get("layer_name")
        year = params.get("year")
        month = params.get("month", None)
        scale = 30 # default

        if not os.path.exists(gpkg):
                txt_e = f"GeoPackage not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
        
        try:
            txt = AddIndexLayer(
                gpkg=gpkg,
                index_type=index_type,
                layer_name=layer_name,
                year=int(year),
                month=month if month is None else int(month),
                scale=int(scale),
            )
            cleaned_txt = re.sub(r' {2,}', ' ', txt).strip()
            return {"text": cleaned_txt, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in AddIndexLayer: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "AddIndexLayer",
                "description": "Compute NDVI/NDBI/NBR for the given year (optionally month) over the AOI in a GeoPackage and save the raster layer to the GeoPackage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage (must contain 'area_boundary' layer)."},
                        "index_type": {"type": "string", "enum": ["NDVI", "NDBI", "NBR"], "description": "Spectral index to compute."},
                        "layer_name": {"type": "string", "description": "Output layer name to save into the GeoPackage."},
                        "year": {"type": "integer", "description": "Year to process, e.g., 2022."},
                        "month": {"type": "integer", "minimum": 1, "maximum": 12, "description": "Optional month (1–12). If omitted, uses the whole year."},
                    },
                    "required": ["gpkg", "index_type", "layer_name", "year"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20110)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="AddIndexLayer")
    parser.add_argument("--service_account", type=str, default="")
    parser.add_argument("--key_file", type=str, default="")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = AddIndexLayerWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        service_account=args.service_account,
        key_file=args.key_file,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

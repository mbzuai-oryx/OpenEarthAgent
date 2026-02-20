import argparse
import uuid
import re
import os
from datetime import datetime
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

CRS = "EPSG:4326"

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"GetAreaBoundary_worker_{worker_id}.log")


def is_valid_bbox(bbox):
    """Validate bbox = (west, south, east, north)."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False
    west, south, east, north = bbox
    if not all(isinstance(v, (int, float)) for v in bbox):
        return False
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        return False
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        return False
    if not (west < east and south < north):
        return False
    return True


def get_area_boundary_as_gpkg(area, buffer_m=None, save_path=None):
    """
    Create a GeoPackage with the area boundary saved as 'area_boundary' layer.

    Parameters
    ----------
    area : str or tuple
        - If str: place name (e.g., "San Francisco, USA")
        - If tuple: bounding box in (west, south, east, north) order
    buffer_m : int, optional
        Buffer distance in meters around the geometry (default: None)

    Returns
    -------
    str, GeoDataFrame
        Path to saved GeoPackage and the boundary GeoDataFrame
    """
    out = f"aoi_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
    # out = f"aoi.gpkg" #for debug
    save_path = os.path.join(save_path, out)

    # Case 1: place name
    if isinstance(area, str):
        gdf = ox.geocode_to_gdf(area).to_crs(CRS)
        if gdf.empty:
            raise ValueError(f"Place '{area}' not found in OSM.")
        geom, name = gdf.geometry.union_all(), area

    # Case 2: bounding box tuple
    elif is_valid_bbox(area):
        geom, name = box(*area), f"{area}"
        if geom.is_empty:
            raise ValueError(f"Bounding box {area} produced an empty geometry.")

    else:
        raise ValueError("Area must be a place name (str), bbox tuple (west, south, east, north)")

    # Apply buffer if requested
    if buffer_m and buffer_m > 0:
        gseries = gpd.GeoSeries([geom], crs=CRS).to_crs("EPSG:3857")
        geom = gseries.buffer(buffer_m).to_crs(CRS).iloc[0]
        name = f"{name}_buffer{buffer_m}m"

    gdf = gpd.GeoDataFrame(
        {"name": [name], "created_at": [datetime.now()], "year": [datetime.now().year]},
        geometry=[geom], crs=CRS
    )

    try:
        gdf.to_file(save_path, layer="area_boundary", driver="GPKG")
    except Exception as e:
        raise IOError(f"Error saving GeoPackage '{out}': {e}")

    return out, save_path


class GetAreaBoundaryWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="GetAreaBoundary",
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
            None,  # no model path
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
        logger.info("GetAreaBoundary_Worker does not need a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

    def generate(self, params):
        if "area" not in params:
            txt_e = "Missing required parameter: area"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        area = params.get("area")
        buffer_m = params.get("buffer_m")
    
        try:
            if isinstance(area, str):
                bbox_pattern = r"^[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]$"
                match = re.match(bbox_pattern, area.strip())
                if match:
                    try:
                        area = tuple(map(float, match.groups()))
                    except ValueError:
                        pass

            if self.save_path and os.path.isdir(self.save_path):
                save_path = self.save_path
            else:
                if not os.path.isdir(self.save_path):
                    logger.warning(f"Save path '{self.save_path}' is not a valid directory. "
                                f"Falling back to default ./tools_output/")
                save_dir = os.path.join(os.getcwd(), "tools_output")
                os.makedirs(save_dir, exist_ok=True)
                save_path = save_dir

            gpkg_name, save_path_full = get_area_boundary_as_gpkg(area, buffer_m, save_path)

            txt = f"Saved boundary to {gpkg_name}"
            return {"text": txt, "gpkg": save_path_full, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in GetAreaBoundary: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "GetAreaBoundary",
                "description": "Generates a GeoPackage containing the boundary of a given area (place name or bbox).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string", "description": "Place name or bbox (west,south,east,north)."},
                        "buffer_m": {"type": "number", "description": "Buffer distance in meters (optional)."}
                    },
                    "required": ["area"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20006)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="GetAreaBoundary")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = GetAreaBoundaryWorker(
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

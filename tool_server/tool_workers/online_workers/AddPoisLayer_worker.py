import argparse
import uuid
import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from datetime import datetime

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

CRS = "EPSG:4326"

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"AddPoisLayer_worker_{worker_id}.log")


def clean_for_gpkg(gdf):
    gdf = gdf.copy()
    gdf.columns = gdf.columns.str.lower().str[:30]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf

def get_name_row(row):
    """Return best display name for OSM features in GeoDataFrame (no pandas)."""
    fields = row.index
    # Priority: name:en > name
    for key in ("name:en", "name"):
        if key in fields:
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    # Any other name:* field
    for key in fields:
        if key.startswith("name"):
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    return ""

def add_pois_layer(gpkg, query, layer_name):
    """
    Add POIs into GeoPackage from OSM
    Ensures features fall inside boundary polygon stored in layer "area_boundary".

    Parameters
    ----------
    gpkg : str
        Path to GeoPackage
    query : dict or str
        - dict: tags (e.g., {"amenity": "hospital"})
        - str: single POI name (e.g., "Charité Hospital, Berlin")
    layer_name : str
        New layer name to save in GeoPackage
    
    Returns
    -------
    str
        Name of the created layer.
    """
    # Load boundary
    area_gdf = gpd.read_file(gpkg, layer="area_boundary")
    geom = area_gdf.geometry.iloc[0]

    if isinstance(query, dict):  # tags
        if len(query) != 1:
            raise ValueError(f"Query must contain exactly one key, got: {list(query.keys())}")
        key, value = next(iter(query.items()))
        if not (
            isinstance(value, (str, bool)) or
            (isinstance(value, list) and all(isinstance(v, str) for v in value))
        ):
            raise ValueError(
                f"Invalid query value for key '{key}': "
                f"must be str, bool, or list[str], got {type(value).__name__}"
            )

        pois = ox.features_from_polygon(geom, query).to_crs(CRS)

    elif isinstance(query, str):  # name
        pois = ox.geocode_to_gdf(query).to_crs(CRS)
        pois = pois[pois.geometry.within(geom)]
        if pois.empty:
            raise ValueError(f"POI '{query}' not found inside study area.")

    else:
        raise ValueError("query must be dict or str for OSM")
    
    pois["display_name"] = pois.apply(get_name_row, axis=1)
    pois = pois[pois["display_name"] != ""]

    if pois.empty:
        raise ValueError(f"No POIs found for {query} inside boundary.")
    
    # Deduplicate by name → keep first occurrence
    pois = pois.drop_duplicates(subset="display_name", keep="first")
    
    # Clean and save
    pois = clean_for_gpkg(pois)
    pois.to_file(gpkg, layer=layer_name, driver="GPKG")
    return layer_name, len(pois)

class AddPoisLayerWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="AddPoisLayer",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
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
        logger.info("AddPoisLayerWorker does not need a model. Ready to run.")

    def generate(self, params):
        required_keys = ("gpkg", "query", "layer_name")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        query = params.get("query")
        layer_name = params.get("layer_name")
        
        if not os.path.exists(gpkg):
            txt_e = f"GeoPackage not found"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}
        
        try:
            layer,len = add_pois_layer(gpkg, query, layer_name)
            gpkg_name = os.path.basename(gpkg)
            txt = f"Saved {len} POIs to layer '{layer}' in {gpkg_name}"
            return {"text": txt, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in AddPoisLayer: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}


    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "AddPoisLayer",
                "description": "Adds Points of Interest (POIs) into a GeoPackage inside the area boundary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "query": {"type": "object", "description": "OSM query as dict or POI name as string."},
                        "layer_name": {"type": "string", "description": "Layer name for saving POIs."}
                    },
                    "required": ["gpkg", "query", "layer_name"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20018)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="AddPoisLayer")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = AddPoisLayerWorker(
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

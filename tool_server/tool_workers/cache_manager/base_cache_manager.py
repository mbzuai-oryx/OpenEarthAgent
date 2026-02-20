import os
import re
import json
from pathlib import Path
import geopandas as gpd
from datetime import datetime
from qgis.core import (
    QgsRasterLayer,
    QgsRasterPipe,
    QgsRasterFileWriter
)
from tool_server.utils.server_utils import build_logger

logger = build_logger("cache_manager")

class CacheManager(object):
    def __init__(self, cache_dir = "", save_dir = ""):
        if not cache_dir:
            raise ValueError("cache_dir must be provided and cannot be empty")
        if not save_dir:
            raise ValueError("save_dir must be provided and cannot be empty")
        self.cache_dir = cache_dir
        self.save_dir = save_dir
        logger.info(f"CacheManager is initialized.")
    
    def normalize_name(self,s: str) -> str:
        return re.sub(r"[\W_]+", "", str(s).lower())
    
    def get_response(self, item, tool_name, params):
        ret_message = {"text": f"Failed to get cached results for {tool_name}", "error_code": 1}
        try:
            if tool_name  == "GetAreaBoundary":
                item.cached_layers = {}
                item.cached_gpkg = None
                area     = params["area"]
                buffer_m = params.get("buffer_m")
                cache_id = item.meta_data["sample_num"]
                expected_name = f"{area}_buffer{buffer_m}m" if buffer_m and buffer_m > 0 else f"{area}"
                out = f"aoi_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
                save_path = os.path.join(self.save_dir, out)
                cached_gpkg = None
                cached_gdf = None
                cache_id = str(cache_id)
                paths = []
                paths.extend(Path(self.cache_dir).glob(f"{cache_id}.gpkg"))
                paths.extend(Path(self.cache_dir).glob(f"{cache_id}_*.gpkg"))
                for existing_gpkg in paths:
                    try:
                        gdf = gpd.read_file(existing_gpkg, layer="area_boundary")
                    except Exception:
                        continue                       
                    if (gdf.empty or "name" not in gdf.columns or self.normalize_name(gdf.iloc[0]["name"]) != self.normalize_name(expected_name)):
                        logger.info(f"[{cache_id}] boundary missmatch: {gdf.iloc[0]['name']} : {expected_name}")
                        continue
                    cached_gpkg = str(existing_gpkg)
                    cached_gdf = gdf
                    break
                if cached_gpkg:
                    item.cached_gpkg = cached_gpkg
                    cached_gdf.to_file(save_path, layer="area_boundary", driver="GPKG")
                    txt = f"Saved boundary to {out}"
                    return {"text": txt, "gpkg": save_path, "error_code": 0}

            elif tool_name  == "AddPoisLayer": 
                if item.cached_gpkg:
                    in_gpkg = params["gpkg"]
                    in_query = params["query"]
                    in_layer_name = params["layer_name"]
                    gdf = gpd.read_file(item.cached_gpkg,layer="area_boundary")
                    key = json.dumps(in_query)
                    if key not in gdf.columns:
                        logger.info(f"AddPoisLayer Query mis-match")
                        sample_num = item.meta_data["sample_num"]
                        logger.info(f"[{sample_num}] {key} : {gdf.columns}")
                        return ret_message  
                    cached_layer_name = gdf[key].iloc[0]
                    gdf = gpd.read_file(item.cached_gpkg, layer=cached_layer_name)
                    gdf.to_file(in_gpkg, layer=in_layer_name, driver="GPKG")
                    item.cached_layers[in_layer_name] = cached_layer_name
                    txt = f"Saved {len(gdf)} POIs to layer '{in_layer_name}' in {os.path.basename(in_gpkg)}"
                    return {"text": txt, "error_code": 0}
            
            elif tool_name == "ComputeDistance":
                in_gpkg = params["gpkg"]
                src_layer = params["src_layer"]
                tar_layer = params["tar_layer"]
                top = params.get("top", None)
                if item.cached_gpkg and src_layer in item.cached_layers and tar_layer in item.cached_layers:
                    src_layer_or = item.cached_layers[src_layer]
                    tar_layer_or = item.cached_layers[tar_layer]
                    cache_dist_layer =f"{src_layer_or}_to_{tar_layer_or}_distances"
                    gdf = gpd.read_file(item.cached_gpkg,layer="area_boundary")
                    if cache_dist_layer not in gdf.columns:
                        logger.info(f"ComputeDistance dist_layer mis-match")
                        sample_num = item.meta_data["sample_num"]
                        logger.info(f"[{sample_num}] {cache_dist_layer} : {gdf.columns}")
                        return ret_message
                    cached_top = json.loads(gdf[cache_dist_layer].iloc[0])["top"]
                    if cached_top == top:
                        layer_name = f"{src_layer}_to_{tar_layer}_distances"
                        gdf = gpd.read_file(item.cached_gpkg, layer=f"{src_layer_or}_to_{tar_layer_or}_distances")
                        gdf.to_file(in_gpkg, layer=layer_name, driver="GPKG")
                        out_lines = []
                        distances = []
                        travel_times = []
                        out_lines.append(f"Distances (in meters) saved to line layer: '{layer_name}': ")
                        for _, r in gdf.iterrows():
                            txt = f"{r[src_layer]} , {r[tar_layer]}, distance={r['distance_m']:.2f} m"
                            distances.append(round(r["distance_m"], 2))
                            if "travel_time_s" in gdf.columns and not gdf["travel_time_s"].isna().all():
                                txt += f", travel_time={r['travel_time_s']:.1f} s"
                                travel_times.append(round(r["travel_time_s"], 1))
                            out_lines.append(txt)
                        out_lines.append(f"distances = {distances}")
                        if travel_times:
                            out_lines.append(f"travel_times = {travel_times}")
                        item.cached_layers[layer_name] = cache_dist_layer
                        text =  "\n".join(out_lines)
                        return {"text": text, "error_code": 0}
                        
            elif tool_name == "AddIndexLayer":
                in_gpkg = params["gpkg"]
                index_type = params["index_type"]
                layer_name = params["layer_name"]
                year = params["year"]
                month = params.get("month", None)
                if item.cached_gpkg:
                    name = f"{index_type}_{year}_{month}" if month is not None else f"{index_type}_{year}"
                    gdf = gpd.read_file(item.cached_gpkg,layer="area_boundary")
                    if name not in gdf.columns:
                        logger.info(f"AddIndexLayer Args mis-match")
                        sample_num = item.meta_data["sample_num"]
                        logger.info(f"[{sample_num}] {name} : {gdf.columns}")
                        return ret_message
                    meta = json.loads(gdf[name].iloc[0])
                    cached_layer_name = meta["layer_name"]
                    cached_layer_summary = meta["layer_summary"]
                    msg = f"{index_type} raster for {year} saved to {os.path.basename(in_gpkg)} as layer '{layer_name}'\n{index_type} "
                    cached_layer = QgsRasterLayer(f"{item.cached_gpkg}|layername={cached_layer_name}", cached_layer_name)
                    if not cached_layer.isValid():
                        raise RuntimeError(f"Invalid cached raster layer: {cached_layer_name}")
                    pipe = QgsRasterPipe()
                    provider = cached_layer.dataProvider().clone()
                    pipe.set(provider)
                    writer = QgsRasterFileWriter(f"{in_gpkg}|layername={layer_name}")
                    writer.writeRaster(
                        pipe,
                        cached_layer.width(),
                        cached_layer.height(),
                        cached_layer.extent(),
                        cached_layer.crs()
                    )
                    item.cached_layers[layer_name] = cached_layer_name
                    return {"text": msg + cached_layer_summary, "error_code": 0}

            elif tool_name == "ComputeIndexChange":
                in_gpkg = params["gpkg"]
                index_type = params["index_type"]
                layer1_name = params["layer1_name"]
                layer2_name = params["layer2_name"]
                diff_layer_name = params.get("diff_layer_name", f"{index_type}_Change")
                if item.cached_gpkg and layer1_name in item.cached_layers and layer2_name in item.cached_layers:
                    cached_layer1_name = item.cached_layers[layer1_name]
                    cached_layer2_name = item.cached_layers[layer2_name]
                    name = f"{index_type}-{cached_layer1_name}-{cached_layer2_name}"
                    gdf = gpd.read_file(item.cached_gpkg,layer="area_boundary")
                    if name not in gdf.columns:
                        logger.info(f"ComputeIndexChange Args mis-match")
                        sample_num = item.meta_data["sample_num"]
                        logger.info(f"[{sample_num}] {name} : {gdf.columns}")
                        return ret_message
                    meta = json.loads(gdf[name].iloc[0])
                    cached_layer_name = meta["layer_name"]
                    cached_layer_summary = meta["layer_summary"]
                    msg = f"delta-{index_type} layer saved to {os.path.basename(in_gpkg)} as '{diff_layer_name}'\n {index_type}"
                    cached_layer = QgsRasterLayer(f"{item.cached_gpkg}|layername={cached_layer_name}", cached_layer_name)
                    if not cached_layer.isValid():
                        raise RuntimeError(f"Invalid cached raster layer: {cached_layer_name}")
                    pipe = QgsRasterPipe()
                    provider = cached_layer.dataProvider().clone()
                    pipe.set(provider)
                    writer = QgsRasterFileWriter(f"{in_gpkg}|layername={diff_layer_name}")
                    writer.writeRaster(
                        pipe,
                        cached_layer.width(),
                        cached_layer.height(),
                        cached_layer.extent(),
                        cached_layer.crs()
                    )
                    item.cached_layers[diff_layer_name] = cached_layer_name
                    return {"text": msg + cached_layer_summary, "error_code": 0}
            return ret_message
        except Exception as e:
            logger.error(f"Failed to get cached results for {tool_name}: {e}")
            return ret_message
      

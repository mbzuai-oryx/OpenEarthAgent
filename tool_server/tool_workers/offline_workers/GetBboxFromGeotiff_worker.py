import os
import argparse
import rasterio
from rasterio.warp import transform_bounds

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

logger = build_logger("geotiff_bbox_worker")

import rasterio
from rasterio.warp import transform_bounds


def GetBboxFromGeotiff(geotiff, dst_crs_authid="EPSG:4326"):
    """
    Extract bbox (west, south, east, north) from a GeoTIFF file, in dst_crs_authid.
    """
    try:
        with rasterio.open(geotiff) as src:
            if src.crs is None:
                raise ValueError(f"Input GeoTIFF has no CRS: {geotiff}")
            b = src.bounds
            west, south, east, north = (
                (b.left, b.bottom, b.right, b.top)
                if src.crs.to_string() == dst_crs_authid
                else transform_bounds(src.crs, dst_crs_authid, b.left, b.bottom, b.right, b.top, densify_pts=21)
            )
    except Exception:
        raise IOError(f"Cannot load raster: {geotiff}")

    bbox = tuple(round(v, 4) for v in (west, south, east, north))
    print(f"Extracted Bbox (W, S, E, N): {bbox}")
    return bbox

def generate(params):
    """
    Expected params:
    - "geotiff": path to GeoTIFF on disk (required)
    """
    if "geotiff" not in params: 
        txt_e = "Missing required parameter: geotiff"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 2}
    
    geotiff_path = params.get("geotiff")
    if not os.path.isfile(geotiff_path):
        txt_e = f"GeoTIFF not found: {geotiff_path}"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 3}

    try:
        bbox = GetBboxFromGeotiff(geotiff_path)
        txt = f"Bbox (west, south, east, north): {bbox}"
        return {"text": txt, "error_code": 0}
    
    except Exception as e:
        txt_e = f"Error extracting bbox from GeoTIFF: {e}"
        logger.exception(txt_e)
        return {"text": txt_e, "error_code": 1}

if __name__ == "__main__":
    # simple CLI for debugging the worker standalone
    parser = argparse.ArgumentParser()
    parser.add_argument("--geotiff", type=str, required=True, help="Path to GeoTIFF file.")
    parser.add_argument("--dst-crs", type=str, default="EPSG:4326")
    args = parser.parse_args()

    params = {"geotiff": args.geotiff, "dst_crs": args.dst_crs}
    out = generate(params)
    print(out)

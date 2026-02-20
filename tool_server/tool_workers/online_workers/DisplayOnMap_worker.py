import argparse
import uuid
import os
import random
from datetime import datetime
import pandas as pd 
import fiona
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

CRS = "EPSG:4326"
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(
    BASE_DIR,
    "fonts",
    "NotoSansCJK-Regular.ttc"
)
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path)
else:
    font_prop = None
    print("font not found")

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"DisplayOnMap_worker_{worker_id}.log")

def get_name_row(row):
    """Return best display name for OSM features in GeoDataFrame (no pandas)."""
    fields = row.index
    for key in ("name:en", "name"):
        if key in fields:
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    for key in fields:
        if key.startswith("name"):
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    return ""

def DisplayOnMap(gpkg, layers, out_path):

    line_color_col="distance_m"
    line_cmap="plasma"
    basemap=True
    seed=42
    show_names = True

    if isinstance(layers, str): layers = [layers]
    if seed: random.seed(seed)

    # validate layers
    available = fiona.listlayers(gpkg)
    missing = [l for l in layers if l not in available]
    if missing: raise ValueError(f"Missing layers: {missing}. Available: {available}")
    if "area_boundary" not in available: raise ValueError('"area_boundary" layer not found in gpkg')
    area_gdf = gpd.read_file(gpkg, layer="area_boundary", engine="fiona")
    if area_gdf.empty:
        raise ValueError('"area_boundary" layer is empty')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    layer_colors = {}
    legend_added = set()
    def get_label(layer_name):
        if layer_name not in legend_added:
            legend_added.add(layer_name)
            return layer_name
        return None
    for layer in layers:
        gdf = gpd.read_file(gpkg, layer=layer, engine="fiona")
        if gdf.empty:
            continue
        layer_colors.setdefault(layer, "#" + "".join(random.choices("0123456789ABCDEF", k=6)))
        layer_color = layer_colors[layer]
        gtypes = gdf.geometry.geom_type
        lines   = gdf[gtypes.str.contains("Line",    na=False)]
        points  = gdf[gtypes.str.contains("Point",   na=False)]
        polys   = gdf[gtypes.str.contains("Polygon", na=False)]

        # ---- LINES ----
        if not lines.empty:
            has_col = line_color_col in lines.columns
            lines.plot( ax=ax,column=line_color_col if has_col else None,
                cmap=line_cmap if has_col else None,linewidth=2,alpha=0.8,legend=False,label=get_label(layer))
        
        # ---- POINTS + NAMES ----
        if not points.empty:
            points.plot(ax=ax, color=layer_color,markersize=50,alpha=0.9,label=get_label(layer))
            if show_names:
                for idx, row in points.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    if not isinstance(geom, Point):
                        geom = geom.representative_point()
                    name = get_name_row(row)
                    ax.text(
                        geom.x, geom.y, name, fontproperties=font_prop,
                        fontsize=8, color="black",
                        ha="left", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
                    )

        # ---- POLYGONS + NAMES ----
        if not polys.empty:
            polys.boundary.plot(ax=ax, linewidth=1.2, edgecolor=layer_color, label=get_label(layer))
            polys.plot(ax=ax, alpha=0.05, facecolor=layer_color)

            if show_names:
                for idx, row in polys.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    name = get_name_row(row)
                    cp = geom.representative_point()
                    ax.text(
                        cp.x, cp.y, name, fontproperties=font_prop,
                        fontsize=8, color="black",
                        ha="center", va="center",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
                    )

    minx, miny, maxx, maxy = area_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    if basemap: ctx.add_basemap(ax, crs=area_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    ax.legend(loc="upper left")
    ax.set_axis_off()
    # Output path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"map_{ts}.png"
    out_path = os.path.join(out_path,out_file)
        
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_file),out_path


class DisplayOnMapWorker(BaseToolWorker):
    def __init__(
            self,
            controller_addr,
            worker_addr="auto",
            worker_id=worker_id,
            no_register=False,
            model_name="DisplayOnMap",
            host="0.0.0.0",
            port=None,
            limit_model_concurrency=5,
            model_semaphore=None,
            save_path = None,
            wait_timeout=120.0,
            task_timeout=60.0,
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
        logger.info("DisplayOnMapWorker does not need a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

    def generate(self, params):
        required_keys = ("gpkg", "layers")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        layers = params.get("layers")
        
        if not os.path.exists(gpkg):
            txt_e = f"GeoPackage not found"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}
        
        try:
            if self.save_path and os.path.isdir(self.save_path):
                save_path = self.save_path
            else:
                if not os.path.isdir(self.save_path):
                    logger.warning(f"Save path '{self.save_path}' is not a valid directory. "
                                f"Falling back to default ./tools_output/")
                save_dir = os.path.join(os.getcwd(), "tools_output")
                os.makedirs(save_dir, exist_ok=True)
                save_path = save_dir

            out_file, out_path = DisplayOnMap(
                gpkg=gpkg,
                layers=layers,
                out_path=save_path
            )

            layer_list = ", ".join(layers if isinstance(layers, list) else [layers])
            txt = f"Rendered layers [{layer_list}] and saved map to {out_file}"
            return {"text": txt, "image": out_path, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in DisplayOnMap: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "DisplayOnMap",
                "description": "Render GeoPackage layer(s) to a PNG map with optional web basemap.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "layers": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ],
                            "description": "Layer name or list of layer names to render."
                        },
                    },
                    "required": ["gpkg", "layers"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20102)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="DisplayOnMap")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = DisplayOnMapWorker(
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

import os
import argparse
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
import geopandas as gpd
import numpy as np
import pandas as pd
import fiona
from datetime import datetime
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

logger = build_logger("geotiff_display_worker")
OUTPUT_DIR = "./tool_server/tool_workers/tools_output"
PIN_PATH = "./tool_server/tool_workers/offline_workers/pin.png"

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

def DisplayOnGeotiff(gpkg, layers, geotiff, show_names=True):
    """
        overlay vector layers from gpkg onto the GeoTIFF and save
        a georeferenced overlay raster (same CRS, same resolution).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tif = f"out_{ts}.tif"
    if isinstance(layers, str):
        layers = [layers]
    random.seed(42)

    # ---- validate layers ----
    available = fiona.listlayers(gpkg)
    missing = [l for l in layers if l not in available]
    if missing:
        raise ValueError(f"Missing layers: {missing}. Available: {available}")
    if "area_boundary" not in available:
        raise ValueError('"area_boundary" layer not found in gpkg')

    # ---- open source raster ----
    with rasterio.open(geotiff) as src:
        meta = src.meta.copy()
        raster = src.read()          # (count, H, W)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    if crs is None:
        raise ValueError("GeoTIFF has no CRS defined; cannot align vector layers.")

    # ---- prepare base RGB from original raster ----
    if raster.shape[0] >= 3:
        base = raster[:3, :, :]
    else:
        base = np.repeat(raster[0:1, :, :], 3, axis=0)

    blended = base.copy()

    # ------------------------------------------------------------------
    # 2) RASTERIZE REQUESTED LAYERS AS COLORED OVERLAYS
    # ------------------------------------------------------------------
    color_mask = np.zeros((3, height, width), dtype=np.uint8)
    layer_colors = {}   # for legend & pins: layer_name -> "#RRGGBB"
    label_records = []  # {"x": float, "y": float, "text": str, "r": int, "g": int, "b": int}

    for layer in layers:
        gdf = gpd.read_file(gpkg, layer=layer)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Layer '{layer}' has no CRS defined.")
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        # -------------------- set layer color (for pins & legend) --------------------
        # even for 'distances' we want a single color for pins & legend
        layer_colors.setdefault(layer, "#" + "".join(random.choices("0123456789ABCDEF", k=6)))
        pin_hex = layer_colors[layer].lstrip("#")
        pin_r = int(pin_hex[0:2], 16)
        pin_g = int(pin_hex[2:4], 16)
        pin_b = int(pin_hex[4:6], 16)

        # ---- SPECIAL CASE: per-feature color for geometry if 'distances' in layer name ----
        if "distances" in layer.lower():
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue

                if geom.geom_type in ("Polygon", "MultiPolygon"):
                    geom = geom.boundary

                hex_color = "".join(random.choices("0123456789ABCDEF", k=6))
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)

                burned = rasterize(
                    shapes=[(geom, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )

                line_width_pixels = 3
                structure = np.ones((line_width_pixels, line_width_pixels), dtype=bool)
                mask = burned == 1
                mask = binary_dilation(mask, structure=structure)

                color_mask[0][mask] = r
                color_mask[1][mask] = g
                color_mask[2][mask] = b

        # ---- DEFAULT CASE: one color for the whole layer geometry ----
        else:
            geom_hex = layer_colors[layer].lstrip("#")
            r = int(geom_hex[0:2], 16)
            g = int(geom_hex[2:4], 16)
            b = int(geom_hex[4:6], 16)
          
            shapes = []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type in ("Polygon", "MultiPolygon"):
                    geom = geom.boundary
                shapes.append((geom, 1))

            burned = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )

            line_width_pixels = 3
            structure = np.ones((line_width_pixels, line_width_pixels), dtype=bool)
            mask = burned == 1
            mask = binary_dilation(mask, structure=structure)

            color_mask[0][mask] = r
            color_mask[1][mask] = g
            color_mask[2][mask] = b

        # ---- collect label locations + their color (from layer_colors) ----
        if show_names:
            gtypes = gdf.geometry.geom_type

            # points → label at point
            points = gdf[gtypes.str.contains("Point", na=False)]
            for _, row in points.iterrows():
                name = get_name_row(row)
                geom = row.geometry
                if not name or geom is None or geom.is_empty:
                    continue
                label_records.append({
                    "x": geom.x,
                    "y": geom.y,
                    "text": name,
                    "r": pin_r,
                    "g": pin_g,
                    "b": pin_b
                })

            # polygons → label at centroid
            polys = gdf[gtypes.str.contains("Polygon", na=False)]
            for _, row in polys.iterrows():
                name = get_name_row(row)
                geom = row.geometry
                if not name or geom is None or geom.is_empty:
                    continue
                c = geom.centroid
                label_records.append({
                    "x": c.x,
                    "y": c.y,
                    "text": name,
                    "r": pin_r,
                    "g": pin_g,
                    "b": pin_b
                })

    # alpha blend overlays
    alpha_overlay = 0.8
    mask_any = (color_mask[0] > 0) | (color_mask[1] > 0) | (color_mask[2] > 0)
    if mask_any.any():
        for i in range(3):
            band_base = blended[i].astype("float32")
            band_ov   = color_mask[i].astype("float32")
            band_base[mask_any] = (1 - alpha_overlay) * band_base[mask_any] + alpha_overlay * band_ov[mask_any]
            blended[i] = band_base.astype(np.uint8)

    from matplotlib import font_manager as fm
    try:
        font_path = fm.findfont("DejaVu Sans", fontext="ttf")
        font_size = int(height * 0.025)
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.info("WARNING: Could not load font:", e)
        font = ImageFont.load_default()

    img = np.moveaxis(blended, 0, -1)  # (H, W, 3)
    pil_img = Image.fromarray(img).convert("RGBA")
    draw = ImageDraw.Draw(pil_img, "RGBA")

    base_pin_icon = None
    try:
        base_pin_icon = Image.open(PIN_PATH).convert("RGBA")
        target_h = max(16, int(height * 0.03))
        w0, h0 = base_pin_icon.size
        scale = target_h / float(h0)
        target_w = int(w0 * scale)
        base_pin_icon = base_pin_icon.resize((target_w, target_h), resample=Image.LANCZOS)
    except FileNotFoundError:
        logger.info("WARNING: pin.png not found; no colored pins will be drawn.")
    tinted_icons = {}
    if show_names and label_records:
        for rec in label_records:
            xw, yw = rec["x"], rec["y"]
            text = str(rec["text"])
            lr, lg, lb = rec["r"], rec["g"], rec["b"]
            row, col = rowcol(transform, xw, yw)

            if not (0 <= row < height and 0 <= col < width):
                continue

            if base_pin_icon is not None:
                color_key = (lr, lg, lb)
                if color_key not in tinted_icons:
                    alpha = base_pin_icon.split()[3]
                    colored = Image.new("RGBA", base_pin_icon.size, (lr, lg, lb, 255))
                    colored.putalpha(alpha)
                    tinted_icons[color_key] = colored
                pin_icon = tinted_icons[color_key]
                icon_w, icon_h = pin_icon.size

                x0 = int(col - icon_w / 2)
                y0 = int(row - icon_h)
                x1 = x0 + icon_w
                y1 = y0 + icon_h

                if not (x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height):
                    crop_x0 = max(0, x0)
                    crop_y0 = max(0, y0)
                    crop_x1 = min(width, x1)
                    crop_y1 = min(height, y1)

                    icon_crop = pin_icon.crop((
                        crop_x0 - x0,
                        crop_y0 - y0,
                        crop_x1 - x0,
                        crop_y1 - y0
                    ))
                    pil_img.paste(icon_crop, (crop_x0, crop_y0), icon_crop)

            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top

            pad = 2
            text_col = col + icon_w // 2 + 2          
            text_row = row - text_h // 2            

            x0 = text_col - pad
            y0 = text_row - pad
            x1 = text_col + text_w + pad
            y1 = text_row + text_h + pad

            draw.rectangle(
                [x0, y0, x1, y1],
                fill=(255, 255, 255, 10),
                outline=None
            )
            draw.text(
                (text_col - left, text_row - top),
                text,
                fill=(0, 0, 0, 255),
                font=font
            )


    if layer_colors:
        margin = 10
        patch_size = 15
        spacing = 5

        entries = list(layer_colors.items()) 
        text_sizes = []
        for layer_name, _ in entries:
            bbox = draw.textbbox((0, 0), layer_name, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_sizes.append((text_w, text_h))

        max_text_w = max((tw for tw, th in text_sizes), default=0)
        max_text_h = max((th for tw, th in text_sizes), default=patch_size)

        line_height = max(patch_size, max_text_h) + spacing
        legend_width = patch_size + spacing + max_text_w + 2 * margin
        legend_height = len(entries) * line_height + 2 * margin

        x0 = margin
        y0 = margin
        x1 = x0 + legend_width
        y1 = y0 + legend_height

        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(255, 255, 255, 160),
            outline=(0, 0, 0, 200)
        )

        y = y0 + margin
        for (layer_name, hex_color), (tw, th) in zip(entries, text_sizes):
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            px0 = x0 + margin
            py0 = y
            px1 = px0 + patch_size
            py1 = py0 + patch_size
            draw.rectangle(
                [px0, py0, px1, py1],
                fill=(r, g, b, 255),
                outline=(0, 0, 0, 255)
            )

            text_x = px1 + spacing
            text_y = py0 + (patch_size - th) // 2
            draw.text((text_x, text_y), layer_name, fill=(0, 0, 0, 255), font=font)

            y += line_height

    pil_img = pil_img.convert("RGB")
    blended = np.moveaxis(np.array(pil_img), -1, 0)

    meta.update({
        "count": 3,
        "dtype": "uint8",
        "height": height,
        "width": width,
        "transform": transform,
        "crs": crs,
        "driver": "GTiff",
    })
    save_path =  os.path.join(OUTPUT_DIR,out_tif)
    with rasterio.open(save_path, "w", **meta) as dst:
        dst.write(blended)
    return out_tif, save_path

def generate(params):
    """
    Offline worker entry point.

    Expected params:
    - "gpkg":   path to GeoPackage on disk (required)
    - "layers": single layer name (str) or list of layer names (required)
    - "geotiff": path to base GeoTIFF on disk (required)
    - "out_tif": optional output path (default 'out.tif')
    - "show_names": optional bool, default True
    """
    required = ["gpkg", "layers", "geotiff"]
    missing = [k for k in required if k not in params]
    if missing:
        txt_e = f"Missing required parameter(s): {missing}"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 2}

    gpkg = params.get("gpkg")
    geotiff = params.get("geotiff")
    layers = params.get("layers")
    show_names = params.get("show_names", True)

    # Normalize layers: allow comma-separated string, single str, or list
    if isinstance(layers, str):
        if "," in layers:
            layers = [l.strip() for l in layers.split(",") if l.strip()]
        else:
            layers = [layers]
    elif isinstance(layers, (tuple, set)):
        layers = list(layers)

    if not layers:
        txt_e = "No layers specified in 'layers' parameter."
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 2}

    if not os.path.isfile(gpkg):
        txt_e = f"GeoPackage not found: {gpkg}"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 3}

    if not os.path.isfile(geotiff):
        txt_e = f"GeoTIFF not found: {geotiff}"
        logger.error(txt_e)
        return {"text": txt_e, "error_code": 3}

    try:
        logger.info(
            f"Starting DisplayOnGeotiff with gpkg={gpkg}, geotiff={geotiff}, "
            f"layers={layers}, show_names={show_names}"
        )
        out_file, out_path = DisplayOnGeotiff(
            gpkg=gpkg,
            layers=layers,
            geotiff=geotiff,
        )
        txt = f"Rendered layers {layers}, output saved to {out_file}"
        logger.info(txt)
        return {"text": txt, "image": out_path, "error_code": 0}

    except Exception as e:
        txt_e = f"Error rendering layers on GeoTIFF: {e}"
        logger.exception(txt_e)
        return {"text": txt_e, "error_code": 1}


if __name__ == "__main__":
    # simple CLI for debugging the worker standalone
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", type=str, required=True, help="Path to GeoPackage.")
    parser.add_argument("--geotiff", type=str, required=True, help="Path to base GeoTIFF.")
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Layer name or comma-separated list of layer names."
    )
    parser.add_argument("--out-tif", type=str, default="out.tif", help="Output GeoTIFF path.")
    parser.add_argument(
        "--show-names",
        action="store_true",
        help="If set, draw feature names & pins."
    )
    args = parser.parse_args()

    params = {
        "gpkg": args.gpkg,
        "geotiff": args.geotiff,
        "layers": args.layers,
        "out_tif": args.out_tif,
        "show_names": args.show_names,
    }
    out = generate(params)
    logger.info(out)

import argparse
import uuid
import os
import geopandas as gpd
from shapely.geometry import LineString, Point
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsDistanceArea,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
)
import osmnx as ox
import networkx as nx
from networkx.exception import NetworkXNoPath

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

CRS = "EPSG:4326"

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ComputeDistance_worker_{worker_id}.log")


def get_utm_epsg_from_gpkg(gpkg):
    """Return UTM EPSG code based on the centroid of the area boundary layer in the GeoPackage."""
    area = gpd.read_file(gpkg, layer="area_boundary").geometry.iloc[0]
    lon, lat = area.centroid.x, area.centroid.y
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def get_name(f, layer):
    """Return best display name for OSM features."""
    fields = layer.fields().names()
    # Priority: name:en > name > any name ""
    for key in ("name:en", "name"):
        if key in fields and f[key]:
            return f[key]
    for key in fields:
        if key.startswith("name") and f[key]:
            return f[key]    
    return ""

def compute_distances(gpkg, src_layer, tar_layer, top = None):
    """
    Compute pairwise distances between features of two layers in a GeoPackage.
    Saves results as new line layer "{src_layer}_to_{tar_layer}_distances" in GeoPackage.
    Optionally, only the k closest target per source can be retained ( top = k). If top = 1 : single closest target
    
    Parameters
    ----------
    gpkg : str
        Path to GeoPackage file.
    src_layer : str
        Source layer name prefix (e.g., "schools").
    tar_layer : str
        Target layer name prefix (e.g., "restaurants").
    top : bool, optional (default=None)

    Returns
    -------
    str
        Lines of "src tar distance=dist"
    """
    add_time = True 
    default_speed_kph = 40
    src = QgsVectorLayer(f"{gpkg}|layername={src_layer}", src_layer, "ogr")
    tar = QgsVectorLayer(f"{gpkg}|layername={tar_layer}", tar_layer, "ogr")
    if not src.isValid() or not tar.isValid():
        raise Exception("Invalid input layers")

    utm_epsg = get_utm_epsg_from_gpkg(gpkg)
    context = QgsProject.instance().transformContext()
    utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")
    d = QgsDistanceArea()
    d.setSourceCrs(utm_crs, context)
    d.setEllipsoid("WGS84")

    
    noPath = False
    try: 
        boundary = gpd.read_file(gpkg, layer="area_boundary").geometry.iloc[0]
        network = ox.graph_from_polygon(boundary, network_type="drive", simplify=True)
        network = ox.project_graph(network)
        if add_time:
            network = ox.add_edge_speeds(network, fallback=40)
            for u, v, k, da in network.edges(keys=True, data=True):
                if "speed_kph" not in da or da["speed_kph"] is None:
                    da["speed_kph"] = default_speed_kph
            network = ox.add_edge_travel_times(network) 

        results = []
        travel_time = float(0)
        for f1 in src.getFeatures():
            name1 = get_name(f1, src)
            if not name1:  
                continue
            geom1 = f1.geometry().centroid()
            row_results = []
            for f2 in tar.getFeatures():
                name2 = get_name(f2, tar)
                if not name2:   
                    continue
                geom2 = f2.geometry().centroid()
        
                graph_crs = network.graph["crs"]
                pt1_q = geom1.asPoint()
                pt2_q = geom2.asPoint()
                pt1 = gpd.GeoSeries([Point(pt1_q.x(), pt1_q.y())], crs=src.crs().authid())
                pt2 = gpd.GeoSeries([Point(pt2_q.x(), pt2_q.y())], crs=tar.crs().authid())
                pt1 = pt1.to_crs(graph_crs).iloc[0]
                pt2 = pt2.to_crs(graph_crs).iloc[0]
                orig_node = ox.distance.nearest_nodes(network, pt1.x, pt1.y)
                dest_node = ox.distance.nearest_nodes(network, pt2.x, pt2.y)
                try:
                    path = nx.shortest_path(network, orig_node, dest_node, weight="length")
                    dist = nx.shortest_path_length(network, orig_node, dest_node, weight="length")
                    if add_time:
                        tt = nx.shortest_path_length(network, orig_node, dest_node, weight="travel_time")
                        travel_time = float(tt) 
                    graph_crs = network.graph["crs"]

                    src_cent_xy  = (pt1.x, pt1.y)
                    dest_cent_xy = (pt2.x, pt2.y)

                    src_node_xy  = (network.nodes[orig_node]["x"], network.nodes[orig_node]["y"])
                    dest_node_xy = (network.nodes[dest_node]["x"], network.nodes[dest_node]["y"])
                    coords = [src_cent_xy, src_node_xy]
                    current_xy = src_node_xy

                    for u, v in zip(path[:-1], path[1:]):
                        edge_dict = network.get_edge_data(u, v)
                        if edge_dict is None:
                            continue

                        data = min(
                            edge_dict.values(),
                            key=lambda d: d.get("length", 0)
                        )

                        if "geometry" in data and data["geometry"] is not None:
                            edge_line = data["geometry"]        
                            ec = list(edge_line.coords)
                        else:
                            x1, y1 = network.nodes[u]["x"], network.nodes[u]["y"]
                            x2, y2 = network.nodes[v]["x"], network.nodes[v]["y"]
                            ec = [(x1, y1), (x2, y2)]

                        if ec[0] == current_xy:
                            coords.extend(ec[1:])
                            current_xy = ec[-1]
                        elif ec[-1] == current_xy:
                            ec = list(reversed(ec))
                            coords.extend(ec[1:])
                            current_xy = ec[-1]
                        else:
                            coords.extend(ec)
                            current_xy = ec[-1]

                    if current_xy != dest_node_xy:
                        coords.append(dest_node_xy)
                        current_xy = dest_node_xy

                    coords.append(dest_cent_xy)
                    geom_proj = gpd.GeoSeries([LineString(coords)], crs=graph_crs)
                    geom = geom_proj.to_crs("EPSG:4326").iloc[0]


                except NetworkXNoPath:
                    noPath = True
                    break
                        
                row_results.append({
                    f"{src_layer}": name1,
                    f"{tar_layer}": name2,
                    "distance_m": dist,
                    "geometry": geom,
                    "travel_time_s": travel_time if add_time else None
                })
            if noPath:
                break
            row_results = [r for r in row_results if r["distance_m"] is not None]
            row_results.sort(key=lambda r: r["distance_m"])
            if top is None:
                results.extend(row_results)   
            else:
                results.extend(row_results[:top]) 
    except:
        noPath = True

    if noPath:
        # print("inside noPath")
        results = []
        for f1 in src.getFeatures():
            name1 = get_name(f1, src)
            if not name1:  
                continue
            geom1 = f1.geometry().centroid()
            row_results = []
            for f2 in tar.getFeatures():
                name2 = get_name(f2, tar)
                if not name2:   
                    continue
                geom2 = f2.geometry().centroid()
                g1 = QgsGeometry(geom1)   
                g2 = QgsGeometry(geom2)   
                g1.transform(QgsCoordinateTransform(src.crs(), utm_crs, context))
                g2.transform(QgsCoordinateTransform(tar.crs(), utm_crs, context))
                dist = d.measureLine(g1.asPoint(), g2.asPoint())
                geom = LineString([f1.geometry().centroid().asPoint(),
                                    f2.geometry().centroid().asPoint()])
                row_results.append({
                    f"{src_layer}": name1,
                    f"{tar_layer}": name2,
                    "distance_m": dist,
                    "geometry": geom,
                })
            row_results = [r for r in row_results if r["distance_m"] is not None]
            row_results.sort(key=lambda r: r["distance_m"])
            if top is None:
                results.extend(row_results)      
            else:
                results.extend(row_results[:top]) 
    gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326") \
        .to_file(gpkg, layer=f"{src_layer}_to_{tar_layer}_distances", driver="GPKG")
    out_lines = []
    distances = []
    travel_times = []
    out_lines.append(f"Distances (in meters) saved to line layer: '{src_layer}_to_{tar_layer}_distances': ")
    for r in results:
        txt = f"{r[src_layer]} , {r[tar_layer]}, distance={r['distance_m']:.2f} m"
        if add_time and not noPath:
            txt += f", travel_time={r['travel_time_s']:.1f} s"
            travel_times.append(round(r["travel_time_s"], 1))

        out_lines.append(txt)
        distances.append(round(r['distance_m'], 2))
    out_lines.append(f"distances = {distances}")
    if add_time and not noPath: 
        out_lines.append(f"travel_times = {travel_times}")

    output_str = "\n".join(out_lines)
    # print(output_str)
    return output_str


class ComputeDistanceWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="ComputeDistance",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=1,
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
        logger.info("ComputeDistanceWorker does not need a model. Ready to run.")

    def generate(self, params):
        required_keys = ("gpkg", "src_layer", "tar_layer")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        src_layer = params.get("src_layer")
        tar_layer = params.get("tar_layer")
        top = params.get("top", None)
        
        if not os.path.exists(gpkg):
                txt_e = f"GeoPackage not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
        
        try:
            output_str = compute_distances(gpkg, src_layer, tar_layer, top)
            return {"text": output_str, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in ComputeDistance: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "ComputeDistance",
                "description": "Computes pairwise distances between features of two layers in a GeoPackage and saves results to a new line layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "src_layer": {"type": "string", "description": "Source layer name."},
                        "tar_layer": {"type": "string", "description": "Target layer name."},
                        "top": {"type": "integer", "description": "Optional: number of closest targets per source to keep."},
                    },
                    "required": ["gpkg", "src_layer", "tar_layer"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20019)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ComputeDistance")
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ComputeDistanceWorker(
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
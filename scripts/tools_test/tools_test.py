import os
import json
import time
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

"""
Test script for all  tool workers.

Runs each tool in a controlled sequence with predefined example inputs.
Automatically prints formatted results for quick verification.
"""


def abs_path(rel_path: str) -> str:
    """Convert relative test file paths to absolute paths."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))


def main():
    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)

    # ----------------------------------------------------------------------
    # Example test inputs — executed in this exact sequence
    # ----------------------------------------------------------------------
    example_calls = [
        ("Calculator", {"expression": "round(302 / (pi), 4)"}),

        ("TextToBbox", {
            "image": abs_path("tools_test_images/1.jpg"),
            "text": "red car",
            "top1" : True,
        }),

        ("DrawBox", {
            "image": abs_path("tools_test_images/1.jpg"),
            "bbox": "(289,645,330,724)",
            "annotation": "Red Car"
        }),

        ("AddText", {
            "image": abs_path("tools_test_images/1.jpg"),
            "text": "Red Car",
            "position": "(289,640)"
        }),

        ("OCR", {"image": abs_path("tools_test_images/1_withText.jpg")}),

        ("GoogleSearch", {"query": "San Jose State, Spartan stadium location"}),

        ("Plot",{"command":"```python\nimport matplotlib.pyplot as plt\ndef solution():\n fig=plt.figure(); plt.plot([0,1],[2,3]); return fig\n```"}),

        ("Solver",{"command":"```python\nfrom sympy import symbols, Eq, solve\ndef solution():\n x= symbols('x'); return str(solve(Eq(2*x+3,11),x))\n```"}),

        ("ImageDescription", {
            "image": abs_path("tools_test_images/1.jpg")
        }),

        ("RegionAttributeDescription", {
            "image": abs_path("tools_test_images/1.jpg"),
            "attribute": "fence around tennis courts"
        }),

        ("CountGivenObject", {
            "image": abs_path("tools_test_images/1.jpg"),
            "text": "cars",
        }),

        ("ChangeDetection", {
            "text": "Classify the level of damage experienced by the building at location [0, 8, 49, 53].",
            "pre_image": abs_path("tools_test_images/xBD_cls_1.png"),
            "post_image": abs_path("tools_test_images/xBD_cls_2.png")
        }),

        ("ObjectDetection", {
            "image": abs_path("tools_test_images/1.jpg")
        }),

        ("SegmentObjectPixels", {
            "image": abs_path("tools_test_images/1.jpg"),
            "text": "red car"
        }),

        ("GetAreaBoundary", {"area": "Edinburgh Castle, Edinburgh, Scotland, UK","buffer_m": 1000}),

        ("AddPoisLayer", {
            "gpkg": abs_path("tools_test_images/aoi.gpkg"),
            "query": {"amenity": "pharmacy"},
            "layer_name": "pharmacies"
        }),

        ("AddPoisLayer", {
            "gpkg": abs_path("tools_test_images/aoi.gpkg"),
            "query": {"amenity": "school"},
            "layer_name": "schools"
        }),

        ("ComputeDistance", {
            "gpkg": abs_path("tools_test_images/aoi.gpkg"),
            "src_layer": "schools",
            "tar_layer": "pharmacies",
            "top":1
        }),
         ("DisplayOnMap", {
            "gpkg": abs_path("tools_test_images/aoi.gpkg"),
            "layers": ["area_boundary", "pharmacies", "schools", "schools_to_pharmacies_distances"]
        }),

        ("GetAreaBoundary", {"area": "Manchester State Forest, South Carolina, United States"}),

        ("AddIndexLayer", {
            "gpkg": abs_path("tools_test_images/aoi_1.gpkg"),
            "index_type": "NDVI",
            "layer_name": "ndvi_2022",
            "year": 2022
        }),

        ("AddIndexLayer", {
            "gpkg": abs_path("tools_test_images/aoi_1.gpkg"),
            "index_type": "NDVI",
            "layer_name": "ndvi_2025",
            "year": 2025
        }),

        ("ComputeIndexChange", {
            "gpkg": abs_path("tools_test_images/aoi_1.gpkg"),
            "index_type": "NDVI",
            "layer1_name": "ndvi_2022",
            "layer2_name": "ndvi_2025",
            "diff_layer_name": "delta_ndvi"
        }),

        ("ShowIndexLayer", {
            "gpkg": abs_path("tools_test_images/aoi_1.gpkg"),
            "index_type": "NDVI",
            "layer_name": "delta_ndvi"
        }),
        ("GetBboxFromGeotiff", {
            "geotiff": abs_path("tools_test_images/S_07.tif")
        }),
        ("GetAreaBoundary", {"area": (-95.9121, 41.2912, -95.8925, 41.3066)}),
        ("AddPoisLayer", {
            "gpkg": abs_path("tools_test_images/aoi_tif.gpkg"),
            "query": {"aeroway": "terminal"},
            "layer_name": "terminals"
        }),

        ("DisplayOnGeotiff", {
            "gpkg": abs_path("tools_test_images/aoi_tif.gpkg"),
            "geotiff": abs_path("tools_test_images/S_07.tif"),
            "layers": ['terminals'],
        }),
    ]

    # ----------------------------------------------------------------------
    # test all tools
    # ----------------------------------------------------------------------
    print("\n=== Running tests for all workers ===\n")

    tools_with_errors = []
    tools_successful = []
    skipped_tools = []

    skipped = 0
    success = 0
    error = 0

    for tool_name, params in example_calls:

        if tool_name not in tm.available_tools:
            print(f"⚠️  {tool_name} is not registered or unavailable — skipping.\n")
            skipped += 1
            skipped_tools.append(tool_name)
            continue

        print(f"🔹 Testing {tool_name} with parameters:")
        print(json.dumps(params, indent=2))

        try:
            result = tm.call_tool(tool_name, params)

            if result.get("error_code") == 0:
                print(f"✅ {tool_name} executed successfully.")
                print(f"Result:\n{json.dumps(result, indent=2)}\n")

                success += 1
                tools_successful.append(tool_name)
            else:
                print(f"❌ {tool_name} returned an error.")
                print(f"Result:\n{json.dumps(result, indent=2)}\n")

                error += 1
                tools_with_errors.append(tool_name)

        except Exception as e:
            print(f"❌ Exception while testing {tool_name}: {str(e)}\n")
            error += 1
            tools_with_errors.append(tool_name)

        print("-" * 100 + "\n")
        time.sleep(0.5)  # small delay between calls

    # ----------------------------------------------------------------------
    # Summary Section
    # ----------------------------------------------------------------------
    print("\n🎯 All example tools executed in sequence.\n")

    print("📊 Summary:")
    print(f"   ✅ Successful : {success}")
    print(f"   ❌ Failed     : {error}")
    print(f"   ⚠️  Skipped    : {skipped}\n")

    if tools_successful:
        print(f"✅ Successful tools:\n   {sorted(set(tools_successful))}\n")

    if tools_with_errors:
        print(f"❌ Failed tools:\n   {sorted(set(tools_with_errors))}\n")

    if skipped_tools:
        print(f"⚠️  Skipped tools:\n   {sorted(set(skipped_tools))}\n")

    if error == 0 and skipped == 0:
        print("🎉 All tools ran successfully without errors.")


if __name__ == "__main__":
    main()

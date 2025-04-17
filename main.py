import os
import json
import subprocess
import tempfile
from osgeo import gdal

config = {}
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        user_config = json.load(f)
        config.update(user_config)

os.makedirs(os.path.dirname(config["output_las"]), exist_ok=True)
os.makedirs(os.path.dirname(config["output_tif"]), exist_ok=True)

pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": config["input_file"]
        },
        {
            "type": "filters.reprojection",
            "out_srs": f"EPSG:{config['epsg']}"
        },
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": config["outlier_mean_k"],
            "multiplier": config["outlier_multiplier"]
        },
        {
            "type": "filters.smrf",
            "scalar": config["smrf_scalar"],
            "slope": config["smrf_slope"],
            "threshold": config["smrf_threshold"],
            "window": config["smrf_window"],
            "cell": 1.0
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "writers.las",
            "filename": config["output_las"]
        },
        {
            "type": "writers.gdal",
            "filename": config["output_tif"],
            "gdaldriver": "GTiff",
            "output_type": config["raster_output_type"],
            "resolution": config["raster_resolution"]
        }
    ]
}

pipeline_file = "temp_pipeline.json"
with open(pipeline_file, "w") as f:
    json.dump(pipeline, f, indent=4)

subprocess.run(["pdal", "pipeline", pipeline_file], check=True)

os.remove(pipeline_file)

print("ℹ️ Running interpolation on raster...")
dem = gdal.Open(config["output_tif"], gdal.GA_Update)
dem_copy = gdal.GetDriverByName("GTiff").CreateCopy(config["interpolated_tif"], dem)

gdal.FillNodata(
    targetBand=dem_copy.GetRasterBand(1),
    maskBand=None,
    maxSearchDist=config["interpolation_maxdist"],
    smoothingIterations=config["interpolation_smoothing"]
)

dem = None
print("✅ Processing complete. Output files generated.")

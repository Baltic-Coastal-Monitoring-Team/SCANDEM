import streamlit as st
import os
import json
import subprocess
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from osgeo import gdal

st.set_page_config(page_title="SCANDEM - LAS Classification", layout="wide")
st.title("SCANDEM – LAS File Classifier")
st.markdown("This app classifies LAS point cloud files and generates digital elevation models (DEM).")

st.markdown("""
<style>
div.stFormSubmitButton > button {
    background-color: #d9534f;
    color: white;
    font-weight: bold;
    border: none;
    padding: 0.5em 1.5em;
    border-radius: 4px;
    transition: 0.3s;
    margin:0 auto;
}
div.stFormSubmitButton > button:hover {
    background-color: #fff;
    color: #d9534f;
    border:2px solid #d9534f;       
}
</style>
""", unsafe_allow_html=True)

# List available LAS files
input_files = [f for f in os.listdir("input") if f.lower().endswith(".las")]

demo_file = "demo.las"
demo_mode = False
if demo_file not in input_files:
    demo_mode = False
    st.warning("No .las files found in the 'input' folder.")
    st.stop()
else:
    demo_mode = st.checkbox("Use demo LAS file", value=True)

if demo_mode:
    selected_file = demo_file
else:
    selected_file = st.selectbox("Select a .las file to process:", input_files)

with st.form("parameters_form"):
    st.subheader("Classification and Interpolation Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        epsg = st.text_input("EPSG Code", value="2180",
                             help="Projection system used for reprojection")
        smrf_scalar = st.number_input("SMRF: scalar", value=1.2,
                             help="Scales window size; higher = more tolerant to vegetation")
        smrf_threshold = st.number_input("SMRF: threshold", value=0.45,
                             help="Elevation difference threshold to classify ground")
        outlier_mean_k = st.number_input("Outlier: mean_k", value=8,
                             help="Number of neighbors in statistical filter")

    with col2:
        smrf_slope = st.number_input("SMRF: slope", value=0.4,
                             help="Slope tolerance for surface model")
        smrf_window = st.number_input("SMRF: window", value=16,
                             help="Initial window size (in pixels)")
        outlier_multiplier = st.number_input("Outlier: multiplier", value=2.5,
                             help="Threshold multiplier for outlier detection")
        raster_output_type = st.selectbox("Raster output type", ["min", "max", "mean"],
                             help="Raster cell aggregation method (e.g., min elevation in cell)")

    with col3:
        interpolation_maxdist = st.number_input("Interpolation: max search distance", value=25,
                             help="Max distance to search for fill values in interpolation")
        interpolation_smoothing = st.number_input("Interpolation: smoothing iterations", value=5,
                             help="Number of smoothing passes applied during nodata filling")
        raster_resolution = st.number_input("Raster resolution", value=0.1,
                             help="Output raster pixel size in meters")

    submitted = st.form_submit_button("Run Processing")

if submitted:
    input_path = os.path.join("input", selected_file)
    base_name = os.path.splitext(selected_file)[0]

    config = {
        "input_file": input_path,
        "output_las": f"output/{base_name}_classified.las",
        "output_tif": f"output/{base_name}_dem.tif",
        "interpolated_tif": f"output/{base_name}_interpolated.tif",
        "epsg": epsg,
        "smrf_scalar": smrf_scalar,
        "smrf_slope": smrf_slope,
        "smrf_threshold": smrf_threshold,
        "smrf_window": smrf_window,
        "outlier_mean_k": outlier_mean_k,
        "outlier_multiplier": outlier_multiplier,
        "raster_resolution": raster_resolution,
        "raster_output_type": raster_output_type,
        "interpolation_maxdist": interpolation_maxdist,
        "interpolation_smoothing": interpolation_smoothing
    }

    os.makedirs("output", exist_ok=True)

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    st.info("Running PDAL classification and DEM generation...")

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": config["input_file"]},
            {"type": "filters.reprojection", "out_srs": f"EPSG:{config['epsg']}"},
            {"type": "filters.outlier", "method": "statistical",
             "mean_k": config["outlier_mean_k"], "multiplier": config["outlier_multiplier"]},
            {"type": "filters.smrf", "scalar": config["smrf_scalar"], "slope": config["smrf_slope"],
             "threshold": config["smrf_threshold"], "window": config["smrf_window"], "cell": 1.0},
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {"type": "writers.las", "filename": config["output_las"]},
            {"type": "writers.gdal", "filename": config["output_tif"], "gdaldriver": "GTiff",
             "output_type": config["raster_output_type"], "resolution": config["raster_resolution"]}
        ]
    }

    pipeline_file = "temp_pipeline.json"
    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f, indent=4)

    try:
        subprocess.run(["pdal", "pipeline", pipeline_file], check=True)
        st.success("PDAL processing completed!")
    except subprocess.CalledProcessError as e:
        st.error("PDAL processing failed.")
        st.code(str(e))
    finally:
        if os.path.exists(pipeline_file):
            os.remove(pipeline_file)

    st.info("Interpolating no-data values in DEM raster...")

    dem = gdal.Open(config["output_tif"], gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName("GTiff")
    dem_copy = driver.CreateCopy(config["interpolated_tif"], dem, strict=0)
    band = dem_copy.GetRasterBand(1)

    gdal.FillNodata(
        targetBand=band,
        maskBand=None,
        maxSearchDist=config["interpolation_maxdist"],
        smoothingIterations=config["interpolation_smoothing"]
    )

    band.FlushCache()
    dem_copy.FlushCache()
    band = None
    dem_copy = None
    dem = None

    st.success("Interpolation complete. Both raster files saved to 'output/'.")

    st.session_state["dem_ready"] = True
    st.session_state["raw_dem_path"] = config["output_tif"]
    st.session_state["interpolated_dem_path"] = config["interpolated_tif"]

# Raster preview section
def show_raster(path, title):
    st.subheader(title)
    with rasterio.open(path) as src:
        band = src.read(1)
        nodata = src.nodata

        hide_nodata = st.checkbox(f"Hide NoData in '{title}' preview", value=True, key=f"nodata_{title}")

        if hide_nodata and nodata is not None:
            mask = band != nodata
        else:
            mask = np.ones_like(band, dtype=bool)

        masked = np.ma.masked_where(~mask, band)
        data_values = masked.compressed()

        st.write(f"File: {path}")
        st.write(f"Min (masked): {masked.min():.2f}, Max: {masked.max():.2f}, NoData: {nodata}")
        st.write(f"Masked pixels (nodata): {(~mask).sum()}")

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(masked, cmap="terrain")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Elevation [m]")
        st.pyplot(fig)

        st.markdown(f"**Elevation Histogram – {title}**")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
        ax_hist.hist(data_values, bins=50, color="gray", edgecolor="black")
        ax_hist.set_xlabel("Elevation [m]")
        ax_hist.set_ylabel("Frequency")
        st.pyplot(fig_hist)

        return {
            "title": title,
            "min": float(masked.min()),
            "max": float(masked.max()),
            "mean": float(np.mean(data_values)),
            "std": float(np.std(data_values)),
            "count": int(data_values.size),
            "nodata_pixels": int((~mask).sum())
        }

if st.session_state.get("dem_ready"):
    st.markdown("### Raster Preview")
    col1, col2 = st.columns(2)
    with col1:
        stats_raw = show_raster(st.session_state["raw_dem_path"], "Raw DEM Raster")
    with col2:
        stats_interp = show_raster(st.session_state["interpolated_dem_path"], "Interpolated DEM Raster")

    st.markdown("### Raster Statistics Comparison")

    def format_stats_row(label, raw_val, interp_val, unit=""):
        return f"| **{label}** | {raw_val:.2f} {unit} | {interp_val:.2f} {unit} |"

    table_md = "\n".join([
        "| Metric | Raw DEM | Interpolated DEM |",
        "|--------|---------|------------------|",
        format_stats_row("Min", stats_raw["min"], stats_interp["min"]),
        format_stats_row("Max", stats_raw["max"], stats_interp["max"]),
        format_stats_row("Mean", stats_raw["mean"], stats_interp["mean"]),
        format_stats_row("Std Dev", stats_raw["std"], stats_interp["std"]),
        format_stats_row("Pixel Count", stats_raw["count"], stats_interp["count"], unit="px"),
        format_stats_row("NoData Pixels", stats_raw["nodata_pixels"], stats_interp["nodata_pixels"], unit="px")
    ])

    st.markdown(table_md)

    st.markdown("### Interpolated Pixel Locations")
    with rasterio.open(st.session_state["raw_dem_path"]) as raw_src, \
         rasterio.open(st.session_state["interpolated_dem_path"]) as interp_src:

        raw = raw_src.read(1)
        interp = interp_src.read(1)

        raw_nodata = raw_src.nodata
        interp_nodata = interp_src.nodata

        interpolated_mask = (raw == raw_nodata) & (interp != interp_nodata)
        filled_count = np.count_nonzero(interpolated_mask)

        st.write(f"Interpolated pixels: {filled_count}")
        fig_diff, ax_diff = plt.subplots(figsize=(8, 6))
        ax_diff.imshow(interpolated_mask, cmap="coolwarm")
        ax_diff.set_title("Pixels filled during interpolation")
        st.pyplot(fig_diff)

if submitted:
    st.success("Processing complete.")
    st.session_state["plots_visible"] = True

if "plots_visible" in st.session_state and st.session_state["plots_visible"]:
    if st.button("Reset visualization"):
        st.session_state.clear()
        st.experimental_rerun()
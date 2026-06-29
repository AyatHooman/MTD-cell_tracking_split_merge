"""
MTD-SplitMerge Processing Script
================================
Runs the full MTD storm-cell tracking pipeline (Steps 1-5) as a single script,
suitable for a batch (PBS) job.

The script lives in <repo>/run/, so it first switches into the repository root,
where the input/output data folders live and "import MTD..." resolves. This makes
it runnable from anywhere (e.g. a PBS compute node).

Author: Hooman Ayat
"""

import os
import sys

# --- Run from the repository root, no matter where the job is launched ------------
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
current_dir = repo_dir
print(f"Repository / working directory: {current_dir}")


# ============================ Step 1  ConvolutionProcessor ============================
from MTD.obj_convolution import ConvolutionProcessor

input_folder = os.path.join(current_dir, 'MRMS-Sample_data', 'inputs')
output_convolved_folder = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'convolved_maps')
input_raster_main_field = 'PrecipRate_0mabovemeansealevel'
input_raster_time_field = 'time'
R = 3
Th = 0.1

processor = ConvolutionProcessor(
    input_folder=input_folder,
    output_folder=output_convolved_folder,
    input_raster_main_field=input_raster_main_field,
    input_raster_time_field=input_raster_time_field,
    R=3,
    Th=0.1,
)
processor.run()


# ============================ Step 2  MergerProcessor ============================
from MTD.obj_merging import MergerProcessor

output_merged_maps_folder_address = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'merged_convolved_maps')
input_covolved_maps_folder_address = output_convolved_folder
processor = MergerProcessor(input_covolved_maps_folder_address, output_merged_maps_folder_address, 2)
processor.merge_files_by_season_and_year()


# ============================ Step 3  ObjectTrackerProcessor ============================
from MTD.obj_saving_trakced import ObjectTrackerProcessor

output_merged_maps_tracked_folder_address = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'merged_convolved_maps_tracked')
connections_folder = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'connections_folder')
processor = ObjectTrackerProcessor(output_merged_maps_folder_address, output_merged_maps_tracked_folder_address, connections_folder)
processor.process_files()


# ============================ Step 4  ObjectPropertiesProcessor ============================
from pathlib import Path
from MTD.obj_object_analysis import ObjectPropertiesProcessor

output_folder_snapshots = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'objects', 'snapshot_properties')
output_folder_averages = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'objects', 'averaged_properties')
smaple_raw_data_address = Path(os.path.join(current_dir, 'MRMS-Sample_data', 'inputs', 'PrecipRate_00.00_20141101-153000_double_double.nc'))
input_raster_main_field = 'PrecipRate_0mabovemeansealevel'
processor = ObjectPropertiesProcessor(
    merged_folder=output_merged_maps_tracked_folder_address,
    output_folder_snapshots=output_folder_snapshots,
    output_folder_averages=output_folder_averages,
    smaple_raw_data_address=smaple_raw_data_address,
    input_raster_main_field=input_raster_main_field,
    area_threshold=1,
    pixel_resolution=10,
    time_resolution=30,
)
Radar_data_obj_list, Radar_data_ave_list = processor.process_files()


# ============================ Step 5  SystemPropertiesProcessor ============================
from MTD.obj_system_analysis import SystemPropertiesProcessor

output_folder_snapshots = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'objects', 'snapshot_properties')
output_folder_averages = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'objects', 'averaged_properties')
output_path = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'systems')
connection_path = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'connections_folder')
processor = SystemPropertiesProcessor(output_folder_snapshots, output_folder_averages, output_path, connection_path)
processor.run_analysis()

print("\nAll pipeline steps completed.")


# ====================== Quick check (optional inspection) ======================
# Print the largest (object_id * raw) value in one tracked file, as a simple sanity check.
import glob
import numpy as np
import xarray as xr

tracked_files = sorted(glob.glob(os.path.join(output_merged_maps_tracked_folder_address, '*.nc')))
if tracked_files:
    image = xr.open_dataset(tracked_files[-1])
    print("Max (object_id * raw) in", os.path.basename(tracked_files[-1]), "=",
          float(np.max(image.fcst_object_id * image.fcst_raw)))

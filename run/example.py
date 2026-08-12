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


# ============================ Step 6  Storm-track plot ============================
# One figure with two maps. The left map shows one trajectory per storm system: its
# position at each time step is the area-weighted centre of all its cells, and the
# open circle marks where the system started. The right map shows the tracks of the
# individual objects (cells), drawn in the colour of the storm system they belong to.
# A system is a group of cells joined by splits and merges, exactly as in Step 5.
# Coastlines and country/state borders are drawn behind the tracks, and the maps are
# focused on the area where objects were detected, so the figure works for any region.
import glob
import numpy as np
import pandas as pd
import networkx as nx
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

n_systems_to_plot = 30  # per season, longest total track first

plot_file = os.path.join(current_dir, 'MRMS-Sample_data', 'outputs', 'storm_tracks.png')

# Pixel row/column to latitude/longitude, taken from any tracked map
tracked_files = sorted(glob.glob(os.path.join(output_merged_maps_tracked_folder_address, '*.nc')))
grid = xr.open_dataset(tracked_files[0])
lat_axis = grid['latitude'].values
lon_axis = grid['longitude'].values

connections = np.load(os.path.join(connection_path, 'connections.npy'), allow_pickle=True)

fig, (ax_storms, ax_objects) = plt.subplots(
    1, 2, figsize=(20, 9), subplot_kw={'projection': ccrs.PlateCarree()})
color_cycle = plt.cm.tab20(np.linspace(0, 1, 20))
n_drawn = 0
lon_seen = []
lat_seen = []
for s in range(4):
    obj = pd.read_feather(os.path.join(output_folder_snapshots, f'Radar_data_obj_{s}.ftr'))
    ave = pd.read_feather(os.path.join(output_folder_averages, f'Radar_data_ave_{s}.ftr'))
    if len(ave) == 0:
        continue

    # Same grouping as Step 5: labels joined by split/merge connections form one system
    labels_in_season = set(int(lb) for lb in np.unique(obj.label))
    G = nx.Graph()
    G.add_nodes_from(labels_in_season)
    for a, b in connections:
        if int(a) in labels_in_season and int(b) in labels_in_season:
            G.add_edge(int(a), int(b))

    # Total distance travelled by each system, to pick the longest ones
    distance_per_label = ave.groupby('label')['d'].sum()
    systems = []
    for members in nx.connected_components(G):
        total = float(distance_per_label.reindex(list(members)).fillna(0).sum())
        systems.append((total, members))
    systems.sort(key=lambda item: item[0], reverse=True)

    n_drawn_season = 0
    for total, members in systems:
        if n_drawn_season == n_systems_to_plot:
            break
        cells = obj[obj.label.isin(list(members))]

        # One position per time step: the area-weighted centre of the system's cells
        track_x = []
        track_y = []
        for when, snapshot in cells.groupby('datetime'):
            weight = snapshot['area'].values
            track_x.append(np.sum(snapshot['Centroid_X'].values * weight) / np.sum(weight))
            track_y.append(np.sum(snapshot['Centroid_Y'].values * weight) / np.sum(weight))
        if len(track_x) < 2:
            continue

        color = color_cycle[n_drawn % len(color_cycle)]

        # Left map: the storm-system track
        lats = lat_axis[np.round(track_y).astype(int)]
        lons = lon_axis[np.round(track_x).astype(int)]
        lon_seen += [lons.min(), lons.max()]
        lat_seen += [lats.min(), lats.max()]
        ax_storms.plot(lons, lats, '-o', color=color, linewidth=1.5, markersize=3,
                       transform=ccrs.PlateCarree())
        ax_storms.plot(lons[0], lats[0], 'o', color=color, markersize=8,
                       markerfacecolor='none', transform=ccrs.PlateCarree())

        # Right map: the tracks of the system's objects, in the same colour
        for cell_label, cell in cells.groupby('label'):
            if len(cell) < 2:
                continue
            cell = cell.sort_values('datetime')
            cell_lats = lat_axis[cell['Centroid_Y'].round().astype(int)]
            cell_lons = lon_axis[cell['Centroid_X'].round().astype(int)]
            lon_seen += [cell_lons.min(), cell_lons.max()]
            lat_seen += [cell_lats.min(), cell_lats.max()]
            ax_objects.plot(cell_lons, cell_lats, '-', color=color, linewidth=0.9,
                            alpha=0.7, transform=ccrs.PlateCarree())
            ax_objects.plot(cell_lons[0], cell_lats[0], 'o', color=color, markersize=2,
                            alpha=0.7, transform=ccrs.PlateCarree())

        n_drawn += 1
        n_drawn_season += 1

# Map background: land, coastlines and borders, focused on the detected objects
if lon_seen:
    margin = 2.0
    extent = [min(lon_seen) - margin, max(lon_seen) + margin,
              min(lat_seen) - margin, max(lat_seen) + margin]
    for ax, title in [(ax_storms, f'{n_systems_to_plot} longest storm-system tracks per season (open circle = start)'),
                      (ax_objects, 'Object tracks, coloured by their storm system')]:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='0.96', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, edgecolor='0.4', zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6, edgecolor='0.4', zorder=0)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='0.75', zorder=0)
        gridliner = ax.gridlines(draw_labels=True, linewidth=0.2, color='0.85')
        gridliner.top_labels = False
        gridliner.right_labels = False
        ax.set_title(title)

fig.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Storm-track plot saved to: {plot_file}")

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

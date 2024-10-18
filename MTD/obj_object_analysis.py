import numpy as np
from netCDF4 import Dataset
import networkx as nx
import math
import xarray as xr
from pathlib import Path
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import regionprops
import glob
import os
import datetime
from datetime import timedelta
import pandas as pd

class ObjectPropertiesProcessor:
    """
    This class calculates object properties in two ways:
    1) Object properties for each time-step, such as area, shape, volume, max intensity, and average intensity.
    2) Object properties between two time-steps, such as velocity and direction.
       It also calculates the average area and other properties from the previous step for two consecutive time-steps.
    """

    def __init__(
        self,
        merged_folder: str,
        output_folder_snapshots: str,
        output_folder_averages: str,
        smaple_raw_data_address: Path,
        input_raster_main_field: str,
        area_threshold: int,
        pixel_resolution: int,
        time_resolution: float
    ):
        self.merged_folder = merged_folder
        self.output_folder_snapshots = output_folder_snapshots
        self.output_folder_averages = output_folder_averages
        self.area_threshold = area_threshold  # Threshold for area in pixels
        self.pixel_resolution = pixel_resolution  # Pixel resolution in km
        self.time_resolution = time_resolution  # Time resolution in minutes
        self.input_raster_main_field = input_raster_main_field

        # Prepare the border array using a sample file
        sample_file = smaple_raw_data_address
        if not sample_file.exists():
            # Use the first file in merged_folder if sample_file doesn't exist
            files = glob.glob(os.path.join(self.merged_folder, "*.nc"))
            if len(files) == 0:
                raise FileNotFoundError(f"Sample file: {smaple_raw_data_address} to detect the borders does not exist")
            sample_file = files[0]

        self.border_array, self.processed_image = self.detect_edge_and_frame_pixels(sample_file)

    def Cal_Distance(self, point1, point2):
        """
        Calculate the distance between two points in pixels and convert to km.
        """
        dy = point2[0] - point1[0]
        dx = point2[1] - point1[1]
        dist = np.hypot(dx, dy)
        return dist * self.pixel_resolution

    def date_average(self, date0, date1):
        """
        Calculate the average of two datetime objects.
        """
        return date0 + (date1 - date0) / 2

    def Cal_Velocity(self, point1, point2, dt):
        """
        Calculate the velocity between two points over time dt.
        """
        dy = point2[0] - point1[0]
        dx = point2[1] - point1[1]
        speed = np.hypot(dx, dy) / (dt / self.time_resolution) * self.pixel_resolution
        return speed

    def calc_speed(self, x0, x1, y0, y1, dt):
        """
        Alternative method to calculate speed given coordinates and time difference.
        """
        dx = x1 - x0
        dy = y1 - y0
        return np.hypot(dx, dy) / (dt / self.time_resolution) * self.pixel_resolution

    def create_one_d_mask(self, ref_arr, Id):
        """
        Create a binary mask for a specific object ID.
        """
        arr = np.copy(ref_arr)
        arr[arr != Id] = 0
        arr[arr == Id] = 1
        return arr

    def calc_angle(self, x1, y1, x2, y2):
        """
        Calculate the angle in degrees between two points.
        """
        myradians = math.atan2(y2 - y1, x2 - x1)
        mydegrees = math.degrees(myradians)
        return mydegrees

    def detect_edge_and_frame_pixels(self, file_path, size_threshold=1000, distance_threshold=2):
        """
        Detect edge and frame pixels to identify objects touching the borders.
        """
        # Load the dataset
        radar_data = xr.open_dataset(file_path)

        # Replace -3 with 0, and other values with 1
        radar_data_precip = radar_data[self.input_raster_main_field]
        modified_data = xr.where(radar_data_precip == -3, 1, 0)

        # Copy the first time slice to a NumPy array for processing
        image_array = np.copy(modified_data[0].values)

        # Label the connected components in the binary array
        labeled_array, num_features = label(image_array)

        # Get region properties for the labeled connected components
        regions = regionprops(labeled_array)

        # Remove regions smaller than the size threshold
        for region in regions:
            if region.area < size_threshold:
                image_array[labeled_array == region.label] = 0

        # Compute the distance transform on the binary array
        distance = distance_transform_edt(image_array == 0)

        # Create a mask of pixels within the threshold distance to edge pixels
        edge_pixels_mask = (distance > 0) & (distance <= distance_threshold)

        # Create a mask for the frame of the image (boundary)
        frame_pixels_mask = np.zeros_like(image_array, dtype=bool)
        frame_pixels_mask[0:distance_threshold+1, :] = 1  # Top rows
        frame_pixels_mask[-distance_threshold:, :] = 1    # Bottom rows
        frame_pixels_mask[:, 0:distance_threshold+1] = 1  # Left columns
        frame_pixels_mask[:, -distance_threshold:] = 1    # Right columns

        # Combine the edge pixels mask and frame pixels mask
        final_mask = edge_pixels_mask | frame_pixels_mask

        return final_mask, image_array  # Return the final mask and processed image

    def find_touched(self, twod_array):
        """
        Find labels of objects that touch the borders.
        """
        twod_array = np.asarray(twod_array)
        labels = np.unique(self.border_array * twod_array)
        labels = labels[labels != 0]
        return labels

    def Storm_Info(self, MTD_Cube, Raw_Cube, dt, dates_array):
        """
        Extract storm information from the data cubes.
        """
        Mod_Objects_prop_list = []
        Mod_Objects = np.copy(MTD_Cube)
        touched_labels = []

        # Loop through each time step
        for t in range(len(Mod_Objects)):
            touched_labels_temp = []
            props = regionprops(np.asarray(Mod_Objects[t]).astype(int))
            touched_labelz = self.find_touched(Mod_Objects[t])
            for ppp in range(len(props)):
                if props[ppp].label in touched_labelz:
                    touched_labels_temp.append(1)
                else:
                    touched_labels_temp.append(0)
            touched_labels.append(touched_labels_temp)
            Mod_Objects_prop_list.append(props)

        Datalist = np.copy(Raw_Cube)

        NoH_obj_area_list = []
        NoH_obj_centroid_list = []
        NoH_obj_label_list = []
        NoH_obj_Iv_list = []
        NoH_obj_Ismax_list = []
        NoH_obj_orientation_list = []
        NoH_obj_aspectratio_list = []
        NoH_obj_dates_list = []
        NoH_obj_touched_list = []

        # Extract properties for each object at each time step
        for i in range(len(Mod_Objects_prop_list)):
            for j in range(len(Mod_Objects_prop_list[i])):
                prop = Mod_Objects_prop_list[i][j]
                if (prop.minor_axis_length != 0 and
                    prop.major_axis_length != 0 and
                    prop.area > self.area_threshold and
                    np.sum(prop._label_image * Datalist[i]) > 0):
                    NoH_obj_area_list.append(prop.area)
                    NoH_obj_centroid_list.append(prop.centroid)
                    NoH_obj_label_list.append(prop.label)
                    lb = prop.label
                    image = np.copy(prop._label_image)
                    image[image != lb] = 0
                    image[image == lb] = 1
                    NoH_obj_Ismax_list.append(np.max(image * Datalist[i]))
                    NoH_obj_Iv_list.append(np.sum(image * Datalist[i]) * 1000)
                    aspect_ratio = prop.minor_axis_length / prop.major_axis_length
                    NoH_obj_aspectratio_list.append(aspect_ratio)
                    NoH_obj_orientation_list.append(math.degrees(prop.orientation))
                    NoH_obj_dates_list.append(dates_array[i])
                    NoH_obj_touched_list.append(touched_labels[i][j])

        NoH_v_list = []
        NoH_d_list = []
        NoH_a_list = []
        NoH_Ismax_list = []
        NoH_Iv_list = []
        NoH_orientation_list = []
        NoH_dir_list = []
        NoH_aspectratio_list = []
        NoH_label_list = []
        NoH_dates_list = []
        NoH_centroid_list = []
        NoH_touched_list = []

        obj_no = np.unique(NoH_obj_label_list)
        # Calculate properties between consecutive time steps
        for i in range(len(obj_no)):
            mask = np.copy(NoH_obj_label_list)
            mask[mask != obj_no[i]] = 0
            mask[mask == obj_no[i]] = 1
            locs = np.where(mask == 1)
            if len(locs[0]) > 1:
                for j in range(len(locs[0]) - 1):
                    p0 = NoH_obj_centroid_list[locs[0][j]]
                    p1 = NoH_obj_centroid_list[locs[0][j + 1]]
                    x1, y1 = p0[1], p0[0]
                    x2, y2 = p1[1], p1[0]
                    NoH_centroid_list.append((np.asarray(p1) + np.asarray(p0)) / 2)
                    NoH_dir_list.append(self.calc_angle(x1, y1, x2, y2))
                    t0 = NoH_obj_dates_list[locs[0][j]]
                    t1 = NoH_obj_dates_list[locs[0][j + 1]]
                    dt_new = (t1 - t0).total_seconds() / 60
                    NoH_v_list.append(self.Cal_Velocity(p0, p1, dt_new))
                    NoH_d_list.append(self.Cal_Distance(p0, p1))
                    NoH_a_list.append((NoH_obj_area_list[locs[0][j]] + NoH_obj_area_list[locs[0][j + 1]]) / 2)
                    NoH_Ismax_list.append((NoH_obj_Ismax_list[locs[0][j]] + NoH_obj_Ismax_list[locs[0][j + 1]]) / 2)
                    NoH_Iv_list.append((NoH_obj_Iv_list[locs[0][j]] + NoH_obj_Iv_list[locs[0][j + 1]]) / 2)
                    NoH_orientation_list.append(
                        (NoH_obj_orientation_list[locs[0][j]] + NoH_obj_orientation_list[locs[0][j + 1]]) / 2
                    )
                    NoH_aspectratio_list.append(
                        (NoH_obj_aspectratio_list[locs[0][j]] + NoH_obj_aspectratio_list[locs[0][j + 1]]) / 2
                    )
                    NoH_dates_list.append(self.date_average(
                        NoH_obj_dates_list[locs[0][j]], NoH_obj_dates_list[locs[0][j + 1]])
                    )
                    NoH_touched_list.append(
                        (NoH_obj_touched_list[locs[0][j]] + NoH_obj_touched_list[locs[0][j + 1]]) / 2
                    )
                    NoH_label_list.append(obj_no[i])

        return (NoH_touched_list, NoH_obj_touched_list, NoH_d_list, NoH_obj_centroid_list,
                NoH_centroid_list, NoH_obj_dates_list, NoH_dates_list, NoH_label_list,
                NoH_obj_label_list, NoH_obj_aspectratio_list, NoH_aspectratio_list,
                NoH_dir_list, NoH_obj_area_list, NoH_a_list, NoH_obj_Iv_list, NoH_Iv_list,
                NoH_obj_Ismax_list, NoH_Ismax_list, NoH_v_list, NoH_obj_orientation_list,
                NoH_orientation_list)

    def create_folder(self, output_address):
        """
        Create a folder if it doesn't exist.
        """
        li = output_address.rfind('/')
        output_dir = output_address[:li+1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def Read_date_from_filename(self, filename):
        """
        Extract start and end dates from the filename.
        """
        shift = 3
        fyear = int(filename[3-shift:7-shift])
        fmonth = int(filename[7-shift:9-shift])
        fday = int(filename[9-shift:11-shift])
        fhour = int(filename[12-shift:14-shift])
        fmin = int(filename[14-shift:16-shift])
        st_dt = datetime.datetime(fyear, fmonth, fday, fhour, fmin)
        fyear = int(filename[17-shift:21-shift])
        fmonth = int(filename[21-shift:23-shift])
        fday = int(filename[23-shift:25-shift])
        fhour = int(filename[26-shift:28-shift])
        fmin = int(filename[28-shift:30-shift])
        en_dt = datetime.datetime(fyear, fmonth, fday, fhour, fmin)
        return st_dt, en_dt

    def create_season_mask(self, seasons_months, months):
        """
        Create a mask array for the given season months.
        """
        dt_ar = np.copy(months)
        for month in seasons_months:
            dt_ar[dt_ar == month] = -1
        dt_ar[dt_ar != -1] = 0
        dt_ar[dt_ar == -1] = 1
        return dt_ar

    def break_to_seasons(self, dates_array):
        """
        Break dates into seasonal masks.
        """
        spring_months = np.array([9, 10, 11])
        summer_months = np.array([12, 1, 2])
        autumn_months = np.array([3, 4, 5])
        winter_months = np.array([6, 7, 8])
        dates = pd.DatetimeIndex(dates_array)
        months = np.asarray(dates.month)
        spring_mask = self.create_season_mask(spring_months, months)
        summer_mask = self.create_season_mask(summer_months, months)
        autumn_mask = self.create_season_mask(autumn_months, months)
        winter_mask = self.create_season_mask(winter_months, months)
        return spring_mask, summer_mask, autumn_mask, winter_mask

    def read_from_inside_file(self, file, dt):
        """
        Read dates from inside the NetCDF file.
        """
        Radarfile = xr.open_dataset(file)
        dtt = int(dt.total_seconds() // 60)
        rounded_time = Radarfile.time.dt.floor(str(dtt)+'min')
        dates_array = pd.to_datetime(rounded_time.values).round('min').to_pydatetime()
        st_date = np.min(dates_array)
        end_date = np.max(dates_array)
        return st_date, end_date, dates_array

    def process_files(self):
        """
        Process all files to calculate object properties.
        """
        merged_folder = self.merged_folder
        files = glob.glob(os.path.join(merged_folder, "*.nc"))
        files.sort()
        filenames = [os.path.basename(file) for file in files]
        filenames.sort()

        # Initialize lists to store properties for each season
        obj_area_list = [[] for _ in range(4)]
        a_list = [[] for _ in range(4)]
        obj_Iv_list = [[] for _ in range(4)]
        Iv_list = [[] for _ in range(4)]
        obj_Ismax_list = [[] for _ in range(4)]
        Ismax_list = [[] for _ in range(4)]
        v_list = [[] for _ in range(4)]
        d_list = [[] for _ in range(4)]
        obj_orientation_list = [[] for _ in range(4)]
        orientation_list = [[] for _ in range(4)]
        dir_list = [[] for _ in range(4)]
        aspectratio_list = [[] for _ in range(4)]
        obj_label_list = [[] for _ in range(4)]
        label_list = [[] for _ in range(4)]
        obj_dates_list = [[] for _ in range(4)]
        dates_list = [[] for _ in range(4)]
        obj_centroid_list0 = [[] for _ in range(4)]
        centroid_list0 = [[] for _ in range(4)]
        obj_aspectratio_list = [[] for _ in range(4)]
        obj_touched_list = [[] for _ in range(4)]
        touched_list = [[] for _ in range(4)]

        area_threshold = self.area_threshold
        # Loop over each file
        for j in range(len(files)):
            st_date0, end_date0 = self.Read_date_from_filename(filenames[j])
            dt = timedelta(minutes=30)
            st_date, end_date, dates_array = self.read_from_inside_file(files[j], dt)
            Radarfile = Dataset(files[j], 'r')
            MTD_Cube = Radarfile.variables['fcst_object_id'][:, :, :]
            Raw_Cube = Radarfile.variables['fcst_raw'][:, :, :] * MTD_Cube
            if len(MTD_Cube) > 2 and len(np.unique(MTD_Cube)) > 1:
                print(filenames[j])
                spring_mask, summer_mask, autumn_mask, winter_mask = self.break_to_seasons(dates_array)
                seasons_mask = [spring_mask, summer_mask, autumn_mask, winter_mask]
                seasons = ['SON', 'DJF', 'MAM', 'JJA']

                detlta_t = dt.seconds / 60

                for i in range(len(seasons)):
                    # Apply seasonal mask
                    MTD_Cube_seasonal = np.asarray(xr.DataArray(
                        seasons_mask[i]) * xr.DataArray(MTD_Cube))
                    if np.sum(MTD_Cube_seasonal) != 0:
                        print(seasons[i])
                        results = self.Storm_Info(
                            MTD_Cube_seasonal, Raw_Cube, detlta_t, dates_array)
                        (NoH_touched_list, NoH_obj_touched_list, NoH_d_list, NoH_obj_centroid_list,
                         NoH_centroid_list, NoH_obj_dates_list, NoH_dates_list, NoH_label_list,
                         NoH_obj_label_list, NoH_obj_aspectratio_list, NoH_aspectratio_list,
                         NoH_dir_list, NoH_obj_area_list, NoH_a_list, NoH_obj_Iv_list, NoH_Iv_list,
                         NoH_obj_Ismax_list, NoH_Ismax_list, NoH_v_list, NoH_obj_orientation_list,
                         NoH_orientation_list) = results

                        # Aggregate results for the season
                        obj_area_list[i] += NoH_obj_area_list
                        a_list[i] += NoH_a_list
                        obj_Iv_list[i] += NoH_obj_Iv_list
                        Iv_list[i] += NoH_Iv_list
                        obj_Ismax_list[i] += NoH_obj_Ismax_list
                        Ismax_list[i] += NoH_Ismax_list
                        v_list[i] += NoH_v_list
                        obj_orientation_list[i] += NoH_obj_orientation_list
                        orientation_list[i] += NoH_orientation_list
                        dir_list[i] += NoH_dir_list
                        aspectratio_list[i] += NoH_aspectratio_list
                        obj_aspectratio_list[i] += NoH_obj_aspectratio_list
                        obj_label_list[i] += NoH_obj_label_list
                        label_list[i] += NoH_label_list
                        obj_dates_list[i] += NoH_obj_dates_list
                        dates_list[i] += NoH_dates_list
                        obj_centroid_list0[i] += NoH_obj_centroid_list
                        centroid_list0[i] += NoH_centroid_list
                        d_list[i] += NoH_d_list
                        touched_list[i] += NoH_touched_list
                        obj_touched_list[i] += NoH_obj_touched_list

        # Prepare data for output
        obj_centroid_x_list = []
        obj_centroid_y_list = []
        centroid_x_list = []
        centroid_y_list = []
        for i in range(4):
            if len(obj_centroid_list0[i]) > 0:
                obj_centroid_x_list.append(np.asarray(obj_centroid_list0[i])[:, 1])
                obj_centroid_y_list.append(np.asarray(obj_centroid_list0[i])[:, 0])
            else:
                obj_centroid_x_list.append([])
                obj_centroid_y_list.append([])
            if len(centroid_list0[i]) > 0:
                centroid_x_list.append(np.asarray(centroid_list0[i])[:, 1])
                centroid_y_list.append(np.asarray(centroid_list0[i])[:, 0])
            else:
                centroid_x_list.append([])
                centroid_y_list.append([])
        Iave_list = []
        for i in range(4):
            Iave_list.append(np.asarray(Iv_list[i]) / (np.asarray(a_list[i]) * 1e6) * 1000)
        obj_Iave_list = []
        for s in range(4):
            obj_Iave_list.append(np.asarray(obj_Iv_list[s]) / (np.asarray(obj_area_list[s]) * 1e6) * 1000)

        Radar_data_obj_list = []
        Radar_data_ave_list = []
        # Create DataFrames for output
        for s in range(4):
            Radar_data_obj = pd.DataFrame()
            Radar_data_obj['touched_borders?'] = obj_touched_list[s]
            Radar_data_obj['Centroid_X'] = obj_centroid_x_list[s]
            Radar_data_obj['Centroid_Y'] = obj_centroid_y_list[s]
            Radar_data_obj['datetime'] = obj_dates_list[s]
            Radar_data_obj['label'] = obj_label_list[s]
            Radar_data_obj['aspectratio'] = obj_aspectratio_list[s]
            Radar_data_obj['area'] = obj_area_list[s]
            Radar_data_obj['Iv'] = obj_Iv_list[s]
            Radar_data_obj['Ismax'] = obj_Ismax_list[s]
            Radar_data_obj['orientation'] = obj_orientation_list[s]
            Radar_data_obj['Iave'] = obj_Iave_list[s]
            Radar_data_obj_list.append(Radar_data_obj)

            Radar_data_ave = pd.DataFrame()
            Radar_data_ave['touched_borders?'] = touched_list[s]
            Radar_data_ave['d'] = d_list[s]
            Radar_data_ave['Centroid_X'] = centroid_x_list[s]
            Radar_data_ave['Centroid_Y'] = centroid_y_list[s]
            Radar_data_ave['datetime'] = dates_list[s]
            Radar_data_ave['label'] = label_list[s]
            Radar_data_ave['aspectratio'] = aspectratio_list[s]
            Radar_data_ave['Velocity'] = v_list[s]
            Radar_data_ave['dir'] = dir_list[s]
            Radar_data_ave['area'] = a_list[s]
            Radar_data_ave['Iv'] = Iv_list[s]
            Radar_data_ave['Ismax'] = Ismax_list[s]
            Radar_data_ave['orientation'] = orientation_list[s]
            Radar_data_ave_list.append(Radar_data_ave)

        # Write data to .ftr files
        for s in range(4):
            Radar_data_obj = Radar_data_obj_list[s]
            Radar_data_ave = Radar_data_ave_list[s]
            Radar_data_obj = Radar_data_obj[Radar_data_obj.Ismax > 0].dropna()
            Radar_data_ave = Radar_data_ave[Radar_data_ave.Ismax > 0].dropna()
            Radar_data_ave = Radar_data_ave[Radar_data_ave.Velocity > 0].dropna()
            Radar_data_obj.drop_duplicates(subset=['Centroid_X', 'Centroid_Y', 'label'], inplace=True)
            Radar_data_ave.drop_duplicates(subset=['Centroid_X', 'Centroid_Y', 'label'], inplace=True)
            os.makedirs(self.output_folder_snapshots, exist_ok=True)
            os.makedirs(self.output_folder_averages, exist_ok=True)
            Radar_data_obj.reset_index().to_feather(
                os.path.join(self.output_folder_snapshots, f'Radar_data_obj_{s}.ftr')
            )
            Radar_data_ave.reset_index().to_feather(
                os.path.join(self.output_folder_averages, f'Radar_data_ave_{s}.ftr')
            )



# Usage
if __name__ == "__main__":
    mtd_folder = "/path/to/merged_convolved_maps_tracked/"
    output_folder_objects = "/path/to/Thunderstorm_properties_ftr/Objects/"
    output_folder_averages = "/path/to/Thunderstorm_properties_ftr/Averages/"
    area_threshold = 1  # Number of pixels
    pixel_resolution = 10  # In km
    time_resolution = 0.5  # In hours (e.g., 30 minutes)

    # Create an instance of the class with input files and output folder
    processor = ObjectPropertiesProcessor(
        merged_folder=mtd_folder,
        output_folder_snapshots=output_folder_objects,
        output_folder_averages=output_folder_averages,
        smaple_raw_data_address=Path("/path/to/sample_file.nc"),
        input_raster_main_field='precipitationCal',  # Adjust field name as needed
        area_threshold=area_threshold,
        pixel_resolution=pixel_resolution,
        time_resolution=time_resolution
    )
    processor.process_files()

import os
import sys
# Add the current working directory to sys.path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import glob
import xarray as xr
import pandas as pd
import re
import calendar

class MergerProcessor:
    def __init__(self, input_covolved_maps_folder_address: str, output_merged_maps_folder_address: str, time_gap_threshold_hours: float):
        """
        Initializes the MergerProcessor class.

        Parameters:
        - input_covolved_maps_folder_address (str): Path to the folder containing convolved map files.
        - output_merged_maps_folder_address (str): Path to the folder where merged maps will be saved.
        - time_gap_threshold_hours (float): Threshold in hours to determine the acceptable time gaps to merge the snapshots.
        """
        self.input_covolved_maps_folder_address = input_covolved_maps_folder_address
        self.output_merged_maps_folder_address = output_merged_maps_folder_address
        self.time_gap_threshold_hours = time_gap_threshold_hours

        # Ensure the output directory exists
        os.makedirs(self.output_merged_maps_folder_address, exist_ok=True)

        # Define the seasons and their corresponding months
        self.seasons = {
            "DJF": [12, 1, 2],   # December, January, February
            "MAM": [3, 4, 5],    # March, April, May
            "JJA": [6, 7, 8],    # June, July, August
            "SON": [9, 10, 11]   # September, October, November
        }
        
        # Regular expression to match the file name pattern with time in the format 'convolved_YYYYMMDDTHHMM.nc'
        # Captures year, month, day, hour, and minute
        self.pattern = re.compile(r"convolved_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\.nc")


    def open_seasonal_files(self, year, season):
        """
        Opens and concatenates files for a specific year and season.

        Parameters:
        - year (int): The year to process.
        - season (str): The season code (e.g., 'DJF', 'MAM', 'JJA', 'SON').

        Returns:
        - ds_list (list of xarray.Dataset): List of datasets split by time gaps larger than the threshold.
        """
        month_list = self.seasons[season]
        datasets = []  # List to store individual datasets

        # Get the list of all .nc files in the folder
        convolved_files_addresses = glob.glob(os.path.join(self.input_covolved_maps_folder_address, "*.nc"))
        
        # Loop through each file address
        for convolved_files_address in convolved_files_addresses:
            # Extract the time information from the filename using the regex pattern
            match = self.pattern.search(convolved_files_address)
            if match:
                file_year = int(match.group(1))
                file_month = int(match.group(2))

                # Check if the file corresponds to the specified season and year
                if file_year == year and file_month in month_list:
                    # Open the dataset and append to the list
                    ds = xr.open_dataset(convolved_files_address)
                    datasets.append(ds)

        if datasets:
            # Concatenate the datasets along the time dimension
            ds_combined = xr.concat(datasets, dim='time')
            precip = ds_combined["fcst_raw"]

            # Compute time differences and find gaps larger than the specified threshold
            # Convert time values to hours
            time_hours = pd.to_timedelta(precip.time.values.astype("int64"), unit="ns") / pd.Timedelta(hours=1)
            time_hours_series = pd.Series(time_hours, index=precip.time)
            time_diff = time_hours_series.diff()  # Calculate time differences between consecutive time points

            # Identify indices where the time difference exceeds the threshold
            gap_indices = time_diff.where(time_diff > self.time_gap_threshold_hours).dropna().index.values

            # Split the dataset at the gap indices to create datasets with no gaps larger than the threshold
            ds_list = [
                ds_combined.sel(time=slice(start_time, end_time))
                for start_time, end_time in zip(
                    [ds_combined.time.values[0]] + list(gap_indices),
                    list(gap_indices) + [ds_combined.time.values[-1]]
                )
            ]        
            return ds_list
        else:
            # Return an empty list if no datasets are found for the specified season and year
            return []

    def merge_files_by_season_and_year(self):
        """
        Merges files by season and year, splitting datasets where time gaps exceed the specified threshold.
        Saves the merged datasets to the output folder.
        """
        # Get all file names in the input folder and sort them
        all_file_names = os.listdir(self.input_covolved_maps_folder_address)
        all_file_names.sort()

        # Extract the start and end year using the regex pattern
        st_year_match = self.pattern.match(all_file_names[0])
        end_year_match = self.pattern.match(all_file_names[-1])

        if st_year_match and end_year_match:
            st_year = int(st_year_match.group(1))  # Group 1 corresponds to the year in the regex pattern
            end_year = int(end_year_match.group(1))
        else:
            print("File names do not match the expected pattern.")
            return

        # Loop through the years from start year to end year
        for year in range(st_year, end_year + 1):
            print(f'Year: {year}')
            
            # Loop through each season
            for season, months in self.seasons.items():
                # Determine the season year
                if months[0] <= 2:  # If the first month of the season is January or February
                    season_year = year - 1
                else:
                    season_year = year
                
                print(f'\t{season} {season_year}')
                
                # Open and process files for the given year and season
                ds_list = self.open_seasonal_files(year=year, season=season)
                
                # Loop through the datasets split by time gaps
                for m in range(len(ds_list)):
                    time_index = pd.to_datetime(ds_list[m].time.values)

                    # Get the first and last timestamps
                    if len(time_index) > 0:
                        first_time = time_index.min().strftime('%Y%m%d_%H%M')
                        if len(time_index) > 1:
                            # Get second-to-last time to avoid overlap
                            last_time = time_index[-2].strftime('%Y%m%d_%H%M')
                        else:
                            # Use the last time if only one timestamp exists
                            last_time = time_index[-1].strftime('%Y%m%d_%H%M')
                                
                        # Construct the output file path
                        output_file = os.path.join(
                            self.output_merged_maps_folder_address,
                            f'{first_time}_{last_time}.nc'
                        )
                        print(f'Saving file: {output_file}')
                        
                        # Save the dataset to a NetCDF file
                        ds_list[m].to_netcdf(output_file, encoding = self.compress_enchoding(ds_list[m]))


    def compress_enchoding(self, xr_data):
        # Define the compression settings
        compression_settings = {
            'zlib': True,        # Enable zlib compression
            'complevel': 4       # Compression level (1-9); higher means more compression but slower
        }
        # Create an encoding dictionary for all variables in the dataset
        encoding = {var: compression_settings for var in xr_data.data_vars}
        return encoding

# Usage example
if __name__ == "__main__":
    # Define input and output folder paths
    input_path = r'/path/to/convolved_maps'  # Replace with your actual input folder path
    output_path = r'/path/to/merged_convolved_maps'  # Replace with your actual output folder path
    time_gap_threshold_hours = 1.0  # Threshold for time gap in hours

    # Create an instance of the MergerProcessor class
    processor = MergerProcessor(
        input_covolved_maps_folder_address=input_path,
        output_merged_maps_folder_address=output_path,
        time_gap_threshold_hours=time_gap_threshold_hours
    )

    # Call the method to merge files by season and year
    processor.merge_files_by_season_and_year()

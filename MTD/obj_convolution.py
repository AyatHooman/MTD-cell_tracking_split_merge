import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage

class ConvolutionProcessor:
    def __init__(self, 
                 input_folder: str, 
                 output_folder: str, 
                 input_raster_main_field: str, 
                 input_raster_time_field: str,
                 R=3, 
                 Th=0.1):
        """
        Initialize the PrecipitationConvolver class with the necessary parameters.
        
        Parameters:
        - input_folder: Folder path containing the input NetCDF files.
        - output_folder: Folder path where the processed NetCDF files will be saved.
        - input_raster_main_field: The main field in the input raster dataset to apply convolution on.
        - input_raster_time_field: The time dimension field in the input raster dataset.
        - R: Radius for the uniform filter convolution.
        - Th: Threshold for filtering precipitation values.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_raster_main_field = input_raster_main_field
        self.input_raster_time_field = input_raster_time_field
        self.R = R
        self.Th = Th
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        # Collect and sort all NetCDF files from the input folder
        self.files = glob.glob(os.path.join(input_folder, '*'))
        self.files.sort()

    def convolve(self, two_d_arr):
        """
        Perform a uniform filter convolution on the input 2D array.

        Parameters:
        - two_d_arr: A 2D numpy array of precipitation data.

        Returns:
        - convolved: A 2D numpy array after applying the convolution and thresholding.
        """
        # Apply a uniform filter with a window size of 2*R + 1
        convolved = ndimage.uniform_filter(two_d_arr, size=2*self.R+1, mode='constant')
        # Threshold values: below the threshold are set to 0, others to 1
        convolved[convolved < self.Th] = 0
        convolved[convolved > 0] = 1
        return convolved

    def process_single_file(self, file_path):
        """
        Process a single NetCDF file: load, apply convolution, and save.

        Parameters:
        - file_path: Path to the NetCDF file to be processed.
        """
        # Load the dataset from the file and squeeze the time dimension
        rad_day = xr.open_dataset(file_path)
        rad_day = rad_day.squeeze(self.input_raster_time_field)
        # Extract the time information from the dataset
        file_time = pd.to_datetime(rad_day.time.values)
        # Format the time to use in the file name (up to the minute)
        formatted_time = file_time.strftime('%Y%m%dT%H%M')

        # Fill missing values with 0 and ensure non-negative values
        ds = rad_day.fillna(0).where(rad_day.fillna(0) >= 0, 0)

        # Apply the convolve function to the precipitation rate data
        ds_convolved = xr.apply_ufunc(
            self.convolve,
            ds[self.input_raster_main_field],
            dask='parallelized',
            output_dtypes=[np.float32]
        )

        # If the convolved dataset contains any non-zero values, process it
        if ds_convolved.sum().values > 0:
            print(f"Processing file: {file_path}")

            # Add the extracted time as a variable in the convolved dataset
            ds_convolved = ds_convolved.assign_coords(time=file_time)

            # Create a mask based on the convolved data (binary mask)
            ds_mask = ds_convolved.fillna(0).where(ds_convolved <= 0, 1)
            # Create a copy of the original dataset for final output
            ds_hr = ds.fillna(0)
            ds_hr_convolved = ds_hr
            # Add the mask as a new variable in the dataset
            ds_hr_convolved['fcst_object_id'] = ds_mask
            # Rename the original precipitation variable for clarity
            ds_hr_convolved = ds_hr_convolved.rename({self.input_raster_main_field: 'fcst_raw'})

            # Assign the time coordinate to the final dataset
            ds_hr_convolved = ds_hr_convolved.assign_coords(time=[file_time])

            # Save the processed dataset with the time formatted in the filename
            saving_file = os.path.join(self.output_folder, f"convolved_{formatted_time}.nc")


            ds_hr_convolved.to_netcdf(saving_file, encoding = self.compress_enchoding(ds_hr_convolved))
            print(f"Saved convolved file to: {saving_file}")

    def compress_enchoding(self, xr_data):
        # Define the compression settings
        compression_settings = {
            'zlib': True,        # Enable zlib compression
            'complevel': 4       # Compression level (1-9); higher means more compression but slower
        }
        # Create an encoding dictionary for all variables in the dataset
        encoding = {var: compression_settings for var in xr_data.data_vars}
        return encoding

    def run(self):
        """
        Process all files in the input folder by applying the convolution function
        and saving the results in the output folder.
        """
        for file_path in self.files:
            self.process_single_file(file_path)

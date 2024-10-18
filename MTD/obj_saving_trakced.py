import numpy as np
from netCDF4 import Dataset
from skimage.measure import label, regionprops
from collections import OrderedDict
import glob
import os
import xarray as xr

class ObjectTrackerProcessor:
    def __init__(self, output_merged_maps_folder_address: str, output_merged_maps_trcked_folder_address: str, connections_folder: str, label_start: int = 0):
        """
        Initializes the ObjectTrackerProcessor class.

        Parameters:
            output_merged_maps_folder_address (str): Path to the folder containing merged map files.
            output_merged_maps_trcked_folder_address (str): Path to the folder where tracked merged maps will be saved.
            connections_folder (str): Path to the folder where connections will be saved.
            label_start (int): Starting label number for object identification.
        """
        self.output_merged_maps_folder_address = output_merged_maps_folder_address
        self.output_merged_maps_trcked_folder_address = output_merged_maps_trcked_folder_address
        self.connections_folder = connections_folder

        # Ensure output directories exist
        os.makedirs(self.output_merged_maps_trcked_folder_address, exist_ok=True)
        os.makedirs(self.connections_folder, exist_ok=True)

        # Initialize label counter and connection list
        self.label_0 = label_start
        self.connection_list = []

    def Create_3D_Mask(self, MTD_Cube, label_0):
        """
        Creates a 3D mask for tracking objects over time and identifies connections between objects.

        Parameters:
            MTD_Cube (numpy.ndarray): 3D array of objects identified at each time step.
            label_0 (int): Starting label number for object identification.

        Returns:
            Mod_Objects (list): List of labeled objects for each time step.
            Connections (list): List of tuples representing connections between objects across time steps.
        """
        # Lists to store labeled objects and their properties for all time steps
        Mod_Objects = []
        Mod_Objects_prop_list = []
        Mod_Objects_prop_list_timestep = []

        # Copy of the input cube to avoid modifying the original data
        Objects = np.copy(MTD_Cube)

        # Initialize object ID counter
        IDNo = label_0

        # List to store connections between objects (merged or separated)
        appendedpoints = []

        # Counters for progress tracking
        file_counter = 0
        batch_counter = 0

        # Loop over each time step
        for step in range(len(Objects)):
            # Print progress every 5000 steps
            if file_counter == 5000:
                batch_counter += 1
                print("MTD Mask Creating Step=" + str(step))
                file_counter = 0
            file_counter += 1

            if step == 0:
                # First time step: label objects and assign unique IDs
                label_image = np.asarray(label(Objects[step])) + label_0
                label_image[label_image == label_0] = 0

                # Find connected objects using labeled image
                obj_no_list = []
                equi_obj_no_list = []
                obj_list_label_image = np.unique(label_image)
                obj_list_label_image = obj_list_label_image[obj_list_label_image > 0]

                for obj_no in obj_list_label_image:
                    temp_label_image = np.copy(label_image)
                    temp_label_image[temp_label_image != obj_no] = 0
                    temp_label_image[temp_label_image == obj_no] = 1
                    equi_obj_no = np.max(temp_label_image * Objects[step])
                    obj_no_list.append(obj_no)
                    equi_obj_no_list.append(equi_obj_no)

                # Append labeled image and properties to lists
                Mod_Objects.append(label_image)
                Mod_Objects_prop_list.append(regionprops(label_image))
                Mod_Objects_prop_list_timestep.append(step)

                # Update label counter
                prop = regionprops(label_image)
                IDNo += int(len(prop))

            if step > 0:
                # Subsequent time steps: identify connected objects across time
                label_image = label(Objects[step])
                label_image_prop = regionprops(label_image)
                Mod_Objects_prop = regionprops(Mod_Objects[step - 1])

                # Create a temporary 3D object to find connections between time steps
                Temp3Dobj = []
                Temp3Dobj.append(np.copy(Mod_Objects[step - 1]))
                Temp3Dobj.append(np.copy(label_image))
                Temp3Dobj[0][Temp3Dobj[0] > 0] = 1
                Temp3Dobj[1][Temp3Dobj[1] > 0] = 1

                # Label the 3D object
                label_image3D = label(np.asarray(Temp3Dobj))
                label_image3D_prop = regionprops(label_image3D)
                counter = 0

                # Loop over each connected component in the 3D labeled image
                for label_image3D_member in label_image3D_prop:
                    counter += 1

                    # Check if the object spans across two time steps
                    if label_image3D_member.bbox[3] == 2 and label_image3D_member.bbox[0] == 0:
                        # Extract the object slices for the two time steps
                        imagevaset0 = np.copy(label_image3D_member._label_image[0])
                        imagevaset0[imagevaset0 != label_image3D_member.label] = 0
                        label_image3D_slice0 = regionprops(np.asarray(imagevaset0))

                        imagevaset1 = np.copy(label_image3D_member._label_image[1])
                        imagevaset1[imagevaset1 != label_image3D_member.label] = 0
                        label_image3D_slice1 = regionprops(np.asarray(imagevaset1))

                        # Initialize flags for merging and separation
                        MergFinder = 1
                        SeperationFinder = 1

                        # Find corresponding object in the previous time step
                        for Mod_Objects_prop_member in Mod_Objects_prop:
                            if label_image3D_slice0[0].centroid == Mod_Objects_prop_member.centroid:
                                newlabel = Mod_Objects_prop_member.label
                                MergFinder = 0

                        # Find corresponding object in the current time step
                        for label_image_prop_member in label_image_prop:
                            if label_image3D_slice1[0].centroid == label_image_prop_member.centroid:
                                oldlabel = label_image_prop_member.label
                                SeperationFinder = 0

                        # Handle merged objects
                        if MergFinder == 1 and SeperationFinder != 1:
                            IDNo += 1
                            newlabel = IDNo

                            # Find connections between objects
                            Devided_Merged_Last_Step_prop = regionprops(label(np.asarray(imagevaset0)))
                            for obj0 in Devided_Merged_Last_Step_prop:
                                for obj0ref in Mod_Objects_prop:
                                    if obj0.centroid == obj0ref.centroid:
                                        if (obj0ref.label, newlabel) not in appendedpoints and (newlabel, obj0ref.label) not in appendedpoints:
                                            appendedpoints.append((obj0ref.label, newlabel))

                        # Handle separated objects
                        if SeperationFinder == 1:
                            oldlabellist = []
                            newlabellist = []
                            centroids1 = []
                            Area1 = []
                            plist = []
                            ID1 = []

                            Devided_Merged_Current_Step_prop = regionprops(label(np.asarray(imagevaset1)))
                            for objj in Devided_Merged_Current_Step_prop:
                                for label_image_prop_member in label_image_prop:
                                    if objj.centroid == label_image_prop_member.centroid:
                                        plist.append(label_image_prop_member.centroid)
                                        oldlabellist.append(label_image_prop_member.label)
                                        IDNo += 1
                                        newlabellist.append(IDNo)
                                        centroids1.append(label_image_prop_member.centroid)
                                        Area1.append(label_image_prop_member.area)
                                        ID1.append(IDNo)

                        # Update labels in the current time step
                        if SeperationFinder == 0:
                            label_image[label_image == oldlabel] = -1 * newlabel
                        if SeperationFinder == 1:
                            for iiii, labels in enumerate(oldlabellist):
                                label_image[label_image == labels] = -1 * newlabellist[iiii]

                            # Record connections between objects
                            Area0 = []
                            centroids0 = []
                            ID0 = []
                            Devided_Merged_Last_Step_prop = regionprops(label(np.asarray(imagevaset0)))
                            for obj0 in Devided_Merged_Last_Step_prop:
                                for obj0ref in Mod_Objects_prop:
                                    if obj0.centroid == obj0ref.centroid:
                                        Area0.append(obj0ref.area)
                                        centroids0.append(obj0ref.centroid)
                                        ID0.append(obj0ref.label)

                            for r0 in range(len(Area0)):
                                for r1 in range(len(Area1)):
                                    p0 = ID0[r0]
                                    p1 = ID1[r1]
                                    if (p0, p1) not in appendedpoints and (p1, p0) not in appendedpoints:
                                        appendedpoints.append((p0, p1))

                    # Handle new objects appearing in the current time step
                    if label_image3D_member.bbox[3] == 2 and label_image3D_member.bbox[0] == 1:
                        IDNo += 1
                        for label_image_prop_member in label_image_prop:
                            x = label_image3D_member.centroid[1]
                            y = label_image3D_member.centroid[2]
                            if (x, y) == label_image_prop_member.centroid:
                                oldlabel = label_image_prop_member.label
                                newlabel = IDNo
                        label_image[label_image == oldlabel] = -1 * newlabel

                # Convert labels back to positive values
                label_image = np.absolute(label_image)

                # Append the labeled image for the current time step
                Mod_Objects.append(label_image)

        # Remove duplicate connections
        Connections = list(OrderedDict.fromkeys(appendedpoints))
        return Mod_Objects, Connections

    def process_files(self):
        """
        Processes all files in the merged maps folder and tracks objects across time steps.
        """
        # Get list of merged map files
        merged_address_files = glob.glob(self.output_merged_maps_folder_address + r"\*.nc")
        merged_address_files.sort()
        merged_filenames = os.listdir(self.output_merged_maps_folder_address)
        merged_filenames.sort()

        # Loop over each file
        for i in range(len(merged_filenames)):
            print(merged_filenames[i])

            # Open the dataset
            Radarfile = xr.open_dataset(merged_address_files[i])

            # Extract object labels and raw data, replacing NaNs with zeros
            MTD_Cube = np.nan_to_num(Radarfile['fcst_object_id'].values, nan=0)
            Raw_Cube = Radarfile['fcst_raw'].values

            # Create the 3D mask and get connections
            Mod_Objects, Connections = self.Create_3D_Mask(MTD_Cube, self.label_0)

            # Update the label counter
            if len(np.unique(Mod_Objects)) > 1:
                self.label_0 = np.max(np.unique(Mod_Objects))

            # Convert Mod_Objects to a NumPy array
            Mod_Objects = np.asarray(Mod_Objects)

            # Save the tracked objects back into the dataset
            outputaddress = os.path.join(self.output_merged_maps_trcked_folder_address, merged_filenames[i])
            Radarfile['fcst_object_id'].values = Mod_Objects
            Radarfile['fcst_raw'].values = Raw_Cube
            Radarfile.to_netcdf(outputaddress, self.compress_enchoding(Radarfile))

            # Add connections to the connection list
            if len(Connections) > 0:
                self.connection_list += list(Connections)

        # Save all connections to a file
        connections_array = np.array(self.connection_list, dtype=object)
        np.save(os.path.join(self.connections_folder, 'connections.npy'), connections_array)

    def compress_enchoding(xr_data):
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
    output_merged_maps_folder_address = r'C:\Users\h.ayat\Desktop\Object_tracking\MRMS-Sample_data\outputs\merged_convolved_maps'
    saving_folder = r'C:\Users\h.ayat\Desktop\Object_tracking\MRMS-Sample_data\outputs\merged_convolved_maps_tracked'
    connections_folder = r'C:\Users\h.ayat\Desktop\Object_tracking\MRMS-Sample_data\outputs\connections_folder'

    # Create an instance of the ObjectTrackerProcessor
    tracker = ObjectTrackerProcessor(
        output_merged_maps_folder_address,
        saving_folder,
        connections_folder
    )

    # Process the files to track objects
    tracker.process_files()

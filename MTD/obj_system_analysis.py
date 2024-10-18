import glob
import numpy as np
import os
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from datetime import timedelta
import subprocess
import pandas as pd
import xarray as xr
import networkx as nx
from networkx.algorithms.dag import dag_longest_path
from netCDF4 import Dataset, num2date, date2num
import pickle

class SystemPropertiesProcessor:
    def __init__(self, output_folder_snapshots, output_folder_averages, output_path, connection_path):
        """
        Initializes the SystemPropertiesProcessor class.

        Parameters:
        - output_folder_snapshots (str): Path to the folder containing snapshot data.
        - output_folder_averages (str): Path to the folder containing averaged data.
        - output_path (str): Path where the output results will be saved.
        - connection_path (str): Path to the connections data file.
        """
        self.seasons = ['SON', 'DJF', 'MAM', 'JJA']
        self.Radar_data_obj_list = []
        self.Radar_data_ave_list = []
        for s in range(4):
            # Load snapshot and average data for each season
            self.Radar_data_obj_list.append(pd.read_feather(
                os.path.join(output_folder_snapshots, f'Radar_data_obj_{s}.ftr')))
            self.Radar_data_ave_list.append(pd.read_feather(
                os.path.join(output_folder_averages, f'Radar_data_ave_{s}.ftr')))
        # Load connections data
        self.connections = np.load(os.path.join(connection_path, 'connections.npy'), allow_pickle=True)
        self.output_path = output_path

    def find_length(self, label, label_list, season):
        """
        Calculate the total track length for a given label.

        Parameters:
        - label (int): The label of the object.
        - label_list (array): List of labels.
        - season (int): Index of the season.

        Returns:
        - float: Sum of distances for the given label.
        """
        labels = np.copy(label_list)
        labels[labels != label] = 0
        labels[labels == label] = 1
        return np.sum(self.Radar_data_ave_list[season]['d'].values * labels)

    def add_virtual_node(self, G_sub, label_list, s):
        """
        Add a virtual node to the graph to compute the longest path.

        Parameters:
        - G_sub (networkx.DiGraph): Subgraph of the main graph.
        - label_list (array): List of labels.
        - s (int): Season index.

        Returns:
        - networkx.DiGraph: Modified subgraph with a virtual node added.
        """
        G_sub_copy = G_sub.copy()
        leaves = [v for v, d in G_sub.out_degree() if d == 0]
        virtual_node = np.max(list(G_sub)) + 1
        for leaf in leaves:
            G_sub_copy.add_edge(leaf, virtual_node,
                                weight=self.find_length(leaf, label_list, s))
        return G_sub_copy

    def select_elements(self, ref_array, lb_array, selected_nodes):
        """
        Select elements from an array based on the selected nodes.

        Parameters:
        - ref_array (array): Reference array.
        - lb_array (array): Label array.
        - selected_nodes (array): Array of selected node labels.

        Returns:
        - array: Selected elements.
        """
        arr = np.copy(lb_array)
        indexes = np.where(np.isin(arr, selected_nodes) == True)[0]
        selected_elems = np.asarray(lb_array)[indexes]
        return selected_elems

    def create_label_mask(self, lb_array, selected_nodes):
        """
        Create a mask for the labels of selected nodes.

        Parameters:
        - lb_array (array): Label array.
        - selected_nodes (array): Array of selected node labels.

        Returns:
        - array: Mask array with 1s for selected labels and 0s elsewhere.
        """
        arr = np.copy(lb_array)
        indexes = np.where(np.isin(arr, selected_nodes) == True)[0]
        mask_array = np.zeros(len(lb_array), dtype=int)
        mask_array[indexes] = 1
        return mask_array

    def calc_stormlifetime(self, timestep_array, mask_array):
        """
        Calculate the lifetime of a storm based on time steps and mask.

        Parameters:
        - timestep_array (array): Array of time steps.
        - mask_array (array): Mask array indicating the storm presence.

        Returns:
        - float: Storm lifetime.
        """
        timestep_array_masked = timestep_array * mask_array
        ts = timestep_array_masked[timestep_array_masked != 0]
        return np.max(ts) - np.min(ts)

    def Cal_Velocity(self, point1, point2, dt):
        """
        Calculate velocity between two points.

        Parameters:
        - point1 (tuple): Coordinates of the first point (y, x).
        - point2 (tuple): Coordinates of the second point (y, x).
        - dt (float): Time difference in minutes.

        Returns:
        - float: Velocity.
        """
        dy = point2[0] - point1[0]
        dx = point2[1] - point1[1]
        speed = np.hypot(dx, dy) / (dt / 60)
        return speed

    def calc_speed(self, x0, x1, y0, y1, dt):
        """
        Calculate speed between two coordinates.

        Parameters:
        - x0, x1 (float): x-coordinates.
        - y0, y1 (float): y-coordinates.
        - dt (float): Time difference in minutes.

        Returns:
        - float: Speed.
        """
        dx = x1 - x0
        dy = y1 - y0
        return np.hypot(dx, dy) / (dt / 60)

    def calc_angle(self, x1, y1, x2, y2):
        """
        Calculate the angle (in degrees) between two points.

        Parameters:
        - x1, y1 (float): Coordinates of the first point.
        - x2, y2 (float): Coordinates of the second point.

        Returns:
        - float: Angle in degrees.
        """
        radians = math.atan2(y2 - y1, x2 - x1)
        degrees = math.degrees(radians)
        return degrees

    def mean_centroid(self, p0, p1):
        """
        Calculate the mean centroid between two points.

        Parameters:
        - p0 (tuple): First point (y, x).
        - p1 (tuple): Second point (y, x).

        Returns:
        - list: Mean centroid coordinates [y, x].
        """
        xm = (p0[1] + p1[1]) / 2
        ym = (p0[0] + p1[0]) / 2
        return [ym, xm]

    def calc_storm_centroid_and_mean_properties(self, arr_centroid_selected, obj_dates_list_minutes_selected, arr_A_selected, arr_Iv_selected, arr_Ismax_selected):
        """
        Calculate storm centroid and mean properties over time.

        Parameters:
        - arr_centroid_selected (array): Selected centroids.
        - obj_dates_list_minutes_selected (array): Selected dates in minutes.
        - arr_A_selected (array): Selected areas.
        - arr_Iv_selected (array): Selected intensities.
        - arr_Ismax_selected (array): Selected maximum intensities.

        Returns:
        - tuple: Arrays of calculated properties.
        """
        dates = np.unique(obj_dates_list_minutes_selected)
        centroid_list0 = []
        centroid_list0_m = []
        velocity_list_m = []
        dir_list_m = []
        area_list_m = []
        Iv_list_m = []
        Is_max = []
        Is_max_m = []
        date_list_m = []

        for d in range(len(dates)):
            date_mask = np.copy(obj_dates_list_minutes_selected)
            date_mask[date_mask != dates[d]] = 0
            date_mask[date_mask == dates[d]] = 1
            centroids_one_snapshot = arr_centroid_selected[date_mask == 1]
            areas_one_snapshot = arr_A_selected[date_mask == 1]
            # Calculate weighted centroid
            y = np.sum(centroids_one_snapshot[:, 0] * areas_one_snapshot) / np.sum(areas_one_snapshot)
            x = np.sum(centroids_one_snapshot[:, 1] * areas_one_snapshot) / np.sum(areas_one_snapshot)
            centroid_list0.append([y, x])
            Is_max.append(np.max(arr_Ismax_selected[date_mask == 1]))
            if d > 0:
                dt = dates[d] - dates[d - 1]
                # Calculate velocity
                Sp = self.Cal_Velocity(centroid_list0[-2], centroid_list0[-1], dt)
                centroid_list0_m.append(self.mean_centroid(centroid_list0[-2], centroid_list0[-1]))
                velocity_list_m.append(Sp)
                p0 = centroid_list0[-2]
                p1 = centroid_list0[-1]
                x1, y1, x2, y2 = p0[1], p0[0], p1[1], p1[0]
                dir_list_m.append(self.calc_angle(x1, y1, x2, y2))
                area_list_m.append((arr_A_selected[d] + arr_A_selected[d - 1]) / 2)
                Iv_list_m.append((arr_Iv_selected[d] + arr_Iv_selected[d - 1]) / 2)
                Is_max_m.append((Is_max[-1] + Is_max[-2]) / 2)
                date_list_m.append((dates[d] + dates[d - 1]) / 2)
        return (np.asarray(centroid_list0_m), np.asarray(centroid_list0), np.asarray(velocity_list_m),
                np.asarray(dir_list_m), np.asarray(area_list_m), np.asarray(Iv_list_m),
                np.asarray(Is_max), np.asarray(Is_max_m), np.asarray(date_list_m))

    def create_timestep_storm_area_Iv_Ismax(self, obj_dates_list_minutes, obj_mask_label, arr_A, arr_Iv, arr_centroid, arr_Ismax):
        """
        Create time step data for storm area, intensity, and maximum intensity.

        Parameters:
        - obj_dates_list_minutes (array): Dates in minutes.
        - obj_mask_label (array): Mask for object labels.
        - arr_A (array): Areas.
        - arr_Iv (array): Intensities.
        - arr_centroid (array): Centroids.
        - arr_Ismax (array): Maximum intensities.

        Returns:
        - tuple: Arrays of calculated properties.
        """
        A = obj_mask_label * obj_dates_list_minutes
        obj_dates_list_minutes_selected_mask = np.copy(A)
        obj_dates_list_minutes_selected_mask[obj_dates_list_minutes_selected_mask > 0] = 1
        obj_dates_list_minutes_selected = A[A > 0]
        dates = np.unique(obj_dates_list_minutes_selected)
        arr_A_selected = np.asarray(arr_A)[obj_dates_list_minutes_selected_mask == 1]
        area_list = np.bincount(np.searchsorted(dates, obj_dates_list_minutes_selected), arr_A_selected)
        arr_Iv_selected = np.asarray(arr_Iv)[obj_dates_list_minutes_selected_mask == 1]
        Iv_list = np.bincount(np.searchsorted(dates, obj_dates_list_minutes_selected), arr_Iv_selected.astype(float))
        arr_Ismax_selected = np.asarray(arr_Ismax)[obj_dates_list_minutes_selected_mask == 1]
        arr_centroid_selected = arr_centroid[obj_dates_list_minutes_selected_mask != 0]
        results = self.calc_storm_centroid_and_mean_properties(
            arr_centroid_selected, obj_dates_list_minutes_selected, arr_A_selected, arr_Iv_selected, arr_Ismax_selected)
        return results

    def selected_point_centroid(self, centroid_list, mask_label):
        """
        Select centroids based on a mask.

        Parameters:
        - centroid_list (array): List of centroids.
        - mask_label (array): Mask array.

        Returns:
        - array: Selected centroids.
        """
        yy = centroid_list[:, 0] * mask_label
        xx = centroid_list[:, 1] * mask_label
        y = yy[yy != 0]
        x = xx[xx != 0]
        centroids = np.array(list(zip(y, x)))
        return centroids

    def best_fit(self, X, Y):
        """
        Perform linear regression to find the best fit line.

        Parameters:
        - X (array): Independent variable data.
        - Y (array): Dependent variable data.

        Returns:
        - tuple: Intercept (a) and slope (b) of the line.
        """
        xbar = np.mean(X)
        ybar = np.mean(Y)
        n = len(X)
        numer = np.sum(X * Y) - n * xbar * ybar
        denum = np.sum(X ** 2) - n * xbar ** 2
        b = numer / denum
        a = ybar - b * xbar
        return a, b

    def calc_compo_x(self, dirr):
        """
        Calculate the x-component of a vector given its direction.

        Parameters:
        - dirr (float): Direction in radians.

        Returns:
        - float: x-component.
        """
        return np.cos(dirr)

    def calc_compo_y(self, dirr):
        """
        Calculate the y-component of a vector given its direction.

        Parameters:
        - dirr (float): Direction in radians.

        Returns:
        - float: y-component.
        """
        return np.sin(dirr)

    def calc_storm_total_velocity_direction_area(self, x0, y0, z0, a0):
        """
        Calculate the total velocity, direction, and area of a storm.

        Parameters:
        - x0 (array): x-coordinates.
        - y0 (array): y-coordinates.
        - z0 (array): Time steps.
        - a0 (array): Areas.

        Returns:
        - tuple: Storm velocity and direction.
        """
        x = np.copy(x0)
        y = np.copy(y0)
        z = np.copy(z0)
        a = np.copy(a0)
        dirrx = 0
        dirry = 0
        area = 0
        vel = 0
        for t in range(len(x) - 1):
            x1, y1, x2, y2 = x[t], y[t], x[t + 1], y[t + 1]
            myradians = math.atan2(y2 - y1, x2 - x1)
            dirrx += self.calc_compo_x(myradians) * (a[t] + a[t + 1]) / 2
            dirry += self.calc_compo_y(myradians) * (a[t] + a[t + 1]) / 2
            dx = x[t + 1] - x[t]
            dy = y[t + 1] - y[t]
            dl = np.hypot(dx, dy)
            dt = (z[t + 1] - z[t]) / 60
            vel += dl / dt * (a[t] + a[t + 1]) / 2
            area += (a[t] + a[t + 1]) / 2
        u = dirrx / area
        v = dirry / area
        storm_direction = math.degrees(math.atan2(v, u))
        storm_velocity = vel / area
        return storm_velocity, storm_direction

    def convert2min(self, date):
        """
        Convert datetime to minutes since a base time.

        Parameters:
        - date (array): Array of datetime objects.

        Returns:
        - array: Array of minutes since base time.
        """
        base_datetime = np.datetime64('1970-01-01T00:00:00')
        return (np.asarray(date) - base_datetime) / np.timedelta64(1, 'm')

    def revesred_convert2min(self, date_min):
        """
        Convert minutes since base time back to datetime.

        Parameters:
        - date_min (float): Minutes since base time.

        Returns:
        - datetime64: Corresponding datetime object.
        """
        return np.datetime64('1970-01-01T00:00:00') + date_min * np.timedelta64(1, 'm')

    def run_analysis(self):
        """
        Run the analysis to process system properties and save results.
        """
        seasons = self.seasons
        Radar_data_obj_list = self.Radar_data_obj_list
        Radar_data_ave_list = self.Radar_data_ave_list
        connections = self.connections

        stormtracklength_list = []
        nsplitmerge_list = []
        roots = []
        nroots_list = []
        stormlifetime_list = []

        for s in range(4):
            if len(Radar_data_ave_list[s]) > 0:
                Allnodes = np.unique(Radar_data_obj_list[s].label)
                G = nx.Graph()
                GD = nx.DiGraph()
                G.add_nodes_from(Allnodes)
                GD.add_nodes_from(Allnodes)

                file_counter = 0
                batch_counter = 0

                # Build graphs based on connections
                for i in range(len(connections)):
                    if file_counter == 10000:
                        batch_counter += 1
                        print(f"{seasons[s]} Creating Graphs: {file_counter * batch_counter}")
                        file_counter = 0
                    file_counter += 1

                    Cz = np.asarray(connections[i])
                    if (Cz[0] in Allnodes) and (Cz[1] in Allnodes):
                        G.add_edge(Cz[0], Cz[1])
                        GD.add_edge(Cz[0], Cz[1], weight=self.find_length(
                            Cz[0], Radar_data_ave_list[s].label.values, s))

                # Analyze connected components
                groups_of_groups = [list(c) for c in sorted(
                    nx.connected_components(G), key=len, reverse=True)]
                stormtracklength_list_temp = []
                nsplitmerge_list_temp = []
                roots_temp = []
                nroots_list_temp = []
                stormlifetime_list_temp = []
                file_counter = 0
                batch_counter = 0

                obj_dates_list_minutes = (Radar_data_obj_list[s].datetime.values - np.datetime64(
                    '1970-01-01T00:00:00')) / np.timedelta64(1, 'm')

                for j in range(len(groups_of_groups)):
                    if file_counter == 1000:
                        batch_counter += 1
                        print(f"{seasons[s]} analysing groups: {file_counter * batch_counter} from {len(groups_of_groups)}")
                        file_counter = 0
                    file_counter += 1

                    G_sub = GD.subgraph(groups_of_groups[j])
                    obj_mask_label = self.create_label_mask(
                        Radar_data_obj_list[s].label.values, list(G_sub.nodes()))
                    if np.sum(obj_mask_label) > 1:
                        lf = self.calc_stormlifetime(obj_dates_list_minutes, obj_mask_label)
                        stormlifetime_list_temp.append(lf)
                        nsplitmerge_list_temp.append(len(list(G_sub.edges)))
                        roots_temp.append([v for v, d in G_sub.in_degree() if d == 0])
                        nroots_list_temp.append(len([v for v, d in G_sub.in_degree() if d == 0]))
                        G_sub_added_virtual_node = self.add_virtual_node(
                            G_sub, Radar_data_ave_list[s].label.values, s)
                        # Calculate longest track length
                        selected_nodes_for_track_length = dag_longest_path(
                            G_sub_added_virtual_node, weight='weight')
                        selected_nodes = np.copy(selected_nodes_for_track_length[:-1])
                        lengest_track_mask_label = self.create_label_mask(
                            Radar_data_ave_list[s].label.values, selected_nodes)
                        stormtracklength = np.sum(
                            lengest_track_mask_label * Radar_data_ave_list[s]['d'].values)
                        stormtracklength_list_temp.append(stormtracklength)
                stormtracklength_list.append(stormtracklength_list_temp)
                nsplitmerge_list.append(nsplitmerge_list_temp)
                roots.append(roots_temp)
                nroots_list.append(nroots_list_temp)
                stormlifetime_list.append(stormlifetime_list_temp)
            else:
                stormtracklength_list.append([])
                nsplitmerge_list.append([])
                roots.append([])
                nroots_list.append([])
                stormlifetime_list.append([])

        # Save results
        output_path = self.output_path
        os.makedirs(output_path, exist_ok=True)
        # Save stormtracklength_list
        with open(os.path.join(output_path, 'stormtracklength_list.pkl'), 'wb') as file:
            pickle.dump(stormtracklength_list, file)
        # Save other results as needed
        # ...

# Usage
if __name__ == "__main__":
    output_folder_snapshots = '/path/to/Thunderstorm_properties_ftr/Objects/'
    output_folder_averages = '/path/to/Thunderstorm_properties_ftr/Averages/'
    output_path = '/path/to/Storms/'
    connection_path = '/path/to/connections_folder'

    # Create an instance of the SystemPropertiesProcessor
    storm_analysis = SystemPropertiesProcessor(
        output_folder_snapshots=output_folder_snapshots,
        output_folder_averages=output_folder_averages,
        output_path=output_path,
        connection_path=connection_path
    )

    # Run the analysis
    storm_analysis.run_analysis()

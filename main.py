from datetime import timedelta

import numpy as np

from src import file_utils, time_utils, simulations, node_information

## Non-straightforward configurations (if modified, check for outliers, etc)
SELECTED_STATION = "Nose Creek Above Airdrie"  # [ "Nose Creek Above Airdrie", "Jumpingpound
# Creek at Township Road 252", "Bow River at Calgary" ]

## User modifiable configurations
NAN_VALUES_REMOVED_OR_ZERO = "REMOVED"  # [ "REMOVED", "ZERO" ]

WATER_LEVEL_THRESHOLD_ON = 8.8
WATER_LEVEL_THRESHOLD_OFF = 8.77

START_VALUE = 150000
END_VALUE = 350000

print("Start...")

monitoring_stations_df = file_utils.read_csv("./csv/River_Level_and_Flow_Monitoring_Stations.csv")
measurements_df = file_utils.read_csv("./csv/River_Levels_and_Flows.csv")

print("Join Stations and Measurements...")
stations = monitoring_stations_df["Station Name"]
data_by_stations = {}
for station in measurements_df["Station Name"].unique():
    df_tmp = measurements_df[measurements_df["Station Name"] == station].copy()
    df_tmp.reset_index(drop=True, inplace=True)
    data_by_stations[station] = df_tmp
    print(f"Station: {station}, length: {len(data_by_stations[station].index)}")

print(f"Selected Station is: '{SELECTED_STATION}'...")
df_node = data_by_stations[SELECTED_STATION].copy()
df_node['Time'] = [time_utils.translate_timestamp(n) for n in df_node["Timestamp"].to_list()]

print("Data Cleaning...")
# --- Check if there are any NAN values ---
nan_level = np.sum(np.isnan(df_node["Level"].to_numpy(), where=1.0))
nan_flow = np.sum(np.isnan(df_node["Flow"].to_numpy(), where=1.0))
print(f"Level nans {nan_level} out of {len(df_node.index)} values")
print(f"Flow nans {nan_flow} out of {len(df_node.index)} values")

if NAN_VALUES_REMOVED_OR_ZERO == "REMOVED":
    # Remove the NAN values from the series. Other options may be considered.
    print("Removing NAN values...")
    df_node_filt = df_node[~np.isnan(df_node['Level'].to_numpy(), where=True)]
    df_node_filt = df_node_filt[~np.isnan(df_node_filt['Flow'].to_numpy(), where=True)]
    df_node = df_node_filt
    nan_level = np.sum(np.isnan(df_node['Level'].to_numpy(), where=1.0))
    nan_flow = np.sum(np.isnan(df_node['Flow'].to_numpy(), where=1.0))
else:
    # Set the NAN values to zero. Other options may be considered.
    print("Replacing NAN values to zeroes...")
    df_node['Level'] = df_node['Level'].fillna(0)
    df_node['Flow'] = df_node['Flow'].fillna(0)
    nan_level = np.sum(np.isnan(df_node['Level'].to_numpy(), where=1.0))
    nan_flow = np.sum(np.isnan(df_node['Flow'].to_numpy(), where=1.0))

print(f"Level nans {nan_level} out of {len(df_node.index)} values")
print(f"Flow nans {nan_flow} out of {len(df_node.index)} values")

print("Visualizations...")

# df_visualization.node_test_display_correlation_flow_level(df_node)
# df_visualization.node_test_display_water_level_with_thresholds(df_node)
# df_visualization.node_test_display_flow_with_thresholds(df_node)

# df_visualization.test_linear_regression(df_node)
# df_visualization.test_random_forest(df_node)

print("Node Information...")

# node_information.test_show_time(df_node, "Time")

# Re-timing data to cut empty-reading times
t0 = df_node.iloc[0]['Time']
_increment = timedelta(minutes=5)
new_time = [t0 + (_increment * n) for n, t in enumerate(df_node['Time'].to_numpy())]
df_node['TimeNew'] = new_time

# node_information.test_show_time(df_node, "TimeNew")

event_node = node_information.flood_events(df_node)

naive_node, naive_times, naive_remaining_charge = simulations.naive_simulation(df_node)

node_only_node, node_only_times, node_only_remaining_charge = \
    simulations.node_only_simulation(df_node)

server_only_node, server_only_times, server_only_remaining_charge = \
    simulations.server_only_simulation(df_node)

complete_node, complete_times, complete_remaining_charge = \
    simulations.node_and_server_simulation(df_node)

node_information.plot_results(
    naive_times, naive_remaining_charge,
    node_only_times, node_only_remaining_charge,
    server_only_times, server_only_remaining_charge,
    complete_times, complete_remaining_charge,
    event_node, naive_node, node_only_node, server_only_node, complete_node
)

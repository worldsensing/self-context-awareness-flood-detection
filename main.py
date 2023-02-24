from src import data_visualization, data_preparation
from src.simulations.naive_simulation import naive_simulation
from src.simulations.node_and_server_simulation import node_and_server_simulation
from src.simulations.node_only_simulation import node_only_simulation
from src.simulations.server_only_simulation import server_only_simulation
from src.utils import file_utils

# Non-straightforward configurations (if modified, check for outliers, etc)
SELECTED_STATION = "Nose Creek Above Airdrie"  # [ "Nose Creek Above Airdrie", "Jumpingpound
# Creek at Township Road 252", "Bow River at Calgary" ]

# User modifiable configurations
NAN_VALUES_REMOVED_OR_ZERO = "REMOVED"  # [ "REMOVED", "ZERO" ]

WATER_LEVEL_THRESHOLD_ON = 8.8
WATER_LEVEL_THRESHOLD_OFF = 8.77

START_VALUE = 150000
END_VALUE = 350000

if __name__ == "__main__":
    print("Start...")

    measurements_df = file_utils.read_csv("./csv/River_Levels_and_Flows.csv")

    print("Data Preparation and Cleaning...")
    df_node = data_preparation.join_stations_and_measurements(measurements_df, SELECTED_STATION)

    df_node = data_preparation.convert_nan_values(df_node, NAN_VALUES_REMOVED_OR_ZERO)

    df_node = data_preparation.prepare_time_and_timenew_column(df_node)

    print("Initial Visualizations...")
    data_visualization.test_display_initial_values(df_node, show=False)
    data_visualization.test_display_correlation_flow_level(df_node, show=False)
    data_visualization.test_display_water_level_with_thresholds(df_node, show=False)
    data_visualization.test_display_flow(df_node, show=False)

    data_visualization.test_linear_regression(df_node, show=False)
    data_visualization.test_random_forest(df_node, show=False)

    data_visualization.test_show_time(df_node, "Time")
    data_visualization.test_show_time(df_node, "TimeNew")

    print("Flood Events Visualizations...")
    event_node = data_visualization.flood_events(df_node, show=False)

    print("Naive Visualizations...")
    naive_node, naive_times, naive_remaining_charge = naive_simulation(df_node, show=False)

    print("Node Only Visualizations...")
    node_only_node, node_only_times, node_only_remaining_charge = \
        node_only_simulation(df_node, show=False)

    print("Server Only Visualizations...")
    server_only_node, server_only_times, server_only_remaining_charge = \
        server_only_simulation(df_node, show=False)

    print("Node and Server Visualizations...")
    complete_node, complete_times, complete_remaining_charge = \
        node_and_server_simulation(df_node, show=False)

    print("Battery Discharge Comparison...")
    data_visualization.charge_usage(
        naive_times, naive_remaining_charge,
        node_only_times, node_only_remaining_charge,
        server_only_times, server_only_remaining_charge,
        complete_times, complete_remaining_charge,
        show=False)

    print("Event Detection Delays Comparison...")
    data_visualization.event_detection_delays(event_node, naive_node, node_only_node,
                                              server_only_node, complete_node)

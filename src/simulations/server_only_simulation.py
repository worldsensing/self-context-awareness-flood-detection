from matplotlib import pyplot as plt

from src.data_model.Node import Node
from src.utils import file_utils
import numpy as np

def server_only_simulation(df_node, show=False):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    _sampling_rate_mins = 5 * 12
    _prev_sampling_rate = 5 * 12

    def _server_only_get_sampling_rate(nd: Node, pred, level):
        nonlocal _prev_sampling_rate
        trans = True
        recep = False
        in_event = nd.in_event
        if in_event:  # a Flood event is active
            sampling_rate = 5 * 7
        elif level < nd.threshold_off:
            sampling_rate = 5 * 12
        else:
            sampling_rate = 5 * 7

        if sampling_rate != _prev_sampling_rate:
            recep = True
            _prev_sampling_rate = sampling_rate

        return sampling_rate, trans, recep

    server_only_node = Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)

    server_only_simulation_running = True
    df_server_only, x0, p0, t0, l0 = server_only_node.init_sampling(df_node)
    server_only_samples = [x0]
    server_only_times = [0]
    server_only_predictions = [p0]
    server_only_levels = [l0]
    server_only_node.end_sample_event(True, False, 0, l0)
    server_only_remaining_charge = [server_only_node.battery_charge_initial,
                                    server_only_node.battery_charge_remaining]
    sampling_rate = _sampling_rate_mins
    transmission = True
    reception = False
    while server_only_simulation_running:
        x, p, l, dt, server_only_simulation_running = \
            server_only_node.get_next_sample(sampling_rate, df_server_only)
        if server_only_simulation_running:
            server_only_node.end_sample_event(transmission, reception, sampling_rate, l)
            server_only_samples.append(x)
            server_only_predictions.append(p)
            server_only_levels.append(l)
            server_only_times.append(dt)
            server_only_remaining_charge.append(server_only_node.battery_charge_remaining)
            sampling_rate, transmission, reception = _server_only_get_sampling_rate(
                server_only_node, p, l)


    server_only_time_days = np.array(server_only_times) / (60 * 24)
    plt.figure()
    plt.title("Context-Aware Samples")
    plt.plot(server_only_time_days, server_only_samples, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "context__samples")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Context-Aware Predictions")
    plt.plot(server_only_time_days, server_only_predictions, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Predicted Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "context__predictions")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Context-Aware Levels")
    plt.plot(server_only_time_days, server_only_levels, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "context__levels")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Context-Aware Remaining Battery Charge")
    plt.xlabel("Monitoring Time (days)")
    plt.ylabel("Remaining Battery Charge (mAh)")
    plt.plot(server_only_time_days, server_only_remaining_charge[1:])
    file_utils.save_plot(plt, "context__remaining_charge")
    file_utils.show_plot(plt, show=show)

    return server_only_node, server_only_times, server_only_remaining_charge

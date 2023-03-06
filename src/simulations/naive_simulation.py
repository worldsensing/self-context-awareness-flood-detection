from matplotlib import pyplot as plt

from src.data_model.Node import Node
from src.utils import file_utils
import numpy as np


def naive_simulation(df_node, show=False):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF
    _naive_sampling_rate_mins = 5 * 9  # one sample per hour

    naive_node = Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)
    naive_simulation_running = True
    df_naive, x0, l0, t0, l0 = naive_node.init_sampling(df_node)
    naive_samples = [x0]
    naive_predictions = [0]
    naive_times = [0]
    naive_levels = [l0]
    naive_node.end_sample_event(True, False, 0, l0)
    naive_remaining_charge = [naive_node.battery_charge_initial,
                              naive_node.battery_charge_remaining]
    while naive_simulation_running:
        x, _, l, dt, naive_simulation_running = naive_node.get_next_sample(
            _naive_sampling_rate_mins, df_naive)
        if naive_simulation_running:
            naive_node.end_sample_event(True, False, _naive_sampling_rate_mins, l)
            naive_samples.append(x)
            naive_predictions.append(0)
            naive_levels.append(l)
            naive_remaining_charge.append(naive_node.battery_charge_remaining)
            naive_times.append(dt)

    naive_time_days = np.array(naive_times) / (60 * 24)
    
    plt.figure()
    plt.title("Naive Samples")
    plt.plot(naive_time_days, naive_samples, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "naive_simulation__samples")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Naive Predictions")
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Predicted Water Level (m)", fontsize=16)
    plt.plot(naive_time_days, naive_predictions, '.')
    file_utils.save_plot(plt, "naive_simulation__predictions")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Naive Remaining Battery Charge")
    plt.xlabel("Monitoring Time (days)")
    plt.ylabel("Remaining Battery Charge (mAh)")
    plt.plot(naive_time_days, naive_remaining_charge[1:])
    file_utils.save_plot(plt, "naive_simulation__remaining_charge")
    file_utils.show_plot(plt, show=show)

    return naive_node, naive_times, naive_remaining_charge

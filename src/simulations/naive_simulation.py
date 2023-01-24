from matplotlib import pyplot as plt

from src.data_model.Node import Node
from src.utils import file_utils


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

    # TODO Fix what is shown in X axis?
    # TODO Fix what is shown in Y axis?
    plt.figure()
    plt.title("Naive Samples")
    plt.plot(naive_samples, '.')
    file_utils.save_plot(plt, "naive_simulation__samples")
    file_utils.show_plot(plt, show=show)

    # TODO Fix what is shown in X axis?
    # TODO Fix what is shown in Y axis?
    plt.figure()
    plt.title("Naive Predictions")
    plt.plot(naive_predictions, '.')
    file_utils.save_plot(plt, "naive_simulation__predictions")
    file_utils.show_plot(plt, show=show)

    # TODO We do not show here Levels?

    # TODO Fix what is shown in X axis?
    plt.figure()
    plt.title("Naive Remaining Charge")
    plt.ylabel("Remaining Battery Charge (mAh)")
    plt.plot(naive_remaining_charge)
    file_utils.save_plot(plt, "naive_simulation__remaining_charge")
    file_utils.show_plot(plt, show=show)

    # TODO Fix X labels for Sample Number instead of 0 to 1
    plt.figure(figsize=(10, 8))

    plt.plot(naive_times, naive_levels, 'k')
    plt.xlabel("Sample Number", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    plt.axhline(y=naive_node.threshold_on, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=naive_node.threshold_off, color='g', linestyle='-', label="Threshold - OFF")

    add_item_flood_starts_in_legend = True
    add_item_flood_ends_in_legend = True
    for i in range(len(naive_node.events)):
        if 'started' == naive_node.event_meaning[i]:
            plt.axvline(x=naive_node.events[i], color='b', linestyle='-',
                        label="Flood starts" if add_item_flood_starts_in_legend else "")
            add_item_flood_starts_in_legend = False
        else:
            plt.axvline(x=naive_node.events[i], color='b', linestyle='--',
                        label="Flood ends" if add_item_flood_ends_in_legend else "")
            add_item_flood_ends_in_legend = False
    plt.legend()
    plt.title("Naive Simulation", fontsize=18)

    file_utils.save_plot(plt, "naive_simulation")
    file_utils.show_plot(plt, show=show)

    return naive_node, naive_times, naive_remaining_charge

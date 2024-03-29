from matplotlib import pyplot as plt

from src.data_model.Node import Node
from src.utils import file_utils
import numpy as np


def node_and_server_simulation(df_node, show=False):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    _sampling_rate_mins = 5 * 12
    _transmission_cnt = 0
    _server_managed = False
    _server_false_flood = False
    _server_true_flood = False
    _server_sampling_rate = 5 * 12
    _prev_event = False

    def _node_and_server_get_sampling_rate(nd: Node, p, level):
        nonlocal _transmission_cnt, _server_managed, _server_sampling_rate, \
            _server_false_flood, _server_true_flood, _prev_event

        trans = False
        recep = False
        in_event = nd.in_event
        if not _server_managed:
            if in_event:  # a Flood event is active
                sampling_rate = 5 * 2
                transmission_thres = 5 * 3
            elif p < nd.threshold_off:
                sampling_rate = 5 * 9
                transmission_thres = 5 * 12
            else:
                sampling_rate = 5 * 2
                transmission_thres = 5 * 3

            if (not in_event and p >= nd.threshold_on) or (in_event and p < nd.threshold_off):
                trans = True
                _transmission_cnt = 0
            elif _prev_event and p < nd.threshold_off:
                trans = True
                _transmission_cnt = 0
            elif not _prev_event and p > nd.threshold_on:
                trans = True
                _transmission_cnt = 0
            elif _transmission_cnt >= transmission_thres:
                trans = True
                _transmission_cnt = 0
            else:
                _transmission_cnt += sampling_rate

            if trans:
                # Flood taking place but not predicted
                if level >= nd.threshold_on and p < nd.threshold_off:
                    recep = True
                    sampling_rate = 5 * 2
                    _server_sampling_rate = sampling_rate
                    _server_true_flood = True
                    _server_manged = True
                # Flood not taking place, but it is predicted by the node
                elif p >= nd.threshold_on and level < nd.threshold_off:
                    recep = True
                    sampling_rate = 5 * 9
                    _server_sampling_rate = sampling_rate
                    _server_false_flood = True
                    _server_managed = True

        else:  # The sampling rate is managed by the server
            trans = True
            sampling_rate = _server_sampling_rate
            # There is a flood currently taking place and the node didn't detect it
            if _server_true_flood:
                if level < nd.threshold_off:
                    recep = True
                    sampling_rate = 5 * 3
                    _server_true_flood = False
                    _server_managed = False
            # No flood is taking place, but the node thinks there is
            else:
                # the Flood actually takes place
                if level >= nd.threshold_on:
                    # the node is able to detect the flood.
                    if p >= nd.threshold_on:
                        sampling_rate = 5 * 2
                        recep = True
                        _server_managed = False
                        _server_false_flood = False
                    # the node is not able to detect the flood
                    else:
                        recep = True
                        sampling_rate = 5 * 2
                        _server_false_flood = False
                        _server_true_flood = True
                        _server_sampling_rate = 5 * 2
                # No flood is taking place
                if level < nd.threshold_off:
                    # Node is able to see there is no flood
                    if p < nd.threshold_off:
                        recep = True
                        sampling_rate = 5 * 9
                        _server_false_flood = False
                        _server_managed = False

        if not in_event:
            _prev_event = False
        else:
            _prev_event = True

        return sampling_rate, trans, recep

    complete_node = Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)

    complete_simulation_running = True
    df_complete, x0, p0, t0, l0 = complete_node.init_sampling(df_node)
    complete_samples = [x0]
    complete_times = [0]
    complete_predictions = [p0]
    complete_levels = [l0]
    complete_node.end_sample_event(True, False, 0, l0)
    complete_remaining_charge = [complete_node.battery_charge_initial,
                                 complete_node.battery_charge_remaining]
    sampling_rate = _sampling_rate_mins
    transmission = True
    reception = False
    while complete_simulation_running:
        x, p, l, dt, complete_simulation_running = \
            complete_node.get_next_sample(sampling_rate, df_complete)
        if complete_simulation_running:
            complete_node.end_sample_event(transmission, reception, sampling_rate, l)
            complete_samples.append(x)
            complete_predictions.append(p)
            complete_levels.append(l)
            complete_times.append(dt)
            complete_remaining_charge.append(complete_node.battery_charge_remaining)
            sampling_rate, transmission, reception = _node_and_server_get_sampling_rate(
                complete_node, p, l)

    complete_time_days = np.array(complete_times) / (60 * 24)
    plt.figure()
    plt.title("Context-and-Self-Aware Samples")
    plt.plot(complete_time_days, complete_samples, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "context_and_self__samples")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Context-and-Self-Aware Predictions")
    plt.plot(complete_time_days, complete_predictions, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Predicted Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "context_and_self__predictions")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Context-and-Self-Aware Levels")
    plt.plot(complete_time_days, complete_levels, '.')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    file_utils.save_plot(plt, "context_and_self__levels")
    file_utils.show_plot(plt, show=show)

    plt.figure()
    plt.title("Context-and-Self-Aware Remaining Battery Charge")
    plt.xlabel("Monitoring Time (days)")
    plt.ylabel("Remaining Battery Charge (mAh)")
    plt.plot(complete_time_days, complete_remaining_charge[1:])
    file_utils.save_plot(plt, "context_and_self__remaining_charge")
    file_utils.show_plot(plt, show=show)

    return complete_node, complete_times, complete_remaining_charge

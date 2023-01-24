from matplotlib import pyplot as plt

from src.data_model.Node import Node
from src.utils import file_utils


def node_only_simulation(df_node, show=False):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    _sampling_rate_mins = 5 * 12
    _transmission_cnt = 0
    _prev_event = False

    def get_sampling_rate(nd: Node, p, l):
        nonlocal _transmission_cnt, _prev_event
        trans = False
        recep = False
        in_event = nd.in_event
        if in_event:  # a Flood event is active
            sampling_rate = 5 * 2
            transmission_thres = 5 * 3
            _prev_event = True
        elif p < nd.threshold_off:  # high level
            sampling_rate = 5 * 12  # 15min
            transmission_thres = 5 * 18  # 30min
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

        if not in_event:
            _prev_event = False
        else:
            _prev_event = True
        return sampling_rate, trans, recep

    node_only_node = Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)

    node_only_simulation_running = True
    df_node_only, x0, p0, t0, l0 = node_only_node.init_sampling(df_node)
    node_only_samples = [x0]
    node_only_times = [0]
    node_only_predictions = [p0]
    node_only_levels = [l0]
    node_only_node.end_sample_event(True, False, 0, l0)
    node_only_remaining_charge = [node_only_node.battery_charge_initial,
                                  node_only_node.battery_charge_remaining]
    sampling_rate = _sampling_rate_mins
    transmission = True
    reception = False
    while node_only_simulation_running:
        x, p, l, dt, node_only_simulation_running = \
            node_only_node.get_next_sample(sampling_rate, df_node_only)
        if node_only_simulation_running:
            node_only_node.end_sample_event(transmission, reception, sampling_rate, l)
            node_only_samples.append(x)
            node_only_predictions.append(p)
            node_only_levels.append(l)
            node_only_times.append(dt)
            node_only_remaining_charge.append(node_only_node.battery_charge_remaining)
            sampling_rate, transmission, reception = get_sampling_rate(node_only_node, p, l)

    # TODO Fix what is shown in X axis?
    # TODO Fix what is shown in Y axis?
    plt.figure()
    plt.title("Node-Only Samples")
    plt.plot(node_only_samples, '.')
    file_utils.save_plot(plt, "node_only__samples")
    file_utils.show_plot(plt, show=show)

    # TODO Fix what is shown in X axis?
    # TODO Fix what is shown in Y axis?
    plt.figure()
    plt.title("Node-Only Predictions")
    plt.plot(node_only_predictions, '.')
    file_utils.save_plot(plt, "node_only__predictions")
    file_utils.show_plot(plt, show=show)

    # TODO Fix what is shown in X axis?
    # TODO Fix what is shown in Y axis?
    plt.figure()
    plt.title("Node-Only Levels")
    plt.plot(node_only_levels, '.')
    file_utils.save_plot(plt, "node_only__levels")
    file_utils.show_plot(plt, show=show)

    # TODO Fix what is shown in Y axis?
    plt.figure()
    plt.title("Node-Only Remaining Charge")
    plt.ylabel("Remaining Battery Charge (mAh)")
    plt.plot(node_only_remaining_charge)
    file_utils.save_plot(plt, "node_only__remaining_charge")
    file_utils.show_plot(plt, show=show)

    # TODO We do not show here full simulation for node_only?

    return node_only_node, node_only_times, node_only_remaining_charge

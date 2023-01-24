from matplotlib import pyplot as plt

from src.data_model.Node import Node


def naive_simulation(df_node):
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

    plt.figure()
    plt.plot(naive_samples, '.')
    plt.show()

    plt.figure()
    plt.plot(naive_predictions, '.')
    plt.show()

    plt.figure()
    plt.plot(naive_remaining_charge)
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.plot(naive_times, naive_levels, 'k')
    plt.axhline(y=naive_node.threshold_on, color='r', linestyle='-')
    plt.axhline(y=naive_node.threshold_off, color='r', linestyle='--')
    for i in range(len(naive_node.events)):
        if 'started' == naive_node.event_meaning[i]:
            plt.axvline(x=naive_node.events[i], color='g', linestyle='-')
        else:
            plt.axvline(x=naive_node.events[i], color='b', linestyle='-')
    plt.show()

    return naive_node, naive_times, naive_remaining_charge


def node_only_simulation(df_node):
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

    plt.figure()
    plt.plot(node_only_samples, '.')
    plt.show()

    plt.figure()
    plt.plot(node_only_predictions, '.')
    plt.show()

    plt.figure()
    plt.plot(node_only_levels, '.')
    plt.show()

    plt.figure()
    plt.plot(node_only_remaining_charge)
    plt.show()

    return node_only_node, node_only_times, node_only_remaining_charge


def server_only_simulation(df_node):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    _sampling_rate_mins = 5 * 12
    _prev_sampling_rate = 5 * 12

    def _server_only_get_sampling_rate(nd: Node, pred, level):
        nonlocal _prev_sampling_rate
        trans = True
        recep = False
        in_event = nd.in_event
        if in_event:  # a Flood event is active
            sampling_rate = 5 * 3
        elif level < nd.threshold_on:  # high level
            sampling_rate = 5 * 18  # 15min
        else:
            sampling_rate = 5 * 3

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

    plt.figure()
    plt.plot(server_only_samples, '.')
    plt.show()

    plt.figure()
    plt.plot(server_only_predictions, '.')
    plt.show()

    plt.figure()
    plt.plot(server_only_levels, '.')
    plt.show()

    plt.figure()
    plt.plot(server_only_remaining_charge)
    plt.show()

    return server_only_node, server_only_times, server_only_remaining_charge


def node_and_server_simulation(df_node):
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
            elif p < nd.threshold_off:  # high level
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

    plt.figure()
    plt.plot(complete_samples, '.')
    plt.show()

    plt.figure()
    plt.plot(complete_predictions, '.')
    plt.show()

    plt.figure()
    plt.plot(complete_levels, '.')
    plt.show()

    plt.figure()
    plt.plot(complete_remaining_charge)
    plt.show()

    return complete_node, complete_times, complete_remaining_charge

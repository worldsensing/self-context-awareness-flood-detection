from matplotlib import pyplot as plt

from src.data_model.Node import Node


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

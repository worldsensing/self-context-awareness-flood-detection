import numpy as np
from matplotlib import pyplot as plt

from src.data_model.Node import Node


def flood_events(df_node, show=False):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    event_node = Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)
    df_events, x0, y0, _, _ = event_node.init_sampling(df_node)
    event_simulation_running = np.True_
    event_levels = [y0]
    event_times = [0]
    while event_simulation_running:
        x, _, l, dt, event_simulation_running = event_node.get_next_sample(5, df_events)
        if event_simulation_running:
            event_levels.append(l)
            event_times.append(dt)
            event_node.end_sample_event(True, False, 5, l)

    plt.figure(figsize=(16, 10))
    plt.plot(event_times, event_levels, 'k')
    plt.axhline(y=event_node.threshold_on, color='r', linestyle='-')
    plt.axhline(y=event_node.threshold_off, color='r', linestyle='--')
    for i in range(len(event_node.events)):
        if 'started' == event_node.event_meaning[i]:
            plt.axvline(x=event_node.events[i], color='g', linestyle='-')
        else:
            plt.axvline(x=event_node.events[i], color='b', linestyle='-')
    plt.show()

    event_list = []
    _event_size_filter = 1
    in_event = df_events['Level'].iloc[0] >= WATER_LEVEL_THRESHOLD_ON
    for n in range(len(df_events) - _event_size_filter):
        if in_event:
            if np.all(df_events['Level'].to_numpy()[n:n + _event_size_filter]
                      < WATER_LEVEL_THRESHOLD_OFF):
                event_list.append(n)
                in_event = False
        else:
            if np.all(df_events['Level'].to_numpy()[n:n + 1] >= WATER_LEVEL_THRESHOLD_ON):
                event_list.append(n)
                in_event = True
    print(event_list)

    plt.figure(figsize=(16, 10))
    plt.plot(df_events['Level'].to_numpy())
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-')
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='r', linestyle='--')
    for e in event_list:
        plt.axvline(x=e, color='g', linestyle='-')

    return event_node


def plot_results(naive_times, naive_remaining_charge, node_only_times, node_only_remaining_charge,
                 server_only_times, server_only_remaining_charge, complete_times,
                 complete_remaining_charge, event_node, naive_node, node_only_node,
                 server_only_node, complete_node, show=False):
    test_names = ['Naive', 'Self-awareness', 'Context-awareness', 'Self-Context-awareness']

    plt.figure(figsize=(8, 5))

    colormap = plt.cm.gist_rainbow  # nipy_spectral, Set1,Paired
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(test_names)))))

    plt.plot(naive_times, naive_remaining_charge[1:], linewidth=2.5, label=test_names[0])
    plt.plot(node_only_times, node_only_remaining_charge[1:], linewidth=2.5, label=test_names[1])
    plt.plot(server_only_times, server_only_remaining_charge[1:], linewidth=2.5,
             label=test_names[2])
    plt.plot(complete_times, complete_remaining_charge[1:], linewidth=2.5, label=test_names[3])

    plt.title("Battery Discharge", fontsize=16)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Remaining battery charge mAh", fontsize=12)
    # plt.legend(loc="right", fancybox=True, bbox_to_anchor=(1.2, 0.5),
    #           shadow=True, prop={'size':12})
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=4, prop={'size': 12})
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    # plt.savefig("remaining_battery_charge.png")
    plt.show()

    # --- Event detection delay ---
    print(event_node.events)
    print(event_node.event_meaning)

    # Remove short events
    ref_events = []
    ref_events_meaning = []
    prev_remove = False
    for i in range(len(event_node.events) - 1):
        if not prev_remove:
            if event_node.events[i + 1] - event_node.events[i] > 10:
                ref_events.append(event_node.events[i])
                ref_events_meaning.append(event_node.event_meaning[i])
            else:
                prev_remove = True
        else:
            prev_remove = False
    if not prev_remove:
        ref_events.append(event_node.events[-1])
        ref_events_meaning.append(event_node.event_meaning[-1])
    print(ref_events)
    print(ref_events_meaning)

    print("////////////////")
    print(naive_node.events)
    print(naive_node.event_meaning)

    print("////////////////")
    print(node_only_node.events)
    print(node_only_node.event_meaning)

    print("////////////////")
    print(server_only_node.events)
    print(server_only_node.event_meaning)

    print("////////////////")
    print(complete_node.events)
    print(complete_node.event_meaning)

    # Detection delay
    def compute_delays(ref, test):
        res = [test[i] - ref[i] for i in range(len(ref))]
        return res

    # node_only_node.events.remove(339035)
    # node_only_node.events.remove(339045)

    delay_naive = compute_delays(ref_events, naive_node.events)
    delay_node_only = compute_delays(ref_events, node_only_node.events)
    delay_server_only = compute_delays(ref_events, server_only_node.events)
    delay_complete = compute_delays(ref_events, complete_node.events)

    print("Delays:")
    print(f"   Naive:    {delay_naive}")
    print(f"   Node:     {delay_node_only}")
    print(f"   Server:   {delay_server_only}")
    print(f"   Complete: {delay_complete}")

    def average_delay(test):
        return np.mean(test)

    avg_delay_naive = average_delay(delay_naive)
    avg_delay_node_only = average_delay(delay_node_only)
    avg_delay_server_only = average_delay(delay_server_only)
    avg_delay_complete = average_delay(delay_complete)
    print("Delays:")
    print(f"   Naive:    {avg_delay_naive}")
    print(f"   Node:     {avg_delay_node_only}")
    print(f"   Server:   {avg_delay_server_only}")
    print(f"   Complete: {avg_delay_complete}")

    print("Delay improvements:")
    print(f"   Naive:    {100 * (1 - avg_delay_naive / avg_delay_naive)}")
    print(f"   Node:     {100 * (1 - avg_delay_node_only / avg_delay_naive)}")
    print(f"   Server:   {100 * (1 - avg_delay_server_only / avg_delay_naive)}")
    print(f"   Complete: {100 * (1 - avg_delay_complete / avg_delay_naive)}")

import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.data_model.Node import Node
from src.utils import file_utils


def test_display_correlation_flow_level(df_node, show=False):
    plt.figure(1)
    ax = sns.violinplot(x=df_node['Level'])
    ax.set_title("Level")
    plt.plot()
    file_utils.save_plot(plt, "test_display_correlation_flow_level__level")
    file_utils.show_plot(plt, show=show)

    plt.figure(2)
    ax2 = sns.violinplot(x=df_node['Flow'])
    ax2.set_title("Flow")
    plt.plot()
    file_utils.save_plot(plt, "test_display_correlation_flow_level__flow")
    file_utils.show_plot(plt, show=show)

    plt.figure(3, figsize=(8, 6))
    node_corr = df_node[['Level', 'Flow']].corr()
    ax3 = sns.heatmap(node_corr, annot=True, cmap='YlGnBu', fmt=".2f")
    ax3.set_title("Level / Flow Correlation")
    plt.plot()
    file_utils.save_plot(plt, "test_display_correlation_flow_level__correlation")
    file_utils.show_plot(plt, show=show)


def test_display_water_level_with_thresholds(df_node, show=False):
    from main import START_VALUE, END_VALUE, WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    plt.figure(figsize=(10, 8))
    plt.plot(df_node["Level"].to_numpy()[START_VALUE:END_VALUE], label="Water Level (m)")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='g', linestyle='-', label="Threshold - OFF")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Water Level (m)", fontsize=18)
    plt.legend()

    file_utils.save_plot(plt, "test_display_water_level_with_thresholds")
    file_utils.show_plot(plt, show=show)


def test_display_flow(df_node, show=False):
    from main import START_VALUE, END_VALUE

    plt.figure(figsize=(10, 8))
    plt.plot(df_node["Flow"].to_numpy()[START_VALUE:END_VALUE], label="Flow (cms)")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Flow (cms)", fontsize=18)
    plt.legend()

    file_utils.save_plot(plt, "test_display_flow_with_thresholds")
    file_utils.show_plot(plt, show=show)


def test_linear_regression(df_node, show=False):
    from sklearn.linear_model import LinearRegression
    from main import START_VALUE, END_VALUE, WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    node_model = LinearRegression()
    x = df_node['Flow'].to_numpy()[END_VALUE:].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[END_VALUE:]
    node_model.fit(x, y)

    x = df_node['Flow'].to_numpy()[START_VALUE:END_VALUE].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[START_VALUE:END_VALUE].reshape(-1, 1)
    preds = node_model.predict(x)

    plt.figure(figsize=(10, 8))
    plt.plot(y, label="Water Level (m)")
    plt.plot(preds, label="Predicted Water Level (m)")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Water Level (m)", fontsize=18)
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='g', linestyle='-', label="Threshold - OFF")
    plt.legend()
    plt.title("Linear Regression", fontsize=20)

    file_utils.save_plot(plt, "test_linear_regression")
    file_utils.show_plot(plt, show=show)


def test_random_forest(df_node, show=False):
    from sklearn.ensemble import RandomForestRegressor
    from main import START_VALUE, WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    node_model = RandomForestRegressor(n_estimators=100)
    x = df_node['Flow'].to_numpy()[:START_VALUE].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[:START_VALUE]
    node_model.fit(x, y)

    x = df_node['Flow'].to_numpy()[START_VALUE:].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[START_VALUE:].reshape(-1, 1)
    preds = node_model.predict(x)

    plt.figure(figsize=(10, 8))
    plt.plot(y, label="Water Level (m)")
    plt.plot(preds, label="Predicted Water Level (m)")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Water Level (m)", fontsize=18)
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='g', linestyle='-', label="Threshold - OFF")
    plt.legend()
    plt.title("Random Forest", fontsize=20)

    file_utils.save_plot(plt, "test_random_forest")
    file_utils.show_plot(plt, show=show)


def test_show_time(df_node, time_column, show=False):
    dates = df_node[time_column].to_numpy()
    y = df_node['Level'].to_numpy()

    plt.figure(figsize=(10, 8))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.plot(dates, y, "*")
    plt.gcf().autofmt_xdate()

    file_utils.show_plot(plt, show=show)


def flood_events(df_node, show=False):
    from main import WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    event_node = Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)
    df_events, x0, y0, _, _ = event_node.init_sampling(df_node)
    event_simulation_running = True
    event_levels = [y0]
    event_times = [0]
    while event_simulation_running:
        x, _, l, dt, event_simulation_running = event_node.get_next_sample(5, df_events)
        if event_simulation_running:
            event_levels.append(l)
            event_times.append(dt)
            event_node.end_sample_event(True, False, 5, l)

    # TODO Fix X labels for Sample Number instead of 0 to 1
    plt.figure(figsize=(10, 8))
    plt.plot(event_times, event_levels, 'k')
    plt.xlabel("Sample Number", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    plt.axhline(y=event_node.threshold_on, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=event_node.threshold_off, color='g', linestyle='-', label="Threshold - OFF")

    add_item_flood_starts_in_legend = True
    add_item_flood_ends_in_legend = True
    for i in range(len(event_node.events)):
        if 'started' == event_node.event_meaning[i]:
            plt.axvline(x=event_node.events[i], color='b', linestyle='-',
                        label="Flood starts" if add_item_flood_starts_in_legend else "")
            add_item_flood_starts_in_legend = False
        else:
            plt.axvline(x=event_node.events[i], color='b', linestyle='--',
                        label="Flood ends" if add_item_flood_ends_in_legend else "")
            add_item_flood_ends_in_legend = False
    plt.legend()
    plt.title("Flood Events", fontsize=18)

    file_utils.save_plot(plt, "flood_events")
    file_utils.show_plot(plt, show=show)

    return event_node


def battery_discharge(naive_times, naive_remaining_charge, node_only_times,
                      node_only_remaining_charge, server_only_times,
                      server_only_remaining_charge, complete_times,
                      complete_remaining_charge, show=False):
    test_names = ['Naive', 'Self-awareness', 'Context-awareness', 'Self-Context-awareness']

    plt.figure(figsize=(10, 8))

    colormap = plt.cm.gist_rainbow  # nipy_spectral, Set1,Paired
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(test_names)))))

    plt.plot(naive_times, naive_remaining_charge[1:], linewidth=2.5, label=test_names[0])
    plt.plot(node_only_times, node_only_remaining_charge[1:], linewidth=2.5, label=test_names[1])
    plt.plot(server_only_times, server_only_remaining_charge[1:], linewidth=2.5,
             label=test_names[2])
    plt.plot(complete_times, complete_remaining_charge[1:], linewidth=2.5, label=test_names[3])

    plt.title("Battery Discharge", fontsize=16)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Remaining battery charge (mAh)", fontsize=12)
    # plt.legend(loc="right", fancybox=True, bbox_to_anchor=(1.2, 0.5),
    #           shadow=True, prop={'size':12})
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=4, prop={'size': 12})
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    file_utils.save_plot(plt, "simulation_results")
    file_utils.show_plot(plt, show=show)


# TODO Should we write this to a file?
def event_detection_delays(event_node, naive_node, node_only_node, server_only_node, complete_node):
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

    # Detection delay
    def compute_delays(ref, test):
        res = [test[i] - ref[i] for i in range(len(ref))]
        return res

    delay_naive = compute_delays(ref_events, naive_node.events)
    delay_node_only = compute_delays(ref_events, node_only_node.events)
    delay_server_only = compute_delays(ref_events, server_only_node.events)
    delay_complete = compute_delays(ref_events, complete_node.events)

    # TODO Add units or similar, here could be time per ¿?
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

    # TODO Add units or similar, here could be seconds?
    print("Delays:")
    print(f"   Naive:    {avg_delay_naive}")
    print(f"   Node:     {avg_delay_node_only}")
    print(f"   Server:   {avg_delay_server_only}")
    print(f"   Complete: {avg_delay_complete}")

    # TODO Add units or similar, here could be %?
    print("Delay improvements:")
    print(f"   Naive:    {100 * (1 - avg_delay_naive / avg_delay_naive)}")
    print(f"   Node:     {100 * (1 - avg_delay_node_only / avg_delay_naive)}")
    print(f"   Server:   {100 * (1 - avg_delay_server_only / avg_delay_naive)}")
    print(f"   Complete: {100 * (1 - avg_delay_complete / avg_delay_naive)}")

import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, gridspec

from src.data_model import Node
from src.utils import file_utils


def test_display_initial_values(df_node, show=False):
    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure()

    ax = fig.add_subplot(gs[0])
    ax.plot(df_node['TimeNew'], df_node['Level'])
    ax.set_ylabel(r'Level (m)', size=15)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        labelbottom='off')  # labels along the bottom edge are off

    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(df_node['TimeNew'], df_node['Flow'])
    ax.set_ylabel(r'Flow (cms)', size=15)

    plt.xlabel("Timestamp")
    plt.plot()
    file_utils.save_plot(plt, "test_display_initial_values")
    file_utils.show_plot(plt, show=show)


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

    time_y = np.arange(len(y)) * 5 / (60 * 24)  # Sampling rate is 5min, time is in days.
    time_preds = np.arange(len(x)) * 5 / (60 * 24)

    plt.figure(figsize=(12, 5))
    plt.plot(time_y, y, label="Water Level (m)")
    plt.plot(time_preds, preds, label="Predicted Water Level (m)")
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='g', linestyle='-', label="Threshold - OFF")
    plt.legend(loc="upper left", prop={'size': 14}, facecolor='white', framealpha=0.9)
    plt.title("Linear Regression", fontsize=18)
    plt.tick_params(axis='both', labelsize=14)

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

    event_node = Node.Node(WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF)
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

    event_time_days = np.array(event_times) / (60 * 24)
    plt.figure(figsize=(12, 5))
    plt.plot(event_time_days, event_levels, 'k')
    plt.xlabel("Monitoring Time (days)", fontsize=16)
    plt.ylabel("Water Level (m)", fontsize=16)
    plt.axhline(y=event_node.threshold_on, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=event_node.threshold_off, color='g', linestyle='-', label="Threshold - OFF")
    plt.tick_params(axis='both', labelsize=14)

    add_item_flood_starts_in_legend = True
    add_item_flood_ends_in_legend = True
    for i in range(len(event_node.events)):
        if 'started' == event_node.event_meaning[i]:
            plt.axvline(x=event_node.events[i] / (60 * 24), color='b', linestyle='-',
                        label="Flood starts" if add_item_flood_starts_in_legend else "")
            add_item_flood_starts_in_legend = False
        else:
            plt.axvline(x=event_node.events[i] / (60 * 24), color='b', linestyle='--',
                        label="Flood ends" if add_item_flood_ends_in_legend else "")
            add_item_flood_ends_in_legend = False
    plt.legend(loc="upper left", prop={'size': 14}, facecolor='white', framealpha=0.9)
    plt.title("Flood Events", fontsize=18)

    file_utils.save_plot(plt, "flood_events")
    file_utils.show_plot(plt, show=show)

    return event_node


def charge_usage(naive_times, naive_remaining_charge, node_only_times,
                 node_only_remaining_charge, server_only_times,
                 server_only_remaining_charge, complete_times,
                 complete_remaining_charge, show=False):
    test_names = ['Naive', 'Self-awareness', 'Context-awareness', 'Self-Context-awareness']

    battery_charge = Node.TOTAL_BATTERY_CHARGE

    # Translate remaining charge to used charge
    naive_used_charge = battery_charge - np.array(naive_remaining_charge)
    node_only_used_charge = battery_charge - np.array(node_only_remaining_charge)
    server_only_used_charge = battery_charge - np.array(server_only_remaining_charge)
    complete_used_charge = battery_charge - np.array(complete_remaining_charge)

    plt.figure(figsize=(12, 5))

    colormap = plt.cm.gist_rainbow  # nipy_spectral, Set1,Paired
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(test_names)))))

    naive_times_days = naive_times / (60 * 24)
    node_only_times_days = node_only_times / (60 * 24)
    server_only_times_days = server_only_times / (60 * 24)
    complete_times_days = complete_times / (60 * 24)

    plt.plot(naive_times_days, naive_used_charge[1:], linewidth=2.5, label=test_names[0])
    plt.plot(node_only_times_days, node_only_used_charge[1:], linewidth=2.5, label=test_names[1])
    plt.plot(server_only_times_days, server_only_used_charge[1:], linewidth=2.5,
             label=test_names[2])
    plt.plot(complete_times_days, complete_used_charge[1:], linewidth=2.5, label=test_names[3])

    plt.title("Used Battery Charge (mAh)", fontsize=18)
    plt.xlabel("Time (days)", fontsize=16)
    plt.ylabel("Battery Charge Usage (mAh)", fontsize=16)
    # plt.legend(loc="right", fancybox=True, bbox_to_anchor=(1.2, 0.5),
    #           shadow=True, prop={'size':12})
    plt.legend(loc="upper left", prop={'size': 14}, facecolor='white', framealpha=0.9)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    file_utils.save_plot(plt, "battery_charge_used")
    file_utils.show_plot(plt, show=show)


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
    print("Flood detection delays in minutes:")
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

    print("Average flood detection delays:")
    print(f"   Naive:    {round(avg_delay_naive, 3)} mins.")
    print(f"   Node:     {round(avg_delay_node_only, 3)} mins.")
    print(f"   Server:   {round(avg_delay_server_only, 3)} mins.")
    print(f"   Complete: {round(avg_delay_complete, 3)} mins.")

    print("Delay improvements:")
    print(f"   Naive:    {round(100 * (1 - avg_delay_naive / avg_delay_naive), 1)}%")
    print(f"   Node:     {round(100 * (1 - avg_delay_node_only / avg_delay_naive), 1)}%")
    print(f"   Server:   {round(100 * (1 - avg_delay_server_only / avg_delay_naive), 1)}%")
    print(f"   Complete: {round(100 * (1 - avg_delay_complete / avg_delay_naive), 1)}%")

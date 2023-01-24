import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import file_utils


def node_test_display_correlation_flow_level(df_node, show=False):
    plt.figure(1)
    ax = sns.violinplot(x=df_node['Level'])
    ax.set_title("Level")
    plt.plot()
    file_utils.save_plot(plt, "node_test_display_correlation_flow_level__level")
    file_utils.show_plot(plt, show=show)

    plt.figure(2)
    ax2 = sns.violinplot(x=df_node['Flow'])
    ax2.set_title("Flow")
    plt.plot()
    file_utils.save_plot(plt, "node_test_display_correlation_flow_level__flow")
    file_utils.show_plot(plt, show=show)

    plt.figure(3, figsize=(8, 6))
    node_corr = df_node[['Level', 'Flow']].corr()
    ax3 = sns.heatmap(node_corr, annot=True, cmap='YlGnBu', fmt=".2f")
    ax3.set_title("Level / Flow Correlation")
    plt.plot()
    file_utils.save_plot(plt, "node_test_display_correlation_flow_level__correlation")
    file_utils.show_plot(plt, show=show)


def node_test_display_water_level_with_thresholds(df_node, show=False):
    from main import START_VALUE, END_VALUE, WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF

    plt.figure(figsize=(10, 8))
    plt.plot(df_node["Level"].to_numpy()[START_VALUE:END_VALUE], label="Water Level (m)")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='g', linestyle='-', label="Threshold - OFF")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Water Level (m)", fontsize=18)
    plt.legend()

    file_utils.save_plot(plt, "node_test_display_water_level_with_thresholds")
    file_utils.show_plot(plt, show=show)


def node_test_display_flow(df_node, show=False):
    from main import START_VALUE, END_VALUE

    plt.figure(figsize=(10, 8))
    plt.plot(df_node["Flow"].to_numpy()[START_VALUE:END_VALUE], label="Flow (cms)")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Flow (cms)", fontsize=18)
    plt.legend()

    file_utils.save_plot(plt, "node_test_display_flow_with_thresholds")
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

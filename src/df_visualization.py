import matplotlib.pyplot as plt
import seaborn as sns

from main import START_VALUE, END_VALUE, WATER_LEVEL_THRESHOLD_ON, WATER_LEVEL_THRESHOLD_OFF
from src import file_utils


def node_test_display_correlation_flow_level(df_node):
    print("Display Level and Flow values.")
    print("Display Level / Flow correlation")

    plt.figure(1)
    ax = sns.violinplot(x=df_node['Level'])
    ax.set_title("Level")
    plt.plot()
    file_utils.save_plot(plt, "node_test_display_correlation_flow_level__level")

    plt.figure(2)
    ax2 = sns.violinplot(x=df_node['Flow'])
    ax2.set_title("Flow")
    plt.plot()
    file_utils.save_plot(plt, "node_test_display_correlation_flow_level__flow")

    plt.figure(3, figsize=(8, 6))
    node_corr = df_node[['Level', 'Flow']].corr()
    ax3 = sns.heatmap(node_corr, annot=True, cmap='YlGnBu', fmt=".2f")
    ax3.set_title("Level / Flow Correlation")
    plt.plot()
    file_utils.save_plot(plt, "node_test_display_correlation_flow_level__correlation")

    plt.show()


def node_test_display_water_level_with_thresholds(df_node):
    plt.figure(figsize=(10, 8))
    plt.plot(df_node["Level"].to_numpy()[START_VALUE:END_VALUE], label="Water Level (m)")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-', label="Threshold - ON")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_OFF, color='g', linestyle='-', label="Threshold - OFF")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Water Level (m)", fontsize=18)
    plt.legend()

    file_utils.save_plot(plt, "node_test_display_water_level_with_thresholds")
    plt.show()


def node_test_display_flow_with_thresholds(df_node):
    plt.figure(figsize=(10, 8))
    plt.plot(df_node["Flow"].to_numpy()[START_VALUE:END_VALUE], label="Flow (cms)")
    plt.xlabel("Sample Number", fontsize=18)
    plt.ylabel("Flow (cms)", fontsize=18)
    plt.legend()

    file_utils.save_plot(plt, "node_test_display_flow_with_thresholds")
    plt.show()


def test_linear_regression(df_node):
    from sklearn.linear_model import LinearRegression

    node_model = LinearRegression()
    x = df_node['Flow'].to_numpy()[END_VALUE:].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[END_VALUE:]
    node_model.fit(x, y)

    x = df_node['Flow'].to_numpy()[START_VALUE:END_VALUE].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[START_VALUE:END_VALUE].reshape(-1, 1)
    preds = node_model.predict(x)

    plt.figure(figsize=(16, 10))
    plt.plot(preds, label="predictions")
    plt.plot(y, label="level")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-')
    plt.legend()

    file_utils.save_plot(plt, "test_linear_regression")
    plt.show()


def test_random_forest(df_node):
    from sklearn.ensemble import RandomForestRegressor

    node_model = RandomForestRegressor(n_estimators=100)
    x = df_node['Flow'].to_numpy()[:START_VALUE].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[:START_VALUE]
    node_model.fit(x, y)

    x = df_node['Flow'].to_numpy()[START_VALUE:].reshape(-1, 1)
    y = df_node['Level'].to_numpy()[START_VALUE:].reshape(-1, 1)
    preds = node_model.predict(x)

    plt.figure(figsize=(16, 10))
    plt.plot(preds, label="predictions")
    plt.plot(y, label="level")
    plt.axhline(y=WATER_LEVEL_THRESHOLD_ON, color='r', linestyle='-')
    plt.legend()

    file_utils.save_plot(plt, "test_random_forest")
    plt.show()

from datetime import timedelta

import numpy as np

from src.utils import time_utils


def join_stations_and_measurements(measurements_df, selected_station, verbose=True):
    data_by_stations = {}
    for station in measurements_df["Station Name"].unique():
        df_tmp = measurements_df[measurements_df["Station Name"] == station].copy()
        df_tmp.reset_index(drop=True, inplace=True)
        data_by_stations[station] = df_tmp
        if verbose:
            print(f"Station: {station}, length: {len(data_by_stations[station].index)}")

    return data_by_stations[selected_station]


def prepare_time_and_timenew_column(df_node):
    df_node['Time'] = [time_utils.translate_timestamp(n) for n in df_node["Timestamp"].to_list()]

    # Re-timing data to cut empty-reading times
    t0 = df_node.iloc[0]['Time']
    _increment = timedelta(minutes=5)
    new_time = [t0 + (_increment * n) for n, t in enumerate(df_node['Time'].to_numpy())]
    df_node['TimeNew'] = new_time

    return df_node


def convert_nan_values(df_node, to_format, verbose=True):
    nan_level = np.sum(np.isnan(df_node["Level"].to_numpy(), where=1.0))
    nan_flow = np.sum(np.isnan(df_node["Flow"].to_numpy(), where=1.0))
    print(f"Level nans {nan_level} out of {len(df_node.index)} values")
    print(f"Flow nans {nan_flow} out of {len(df_node.index)} values")

    if to_format == "REMOVED":
        # Remove the NAN values from the series. Other options may be considered.
        if verbose:
            print("Removing NAN values...")
        df_node_filt = df_node[~np.isnan(df_node['Level'].to_numpy(), where=True)]
        df_node_filt = df_node_filt[~np.isnan(df_node_filt['Flow'].to_numpy(), where=True)]
        df_node = df_node_filt
        nan_level = np.sum(np.isnan(df_node['Level'].to_numpy(), where=1.0))
        nan_flow = np.sum(np.isnan(df_node['Flow'].to_numpy(), where=1.0))
    else:
        # Set the NAN values to zero. Other options may be considered.
        if verbose:
            print("Replacing NAN values to zeroes...")
        df_node['Level'] = df_node['Level'].fillna(0)
        df_node['Flow'] = df_node['Flow'].fillna(0)
        nan_level = np.sum(np.isnan(df_node['Level'].to_numpy(), where=1.0))
        nan_flow = np.sum(np.isnan(df_node['Flow'].to_numpy(), where=1.0))

    if verbose:
        print(f"Level nans {nan_level} out of {len(df_node.index)} values")
        print(f"Flow nans {nan_flow} out of {len(df_node.index)} values")

    return df_node

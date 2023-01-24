from datetime import timedelta

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.utils.time_utils import HOURS_TO_SECONDS

_battery_charge = 5000  # Battery charge in mAh
_idle_current = 73.3 / 1000  # Idle current in mA
_sampling_current = 6.52  # Sampling and processing current mA
_sampling_time = 1.725502  # Sampling and processing time in s
_sampling_charge = _sampling_current * _sampling_time / HOURS_TO_SECONDS
_trans_current = 9.48  # Transmission and processing mA
_trans_time = 7.64409  # Average transmission time s
_trans_charge = _trans_current * _trans_time / HOURS_TO_SECONDS
_reception_charge = _trans_charge


class Node:
    def __init__(self, water_level_threshold_on, water_level_threshold_off):
        #print(f"Sampling and processing charge {_sampling_charge}")
        #print(f"Data transmission reception charge {_trans_charge}")
        self.battery_charge_initial = _battery_charge
        self.battery_charge_remaining = _battery_charge
        self.idle_current = _idle_current
        self.sampling_charge = _sampling_charge
        self.trans_charge = _trans_charge
        self.reception_charge = _reception_charge

        self.threshold_on = water_level_threshold_on
        self.threshold_off = water_level_threshold_off
        self.model = None
        self.prediction_mean = 0
        self.prediction_mean_l = []
        self.prediction_std_l = []
        self.prediction_std = 0
        self.level_mean = 0
        self.alpha = 0.01

        self.t0 = None
        self.tn = None
        self.tend = None
        self.n = 0
        self.dt = 0

        self.in_event = False
        self.e = 0
        self.events = []
        self.event_meaning = []

    def consume_power(self, dt_min, trans, reception):
        dt = dt_min / 60
        total_charge = self.idle_current * dt
        total_charge += self.sampling_charge
        if trans:
            total_charge += self.trans_charge
        if reception:
            total_charge += self.reception_charge
        self.battery_charge_remaining -= total_charge

        return self.battery_charge_remaining <= 0

    def start_events(self, df):
        self.in_event = df['Level'].iloc[0] >= self.threshold_on
        self.events = []
        self.event_meaning = []

    def train_model(self, x, y):
        self.model = LinearRegression()
        x = x.reshape(-1, 1)
        self.model.fit(x, y)

    def init_sampling(self, df: pd.DataFrame):
        # training data
        x = df['Flow'].to_numpy()[350000:].reshape(-1, 1)
        y = df['Level'].to_numpy()[350000:]
        self.train_model(x, y)

        df_out = df.iloc[150000:350000].copy()
        df_out.reset_index(drop=True, inplace=True)
        x_out = df_out['Flow'].to_numpy().reshape(-1, 1)
        df_out['prediction'] = self.model.predict(x_out)
        t0 = df_out['TimeNew'].iloc[0]
        x0 = x_out[0][0]  # Test
        y0 = df_out['prediction'][0]
        l0 = df_out['Level'][0]
        self.prediction_mean = y0
        self.prediction_std = y0
        self.level_mean = y0
        self.t0 = t0
        self.tn = t0
        self.dt = 0
        self.n = 1
        self.tend = df_out['TimeNew'].iloc[-1]
        self.start_events(df_out)

        return df_out, x0, y0, t0, l0

    def update_prediction_stats(self, p):
        self.prediction_mean = self.alpha * p + (1 - self.alpha) * self.prediction_mean
        self.prediction_std = self.alpha * (
                self.prediction_mean - p) ** 2 + self.alpha * self.prediction_std
        self.prediction_mean_l.append(self.prediction_mean)
        self.prediction_std_l.append(self.prediction_std)

    def update_level_stats(self, level):
        self.level_mean = self.alpha * level + (1 - self.alpha) * self.level_mean

    def get_next_sample(self, dt_min, df):
        if dt_min % 5 != 0:
            print("Next sample error, sampling rate is not multiple of 5 min")

        time_delta = timedelta(minutes=dt_min)
        t = self.tn + time_delta
        if t > self.tend:
            return 0, 0, 0, 0, False
        index_l = df.index[df['TimeNew'] == t].to_list()
        if len(index_l) > 1:
            print("Next sample warning, multiple times")
            # display(df.iloc[index_l])
        elif len(index_l) == 0:
            print(f"Next sample warning, no time {index_l}, {t}")
            dt_test = dt_min + 5
            x0, y0 = self.next_sample(dt_test, df)  # TODO complete
            return x0, y0

        index = index_l[0]
        r = df.iloc[index]
        flow = r['Flow']
        pred = r['prediction']
        level = r['Level']
        self.tn = t
        self.dt += dt_min
        self.update_prediction_stats(pred)
        self.update_level_stats(level)

        return flow, pred, level, self.dt, True

    def detect_event(self, level):
        if self.in_event:
            if level < self.threshold_off:
                self.events.append(self.dt)
                self.event_meaning.append("ended")
                self.in_event = False
        else:
            if level >= self.threshold_on:
                self.events.append(self.dt)
                self.event_meaning.append("started")
                self.in_event = True

    def end_sample_event(self, transmit, receive, dt_min, level):
        if transmit:
            self.detect_event(level)
        self.consume_power(dt_min, transmit, receive)
        self.n += 1

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_preprocess(file):
    data = pd.read_csv(file)
    data = data.astype({
        'time': 'int64',
        'class': 'int64',
        'label': 'int64',
        'channel1': 'float64',
        'channel2': 'float64',
        'channel3': 'float64',
        'channel4': 'float64',
        'channel5': 'float64',
        'channel6': 'float64',
        'channel7': 'float64',
        'channel8': 'float64'
    })

    # Seperate the time signals

    # Find the indices where the 'time' column resets to 0
    reset_indices = data.index[data['time'] == 1].tolist()

    # Split the data into separate DataFrames based on the reset indices
    split_data = []

    for i in range(len(reset_indices) - 1):
        split_data.append(data.iloc[reset_indices[i]:reset_indices[i + 1]])

    # Append the last segment of data
    split_data.append(data.iloc[reset_indices[-1]:])

    # Isolate gesture windows using groupby
    gesture_windows = []

    for signals in split_data:
        # Group by the 'class' column and extract segments
        grouped = signals.groupby((signals['class'].diff() != 0).cumsum())
        for _, group in grouped:
            gesture_windows.append(group)

    return gesture_windows

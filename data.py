import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch
import numpy as np

weather_data = pd.read_csv('sotonmet.txt')
cols = weather_data.columns


def date_conversion(date_string):
    """Convert Sotonmet string timestamp to Datetime object."""
    return datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')


weather_data[cols[0]] = weather_data[cols[0]].apply(date_conversion)
weather_data[cols[2]] = weather_data[cols[2]].apply(date_conversion)

# Extracting all ground truth data
all_reading_times = weather_data[cols[2]].values
true_tide_heights = weather_data[cols[10]].values
weather_data.dropna(inplace=True)

# Training data
reading_times = weather_data[cols[2]]

minute_constant = 60e9  # Scale readings to minutes (sensible length-scale)
start_time = weather_data[cols[2]].values[0]

# Create normalised time tensors
scaled_reading_times = torch.tensor((reading_times.values - start_time).astype(float)/minute_constant, dtype=torch.float64)
scaled_all_reading_times = torch.tensor((all_reading_times - start_time).astype(float)/minute_constant, dtype=torch.float64)

quantisation_variance = 0.02886  # Variance of uniform noise due to quantisation of tide height : 0.1^2/12
sigma_n = quantisation_variance**0.5

tide_heights = torch.tensor(weather_data[cols[5]].values, dtype=torch.float64)
data_noise = quantisation_variance*torch.randn(len(tide_heights))  # Approximate w/ Gaussian
tide_heights += data_noise  # Adding quantisation noise to readings to improve stability

tide_std, tide_mean = torch.std_mean(tide_heights)
scaled_tide_heights = (tide_heights-tide_mean)/tide_std
scaled_true_heights = (true_tide_heights-tide_mean.numpy())/tide_std.numpy()  # Scale ground truth for ref. plotting


def column_plotter(dataframe, column_indices, axes, color="tab:red"):
    """Plot select columns from Dataframe given indices"""
    for i, col_no in enumerate(column_indices):
        axes[i].scatter(reading_times, dataframe[cols[col_no]], marker="x", s=5, color=color)
        axes[i].set_ylabel(cols[col_no])
    plt.tight_layout()


@np.vectorize
def itt(scaled_time_reading):
    """Inverse time transform (ITT)
    Convert scaled time readings to original np datetime.

    Args:
        scaled_time_reading: Vector or tensor of scaled time readings measured in minutes from start time.
    """
    elapsed_ns = np.timedelta64(int(scaled_time_reading*minute_constant), "ns")
    return start_time + elapsed_ns


@np.vectorize
def iht(scaled_tide_reading):
    """Inverse height transform (IHT)
    Convert scaled tide heights back to raw value in metres

    Args:
        scaled_tide_reading: Vector or tensor of scaled tide heights, normalised to training data mean and variance.
    """
    return scaled_tide_reading * tide_std + tide_mean

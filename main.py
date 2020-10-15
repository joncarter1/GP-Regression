import numpy as np
import matplotlib.pyplot as plt
from numpy import shape, transpose, array, matrix
from kernels import sinusoidal_covar_generator, exp_covar_generator, add_noise
from gps import GaussianProcess
import pandas as pd
import datetime

def exp_covar_generator(s, l):
    """Generate sinusoidal covariance function"""
    def covariance_func(x1, x2):
        return (s ** 2) * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * l ** 2))
    return covariance_func


exp_covar = exp_covar_generator(1, 50)


if __name__ == "__main__":
    data = pd.read_csv('sotonmet.txt')
    cols = data.columns
    update_times = data["Update Date and Time (ISO)"]
    data.dropna(inplace=True)

    update_times = data[cols[0]]
    update_duration = data[cols[1]]
    reading_times = data[cols[2]]
    air_temp = data[cols[4]]
    tide_heights = data[cols[5]]

    date_conversion = lambda date_string: datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
    reading_times = reading_times.apply(date_conversion)
    scaled_reading_times = reading_times.values.astype(float)/1e11
    scaled_reading_times -= np.min(scaled_reading_times)

    scaled_tide_heights = tide_heights - np.mean(tide_heights)

    gp1 = GaussianProcess(covar_func=exp_covar, sigma=0.03, training_data=scaled_reading_times, labels=scaled_tide_heights)
    test_times = np.linspace(np.min(scaled_reading_times), np.max(scaled_reading_times), 1000)
    test_means, test_vars = gp1.compute_predictive_means_vars(test_times)

    plt.figure()
    plt.scatter(scaled_reading_times, scaled_tide_heights, s=5, marker="x")
    plt.plot(test_times, test_means, color="tab:blue")
    plt.fill_between(test_times, test_means - np.diag(test_vars) ** 0.5, test_means + np.diag(test_vars) ** 0.5,
                     alpha=0.3, color="tab:blue")


    """fig, ax = plt.subplots(2, 1)
    ax[0].scatter(reading_times, tide_heights, marker="x", s=5)
    ax[1].scatter(reading_times, air_temp, marker="x", s=5)
    plt.tight_layout()"""
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape, transpose, array, matrix
from kernels import sinusoidal_covar_generator, exp_covar_generator, add_noise
from gps import GaussianProcess
import pandas as pd

def exp_covar_generator(s, l):
    """Generate sinusoidal covariance function"""
    def covariance_func(x1, x2):
        return (s ** 2) * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * l ** 2))
    return covariance_func

exp_covar = exp_covar_generator(1, 1)
exp_covar2 = exp_covar_generator(11, 1)

def exp_covariance_func(x1, x2, hyperparams):
    v_scale, h_scale = hyperparams
    return (v_scale ** 2) * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * h_scale ** 2))

if __name__ == "__main__":
    train_times = np.linspace(0, 10, 10)
    results = np.sin(train_times)
    gp1 = GaussianProcess(covar_func=exp_covar, sigma=0.1, training_data=train_times, labels=results)
    test_times = np.linspace(-5, 15, 100)
    test_means, test_vars = gp1.compute_predictive_means_vars(test_times)
    plt.figure()
    plt.scatter(train_times, results)
    plt.plot(test_times, test_means, color="tab:blue")
    plt.fill_between(test_times, test_means-np.diag(test_vars)**0.5, test_means+np.diag(test_vars)**0.5,
                     alpha=0.3, color="tab:blue")
    plt.show()

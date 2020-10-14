import numpy as np
import matplotlib.pyplot as plt
from numpy import shape, transpose, array, matrix
from kernels import IsoSQEKernel, PeriodicKernel
from gps import GaussianProcess
import pandas as pd
import torch

def exp_covar_generator(s, l):
    """Generate sinusoidal covariance function"""
    def covariance_func(x1, x2):
        return (s ** 2) * torch.exp(-(torch.norm(x1 - x2) ** 2) / (2 * l ** 2))
    return covariance_func


l_scale, v_scale = 1, 1.0

iso_params = torch.tensor([l_scale, v_scale], requires_grad=True)
iso = IsoSQEKernel(iso_params)

period = 6
periodic_params = torch.tensor([l_scale, v_scale, period], requires_grad=True)
periodic_kernel = PeriodicKernel(periodic_params)

if __name__ == "__main__":
    train_times = torch.linspace(0, 10, 10, requires_grad=False)
    results = torch.sin(train_times)
    gp1 = GaussianProcess(covar_kernel=iso+periodic_kernel, sigma=0.2, training_data=train_times, labels=results)
    test_times = torch.linspace(-5, 15, 100)

    test_means, test_vars = gp1.compute_predictive_means_vars(test_times)
    plt.figure()
    plt.scatter(train_times.detach().numpy(), results.detach().numpy())
    plt.plot(test_times, test_means, color="tab:blue")
    plt.fill_between(test_times, test_means-np.diag(test_vars)**0.5, test_means+np.diag(test_vars)**0.5,
                     alpha=0.3, color="tab:blue")
    plt.show()

    gp1.optimise_hyperparams(100)

    test_means, test_vars = gp1.compute_predictive_means_vars(test_times)
    plt.figure()
    plt.scatter(train_times.detach().numpy(), results.detach().numpy())
    plt.plot(test_times, test_means, color="tab:blue")
    plt.fill_between(test_times, test_means - np.diag(test_vars) ** 0.5, test_means + np.diag(test_vars) ** 0.5,
                     alpha=0.3, color="tab:blue")
    plt.show()

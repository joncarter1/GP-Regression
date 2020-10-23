import numpy as np
import matplotlib.pyplot as plt
from numpy import shape, transpose, array, matrix
from kernels import IsoSQEKernel, PeriodicKernel, Matern12Kernel, Matern32Kernel
from models import GaussianProcess
import pandas as pd
import torch

l_scale, v_scale = 1.0, 1.0
period = 2*np.pi
iso_params = torch.tensor([l_scale, v_scale], requires_grad=True)
iso = IsoSQEKernel(iso_params)
matern12 = Matern12Kernel(iso_params)
matern32 = Matern32Kernel(iso_params)
periodic_params = torch.tensor([l_scale, v_scale, period], requires_grad=True)
periodic = PeriodicKernel(periodic_params)

if __name__ == "__main__":
    train_times = torch.linspace(0, 10, 11, requires_grad=False)
    results = torch.sin(train_times)

    gp1 = GaussianProcess(covar_kernel=iso, sigma_n=0.1, training_data=train_times, labels=results)
    test_times = torch.linspace(-5, 15, 100)

    test_means, test_vars = gp1.compute_predictive_means_vars(test_times)

    plt.figure()
    plt.scatter(train_times.detach().numpy(), results.detach().numpy())
    plt.plot(test_times, test_means, color="tab:blue")
    plt.fill_between(test_times, test_means-np.diag(test_vars)**0.5, test_means+np.diag(test_vars)**0.5,
                     alpha=0.3, color="tab:blue")
    plt.show()

    print(gp1.compute_marginal_nll())
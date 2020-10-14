import numpy as np
import torch

def add_noise(signal_array, sigma):
    noise_vector = np.random.normal(0,sigma, size=np.shape(signal_array))
    return signal_array + noise_vector


def exp_covar_generator(s, l):
    """Generate sinusoidal covariance function"""
    def covariance_func(x1, x2):
        return (s ** 2) * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * l ** 2))
    return covariance_func


def sinusoidal_covar_generator(s, l, p):
    """Generate sinusoidal covariance function"""
    def covariance_func(x1, x2):
        return (s ** 2) * np.exp(-2 * (np.sin((np.pi / p) * np.linalg.norm(x1 - x2)) ** 2) / (l ** 2))
    return covariance_func


def null_covariance(x1, x2):
    return NotImplementedError


class CovarianceKernel:
    def __init__(self, hyperparams, covariance_function=lambda x: NotImplementedError):
        if not isinstance(hyperparams, list):
            self.hyperparams = [hyperparams]
        else:
            self.hyperparams = hyperparams
        self.covariance_function = covariance_function

    def __call__(self, x1, x2):
        """Call to kernel function"""
        return self.covariance_function(x1, x2)

    def __add__(self, other_kernel):
        """Kernel addition"""
        combined_hyperparams = self.hyperparams + other_kernel.hyperparams
        new_covariance_function = lambda *x: self.covariance_function(*x) + other_kernel.covariance_function(*x)
        return CovarianceKernel(combined_hyperparams, new_covariance_function)

    def __mul__(self, other_kernel):
        """Kernel multiplication"""
        combined_hyperparams = self.hyperparams + other_kernel.hyperparams
        new_covariance_function = lambda *x: self(*x) * other_kernel(*x)
        return CovarianceKernel(combined_hyperparams, new_covariance_function)


pi = torch.tensor(np.pi)


class PeriodicKernel(CovarianceKernel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        def periodic_covariance(x1, x2):
            l_scale, v_scale, period = self.hyperparams[0]  # Un-packing sinusoidal hyper-parameters
            return (v_scale ** 2) * torch.exp(-2 * (torch.sin((pi / period) * torch.norm(x1 - x2)) ** 2) / (l_scale ** 2))

        self.covariance_function = periodic_covariance


class IsoSQEKernel(CovarianceKernel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        def iso_sqe_covariance(x1, x2):
            l_scale, v_scale = self.hyperparams[0]
            return (v_scale ** 2) * torch.exp(-(torch.norm(x1 - x2) ** 2) / (2 * l_scale ** 2))

        self.covariance_function = iso_sqe_covariance


if __name__ == "__main__":
    sqe = PeriodicKernel(torch.tensor([2, 2, 2]))
    iso = IsoSQEKernel(torch.tensor([1, 1]))
    combined = sqe + iso
    x1, x2 = 2, 2.5
    x1, x2 = torch.tensor(x1), torch.tensor(x2)
    combined2 = sqe * iso
    combined3 = combined + combined2
    print(sqe(x1, x2))
    print(sqe.hyperparams)
    sqe.hyperparams = torch.tensor([1, 1, 1])
    print(sqe(x1, x2))
    print(sqe.hyperparams)
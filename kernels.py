import numpy as np
import torch
from utils import expand_1d, compute_distance_matrix

class CovarianceKernel:
    def __init__(self, hyperparams, covariance_function=None):
        """
        Args:
            hyperparams: Kernel hyper-params (either a list or PyTorch tensor)
            covariance_function: Kernel co-variance function (if composite i.e. not already specified by sub-class)
        """
        if not isinstance(hyperparams, list):
            self.hyperparams = [hyperparams]
        else:
            self.hyperparams = hyperparams

        if covariance_function is None:
            self.covariance_function = lambda x: NotImplementedError
        else:
            self.covariance_function = covariance_function

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
        """Call to kernel function
        """
        x1, x2 = expand_1d([x1, x2])
        return self.covariance_function(x1, x2)

    def __repr__(self):
        print_str = ""
        for kernel_params in self.hyperparams:
            print_str += str(kernel_params.exp())
        return print_str

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


class IsoSQEKernel(CovarianceKernel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        def iso_sqe_covariance(x1, x2):
            l_scale, v_scale = self.hyperparams[0].exp()  # Un-packing (log) hyper-parameters
            squared_distance_matrix = compute_distance_matrix(x1, x2)
            output = (v_scale ** 2) * torch.exp(-squared_distance_matrix / (2 * l_scale ** 2))
            return torch.clamp(output, 0, float(v_scale**2))
        self.covariance_function = iso_sqe_covariance


class PeriodicKernel(CovarianceKernel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        def periodic_covariance(x1, x2):
            l_scale, v_scale, period = self.hyperparams[0].exp()  # Un-packing (log) hyper-parameters
            distance_matrix = compute_distance_matrix(x1, x2)**0.5
            output = (v_scale ** 2) * torch.exp(-2 * (torch.sin((pi / period) * distance_matrix)**2 / (l_scale ** 2)))
            return torch.clamp(output, 0, float(v_scale**2))

        self.covariance_function = periodic_covariance


class Matern12Kernel(CovarianceKernel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        def matern_12_covariance(x1, x2):
            """Matern co-variance function for nu = 1/2"""
            l_scale, v_scale = self.hyperparams[0].exp()
            distance_matrix = compute_distance_matrix(x1, x2)**0.5
            return (v_scale**2) * torch.exp(-distance_matrix/l_scale)

        self.covariance_function = matern_12_covariance


class Matern32Kernel(CovarianceKernel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        def matern_32_covariance(x1, x2):
            """Matern co-variance function for nu = 3/2"""
            l_scale, v_scale = self.hyperparams[0].exp()
            distance_matrix = compute_distance_matrix(x1, x2)**0.5
            return (v_scale**2) * (1 + (3**0.5)*distance_matrix/l_scale) * torch.exp(-(3**0.5)*distance_matrix/l_scale)

        self.covariance_function = matern_32_covariance


class QuadraticKernel(CovarianceKernel):
    def __init__(self, hyperparams, alpha):
        super().__init__(hyperparams)
        self.alpha = alpha

        def quadratic_covariance(x1, x2):
            l_scale, v_scale, alpha = self.hyperparams[0].exp()
            squared_distance_matrix = compute_distance_matrix(x1, x2)
            return (v_scale**2) * (1 + squared_distance_matrix / (2*alpha*l_scale**2))**(-alpha)
        self.covariance_function = quadratic_covariance


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
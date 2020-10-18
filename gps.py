import numpy as np
from numpy import shape, transpose, array, matrix
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import expand_1d

class GaussianProcess:
    def __init__(self,
                 covar_kernel,
                 training_data=None,
                 labels=None,
                 sigma_n=0,
                 learn_noise=False,
                 lr=1e-2,
                 mean_func=None):
        self.training_data = training_data
        self.labels = labels
        self.mean_func = mean_func
        self.covar_kernel = covar_kernel
        self.learn_noise = learn_noise
        self.log_sigma = torch.tensor(np.log(sigma_n), requires_grad=learn_noise)
        self.trainable_parameters = self.covar_kernel.hyperparams+[self.log_sigma]
        self.lr = lr  # Learning rate for marginal likelihood gradient descent
        self.updated = False  # Up to date inverse co-variance stored?
        self.ill_conditioned = False  # Co-variance matrix condition number high?

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def compute_covariance_matrix(self, training_data, verbose=False):
        covar_matrix = self.covar_kernel(training_data, training_data)
        noise_eye = (self.log_sigma.exp()**2)*torch.eye(len(training_data))
        covar_matrix += noise_eye
        condition_number = np.linalg.cond(covar_matrix.detach().numpy())
        self.ill_conditioned = condition_number > 1e8
        if verbose or self.ill_conditioned:
            print(f"Condition number : {condition_number}")
        return covar_matrix

    @staticmethod
    def unstable_gaussian_nll(y, mu, covar_matrix, dims=1, verbose=False):
        """Non-Cholesky negative log-likelihood method kept for testing."""
        inv_cov = covar_matrix.inverse()
        data_fit_term = -0.5 * torch.mm((y - mu).T, torch.mm(inv_cov, y - mu))
        _, log_det = covar_matrix.slogdet()
        complexity_term = -0.5 * log_det
        if verbose:
            print("Data fit : ", data_fit_term)
            print("Complexity : ", complexity_term)
        return -1 * (data_fit_term + complexity_term - (dims / 2) * np.log(2 * np.pi))

    @staticmethod
    def gaussian_nll(y, mu, covar_matrix, dims=1, jitter=1e-4, verbose=False):
        """Compute -ve log-likelihood of data under a multivariate Gaussian."""
        y, mu = expand_1d([y, mu])
        covar_matrix += (jitter**2)*torch.eye(*covar_matrix.shape)
        condition_number = np.linalg.cond(covar_matrix.detach().numpy())
        if condition_number > 1e10:
            print(f"Condition number : {condition_number}")
        L_covar = torch.cholesky(covar_matrix)
        inv_covar = torch.cholesky_inverse(L_covar)
        alpha = torch.mm(inv_covar, y-mu)
        data_fit_term = -0.5 * torch.mm((y - mu).T, alpha)
        complexity_term = -torch.sum(torch.diagonal(L_covar).log())
        if verbose:
            print("Data fit : ", data_fit_term)
            print("Complexity : ", complexity_term)
        return -1 * (data_fit_term + complexity_term - (dims/2)*np.log(2*np.pi))

    def compute_test_nll(self, inputs, labels):
        """Compute negative log-likelihood of a test set."""
        mu, covar_matrix = self.compute_predictive_means_vars(inputs, to_np=False)
        return self.gaussian_nll(labels, mu, covar_matrix)

    def compute_marginal_nll(self, verbose=False):
        """Compute negative log-likelihood of data under mv Gaussian."""
        covar_matrix = self.compute_covariance_matrix(self.training_data)
        return self.gaussian_nll(self.labels, torch.tensor(0), covar_matrix)

    def optimise_hyperparams(self, epochs=10, lr=None):
        """Optimise hyper-parameters via gradient descent of the marginal likelihood."""
        print(f"Old hyper-parameters: {self.covar_kernel}")
        if self.learn_noise:
            print(f"Old noise std : {self.sigma}")
        loss = self.compute_marginal_nll(verbose=False)
        print(f"Negative marginal likelihood : {loss}")
        lr = self.lr if lr is None else lr
        optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr)
        for _ in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.compute_marginal_nll(verbose=False)
            if self.ill_conditioned:
                print("Aborting hyper-parameter optimisation")
                break
            loss.backward()
            optimizer.step()
        print(f"New hyper-parameters: {self.covar_kernel}")
        if self.learn_noise:
            print(f"New noise std : {self.sigma}")
        print(f"Negative log-likelihood : {loss}")

    def compute_predictive_means_vars(self, test_data, training_data=None, labels=None, jitter=1e-4, to_np=True):
        """Compute predictive mean and variance over a set of test points."""
        if training_data is None:
            training_data = self.training_data  # Use all provided (non-causal)
        if labels is None:
            labels = self.labels
        covar_matrix = self.compute_covariance_matrix(training_data)
        covar_matrix += (jitter ** 2) * torch.eye(*covar_matrix.shape)
        condition_number = np.linalg.cond(covar_matrix.detach().numpy())
        if condition_number > 1e10:
            print(f"Condition number : {condition_number}")
        L_covar = torch.cholesky(covar_matrix)
        inv_covar = torch.cholesky_inverse(L_covar)
        KxX = self.covar_kernel(test_data, training_data)
        product1 = torch.mm(KxX, inv_covar)
        mu_array = torch.mv(product1, labels)
        product2 = torch.mm(product1, torch.transpose(KxX, 0, 1))
        auto_cov = self.covar_kernel(test_data, test_data)
        var_array = auto_cov - product2
        if not to_np:
            return mu_array, var_array
        return mu_array.detach().numpy(), var_array.detach().numpy()


class LookaheadGP(GaussianProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_lookahead_predictive_means_vars(self, test_data):
        """Compute lookahead predictive mean and variance at time T.
            Any training data from future time points is discarded to perform inference

        Args:
            test_data: Tensor of test points in time series.
        """
        T_start = torch.min(test_data)
        causal_training_data = self.training_data[self.training_data < T_start]
        causal_labels = self.labels[self.training_data < T_start]
        return self.compute_predictive_means_vars(test_data, causal_training_data, causal_labels)



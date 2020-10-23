import numpy as np
import torch
from tqdm import tqdm
from GPy2.utils import gaussian_nll
cpu = torch.device("cpu")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        self.training_data.to(dev)
        self.labels.to(dev)
        self.mean_func = mean_func
        self.covar_kernel = covar_kernel
        self.learn_noise = learn_noise
        self.log_sigma = torch.tensor(np.log(sigma_n), requires_grad=learn_noise)
        self.log_sigma.to(dev)
        self.trainable_parameters = self.covar_kernel.hyperparams+[self.log_sigma]
        self.lr = lr  # Learning rate for marginal likelihood gradient descent
        self.ill_conditioned = False

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def compute_covariance_matrix(self, training_data, jitter=0, verbose=False):
        covar_matrix = self.covar_kernel(training_data, training_data)
        noise_eye = (self.sigma**2 + jitter ** 2)*torch.eye(*covar_matrix.shape)
        covar_matrix += noise_eye
        condition_number = np.linalg.cond(covar_matrix.detach().numpy())
        self.ill_conditioned = condition_number > 1e8
        if verbose or self.ill_conditioned:
            print(f"Condition number : {condition_number}")
        return covar_matrix

    def compute_test_nll(self, inputs, labels):
        """Compute negative log-likelihood of a test set."""
        mu, covar_matrix = self.compute_predictive_means_vars(inputs, to_np=False)
        return gaussian_nll(labels, mu, covar_matrix)

    def compute_marginal_nll(self, verbose=False):
        """Compute negative log-likelihood of data under zero-mean multivariate Gaussian."""
        covar_matrix = self.compute_covariance_matrix(self.training_data)
        return gaussian_nll(self.labels, torch.tensor(0, device=dev), covar_matrix, verbose)

    def optimise_hyperparams(self, epochs=10, lr=None):
        """Optimise hyper-parameters via gradient descent of the marginal likelihood.

        Args:
            epochs: No. epochs to optimise for.
            lr: Specified learning rate, otherwise stored value use.
        """
        print(f"Old hyper-parameters: {self.covar_kernel}")
        if self.learn_noise:
            print(f"Old noise std : {self.sigma}")
        loss = self.compute_marginal_nll(verbose=False)
        lr = self.lr if lr is None else lr
        optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr)
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.compute_marginal_nll(verbose=False)
            if self.ill_conditioned:
                print(f"Aborting hyper-parameter optimisation at epoch {epoch}")
                break
            loss.backward()
            optimizer.step()
        print(f"New hyper-parameters: {self.covar_kernel}")
        if self.learn_noise:
            print(f"New noise std : {self.sigma}")
        print(f"Negative log-likelihood : {loss}")

    def compute_predictive_means_vars(self, test_data, training_data="stored", labels="stored", jitter=1e-4, to_np=True):
        """Compute predictive mean and variance over a set of test points.

        Args:
            test_data: Tensor of test points in time series.
            training_data: Used to condition GP, uses stored if None.
            labels: Used to condition GP, uses stored labels if None.
            jitter: Noise added to co-variance matrix diagonal for stability.
            to_np: If casting result to Numpy array.
        """
        if training_data == "stored":
            training_data = self.training_data  # Use all provided
            labels = self.labels

        if training_data.nelement():
            covar_matrix = self.compute_covariance_matrix(training_data, jitter)
            condition_number = np.linalg.cond(covar_matrix.detach().numpy())
            if condition_number > 1e10:
                print(f"Condition number : {condition_number}")
            L_covar = torch.cholesky(covar_matrix)
            inv_covar = torch.cholesky_inverse(L_covar)
            KxX = self.covar_kernel(test_data, training_data)
            product1 = torch.mm(KxX, inv_covar)
            mu_array = torch.mv(product1, labels)
            product2 = torch.mm(product1, torch.transpose(KxX, 0, 1))
        else:
            product2 = 0  # No conditioning of co-variance matrix
            mu_array = torch.tensor([0])  # Zero-mean prior
        auto_cov = self.covar_kernel(test_data, test_data)
        var_array = auto_cov - product2
        if not to_np:
            return mu_array, var_array
        return mu_array.detach().numpy(), var_array.detach().numpy()


class LookaheadGP(GaussianProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_lookahead_predictive_means_vars(self, test_data, lookahead, jitter=1e-4, to_np=True):
        """Compute lookahead predictive mean and variance at time T.
            Any training data from future time points is discarded to perform inference

        Args:
            lookahead: Lookahead measured in units of time
            test_data: Tensor of test points in time series.
            jitter: Noise added to co-variance matrix diagonal for stability.
            to_np: If casting result to Numpy array.
        """

        mus, sigmas = torch.tensor([]), torch.tensor([])
        for t in tqdm(test_data):
            lookahead_training_data = self.training_data[self.training_data <= t - lookahead]
            lookahead_labels = self.labels[self.training_data <= t - lookahead]
            mu, sigma = self.compute_predictive_means_vars(t, lookahead_training_data, lookahead_labels,
                                                           jitter=jitter, to_np=False)
            mus = torch.cat([mus, mu])
            sigmas = torch.cat([sigmas, sigma])

        mus, sigmas = mus.detach(), sigmas.detach()
        if not to_np:
            return mus, sigmas
        return mus.numpy(), sigmas.numpy()


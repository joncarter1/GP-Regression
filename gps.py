import numpy as np
from numpy import shape, transpose, array, matrix
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

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
        if training_data is not None:
            self._update()

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def slow_covariance(self, X1, X2):
        covar_matrix = torch.zeros([len(X1), len(X2)])
        for i, element in enumerate(X1):
            for j, element2 in enumerate(X2):
                covar_matrix[i][j] = self.covar_kernel(element, element2)
        return covar_matrix

    def compute_covariance(self, X1, X2):
        return self.covar_kernel(X1, X2)

    def _update(self, verbose=False):
        """Update stored covariance/inverse as hyper-params/data changes for computational saving."""
        covar_matrix = self.compute_covariance(self.training_data, self.training_data)
        self.covar_matrix = covar_matrix + (self.log_sigma.exp()**2)*torch.eye(len(self.training_data))
        condition_number = np.linalg.cond(self.covar_matrix.detach().numpy())
        self.ill_conditioned = condition_number > 1e8
        if verbose or self.ill_conditioned:
            print(f"Condition number : {condition_number}")
        self.inv_cov = self.covar_matrix.inverse()
        self.updated = True

    def compute_nll(self, verbose=False):
        self._update(verbose)
        term1 = -0.5 * torch.dot(self.labels, torch.mv(self.inv_cov, self.labels))
        _, log_det = torch.slogdet(self.covar_matrix)
        return -1*(term1 - 0.5 * log_det - (len(self.training_data) / 2) * 2 * np.pi)

    def optimise_hyperparams(self, epochs=10, lr=None):
        """Optimise hyper-parameters via gradient descent of the marginal likelihood."""
        print(f"Old hyper-parameters: {self.covar_kernel}")
        if self.learn_noise:
            print(f"Old noise std : {self.sigma}")
        loss = self.compute_nll(verbose=False)
        print(f"Negative log-likelihood : {loss}")
        lr = self.lr if lr is None else lr
        optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr)
        for _ in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.compute_nll(verbose=False)
            if self.ill_conditioned:
                print("Aborting hyper-parameter optimisation")
                break
            loss.backward()
            optimizer.step()
        print(f"New hyper-parameters: {self.covar_kernel}")
        if self.learn_noise:
            print(f"New noise std : {self.sigma}")
        print(f"Negative log-likelihood : {loss}")

    def compute_predictive_means_vars(self, test_data):
        KxX = self.compute_covariance(test_data, self.training_data)
        self._update(verbose=True)
        product1 = torch.mm(KxX, self.inv_cov)
        mu_array = torch.mv(product1, self.labels)
        product2 = torch.mm(product1, torch.transpose(KxX, 0, 1))
        autocovariance = self.covar_kernel(test_data, test_data)
        var_array = autocovariance - product2
        return mu_array.detach().numpy(), var_array.detach().numpy()

    def plot_predictive(self, test_data):
        mu_array, var_array = self.compute_predictive_means(test_data)
        x_data = test_data.numpy()
        plt.scatter(self.training_data.numpy(), self.labels.numpy(), s=4)
        plt.scatter(x_data, mu_array, color='green', s=4)
        samples = []
        for _ in range(100):
            sample = np.random.multivariate_normal(mu_array, var_array)
            samples.append(sample)
        max_array = np.maximum.reduce(samples)
        min_array = np.minimum.reduce(samples)
        plt.scatter(test_data, min_array, s=4)
        plt.scatter(test_data, max_array, s=4)

    def plot(self):
        no_plots = len(self.plots)
        rows = int(np.ceil(no_plots / 3))
        cols = int(np.ceil(no_plots / rows))
        print(rows, cols)
        plt.figure(figsize=(18, 8))
        # fig, axs = plt.subplots(rows, cols)
        for i in range(no_plots):
            plt.subplot(rows, cols, i + 1)
            plt.plot(self.plots[i]['x'], self.plots[i]['y'])
            plt.xlabel(self.plots[i]['xlabel'])
            plt.ylabel(self.plots[i]['ylabel'])
            plt.title(self.plots[i]['title'])
            plt.show()
        plt.tight_layout(pad=3.4, w_pad=1.5, h_pad=3.0)
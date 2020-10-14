import numpy as np
from numpy import shape, transpose, array, matrix
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

class GaussianProcess:
    def __init__(self, covar_kernel, training_data=None, labels=None, sigma=0, mean_func=None):
        self.training_data = training_data
        self.labels = labels
        self.mean_func = mean_func
        self.covar_kernel = covar_kernel
        self.sigma = sigma
        self.lr = 1e-2  # Learning rate for marginal likelihood gradient descent
        self.updated = False  # Up to date inverse co-variance stored?
        if training_data is not None:
            self._update()

    def compute_covariance(self, X1, X2):
        covar_matrix = torch.zeros([len(X1), len(X2)])
        for i, element in enumerate(X1):
            for j, element2 in enumerate(X2):
                covar_matrix[i][j] = self.covar_kernel(element, element2)
        return covar_matrix

    def _update(self, verbose=False):
        """Update stored covariance/inverse as hyper-params/data changes for computational saving."""
        covar_matrix = self.compute_covariance(self.training_data, self.training_data)
        self.covar_matrix = covar_matrix + torch.eye(len(self.training_data)) * self.sigma ** 2
        condition_number = np.linalg.cond(self.covar_matrix.detach().numpy())
        if verbose or condition_number > 10e10:
            print(f"Condition number : {condition_number}")
        self.inv_cov = torch.inverse(self.covar_matrix)
        self.updated = True

    def compute_nll(self, verbose=False):
        self._update(verbose)
        term1 = -0.5 * torch.dot(self.labels, torch.mv(self.inv_cov, self.labels))
        _, log_det = torch.slogdet(self.covar_matrix)
        return -1*(term1 - 0.5 * log_det - (len(self.training_data) / 2) * 2 * np.pi)

    def optimise_hyperparams(self, epochs=10):
        """Optimise hyper-parameters via gradient descent of the marginal likelihood."""
        print(self.covar_kernel.hyperparams)
        optimizer = torch.optim.Adam(self.covar_kernel.hyperparams, lr=self.lr)
        for _ in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.compute_nll(verbose=False)
            loss.backward()
            if not (epochs % (epochs//10)):
                tqdm.write(f"Hyper-parameters: {self.covar_kernel.hyperparams}")
                tqdm.write(f"Negative log-likelihood : {loss}")
            optimizer.step()

    def compute_predictive_means_vars(self, test_data):
        KxX = self.compute_covariance(test_data, self.training_data)
        if self.inv_cov is None or not self.updated:
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
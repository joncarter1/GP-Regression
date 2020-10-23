from data import reading_times, scaled_reading_times, tide_heights, tide_std, scaled_all_reading_times, scaled_tide_heights,\
     itt, iht, true_tide_heights, all_reading_times
import numpy as np
import torch
from GPy2.models import GaussianProcess
from GPy2.utils import gaussian_nll, dev, cpu
import matplotlib.pyplot as plt


def compute_gp_performance(gp, jitter=0):
    """Compute performance metrics of a given GP on the data set.

    Args:
            gp: GaussianProcess object.
            jitter: Standard deviation of noise added to output co-variance matrix for stability.
    """

    # Get predictive distribution over test points
    test_means, test_vars = gp.compute_predictive_means_vars(scaled_all_reading_times, jitter=jitter, to_np=True)
    # Un-normalise means and covariances back to metres
    test_predictions = iht(test_means)
    test_covariance = (tide_std.cpu() ** 2) * test_vars
    print("covar")
    # Compute NLL for all test points and missing points
    test_nll = gaussian_nll(torch.tensor(true_tide_heights), torch.tensor(test_predictions),
                               torch.tensor(test_covariance, device=dev), jitter=jitter) / len(true_tide_heights)
    print("2")
    # Computing RMSEs in metres
    test_rmse = np.mean((true_tide_heights - test_predictions) ** 2) ** 0.5

    print(f"Marginal LL : {-gp.compute_marginal_nll()}")
    print(f"Test data LL : {-test_nll}")
    print(f"Test data rmse : {test_rmse}")

    return


def gp_inference(covar_kernel, epochs=250, sigma_n=0.1, jitter=0, lr=1e-3):
    """Form and optimise zero-mean GP for a set no. epochs / learning rate.

        Args:
            covar_kernel: Covariance function used by GP
            epochs
            sigma_n: GP noise added to training data covariance matrix
            jitter: Additional noise added to test output covariance matrix for stability
            lr: Adam optimiser learning rate for optimising hyper-parameters of GP.
    """

    gp = GaussianProcess(covar_kernel=covar_kernel, sigma_n=sigma_n, training_data=scaled_reading_times,
                         labels=scaled_tide_heights, learn_noise=False)
    print("Prior to optimisation:\n")
    compute_gp_performance(gp, jitter=jitter)
    # Maximising marginal likelihood w.r.t hyper-parameters
    gp.optimise_hyperparams(epochs, lr=lr)
    # Sample GP in range +/-40% of data
    sample_times = torch.linspace(-0.4 * torch.max(scaled_reading_times), 1.4 * torch.max(scaled_reading_times), 1000)
    # Sample times converted back to timestamps
    transformed_times = itt(sample_times)
    # Converting distribution back to metres
    sample_means, sample_vars = gp.compute_predictive_means_vars(sample_times, jitter=jitter)
    sample_means, sample_vars = iht(sample_means), (tide_std ** 2)*sample_vars
    # Diagonal covariance for uncertainty plots.
    sigma_vector = np.diag(sample_vars) ** 0.5

    print("After optimisation:\n")
    compute_gp_performance(gp, jitter=jitter)

    plt.figure(figsize=(8, 5))
    # Plot results back in original units.
    plt.plot(transformed_times, sample_means, color="tab:blue", label="GP mean")

    for i in range(3):
        function_draw = np.random.multivariate_normal(sample_means, sample_vars)
        if i == 0:
            plt.plot(transformed_times, function_draw, "--", lw=0.5, color="red", label=f"Function draws")
        else:
            plt.plot(transformed_times, function_draw, "--", lw=0.5, color="red")

    plt.scatter(reading_times, tide_heights, s=10, marker="x", label="Training data", color="black")

    alphas = [0.4, 0.3, 0.1]
    for i in range(3):
        # Plot predictive distribution converted back to metres
        plt.fill_between(transformed_times, sample_means - (i + 1) * sigma_vector,
                         sample_means + (i + 1) * sigma_vector,
                         alpha=alphas[i], color="tab:blue", label=fr"GP Uncertainty - ${i + 1}\sigma$")

    plt.ylabel("Tide height / m", fontsize=14)
    plt.xlabel("Timestamp / days", fontsize=14)
    bottom, top = plt.ylim()
    plt.ylim(bottom=bottom - 2)  # Add extra space for legend
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    plt.plot(all_reading_times, true_tide_heights, label="Ground truth", color="green")
    plt.legend(fontsize=11, ncol=2)
    plt.tight_layout()
    plt.show()

    return gp
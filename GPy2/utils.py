import torch
import numpy as np
cpu = torch.device("cpu")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def compute_distance_matrix(x1, x2):
    """
    Computes the Mahalanobis distance matrix between data matrices

    Args:
        x1 (m x d) PyTorch tensor: Input data matrix
        x2 (n x d) PyTorch tensor: Input data matrix

    Returns:
        (m x n) PyTorch tensor: Pair-wise squared Euclidean distances
    """
    m, n, d = x1.size(0), x2.size(0), x2.size(1)
    x1 = x1.unsqueeze(1).expand(m, n, d)
    x2 = x2.unsqueeze(0).expand(m, n, d)
    return torch.pow(x1 - x2, 2).sum(2)


def expand_1d(tensor_list):
    """Expand any 1D tensors in a list of tensors to 2D"""
    expand_func = lambda x: x.view(-1, 1) if len(x.size()) <= 1 else x
    return [expand_func(tensor) for tensor in tensor_list]


def gaussian_nll(y, mu, covar_matrix, dims=1, jitter=1e-4, verbose=False):
    """Compute -ve log-likelihood of data under a multivariate Gaussian.

    Args:

    """
    y, mu = expand_1d([y, mu])
    covar_matrix += (torch.tensor(jitter) ** 2) * torch.eye(*covar_matrix.shape)
    condition_number = np.linalg.cond(covar_matrix.detach().cpu().numpy())
    if condition_number > 1e10:
        print(f"Condition number : {condition_number}")
    L_covar = torch.cholesky(covar_matrix)
    inv_covar = torch.cholesky_inverse(L_covar)
    alpha = torch.mm(inv_covar, y - mu)
    data_fit_term = -0.5 * torch.mm((y - mu).T, alpha)
    complexity_term = -torch.sum(torch.diagonal(L_covar).log())
    if verbose:
        print("Data fit : ", data_fit_term)
        print("Complexity : ", complexity_term)
    return -1 * (data_fit_term + complexity_term - (dims / 2) * np.log(2 * np.pi))

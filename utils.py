import torch

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

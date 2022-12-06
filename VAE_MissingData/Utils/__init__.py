import torch

def safe_log_sum_exp(x, dim=None, keepdim=False):
    """Numerically stable version of log_sum_exp.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to reduce.
        keepdim (bool): Whether to keep the reduced dimension or not.
    """
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    return max_x + (x - max_x).exp().sum(dim=dim, keepdim=keepdim).log()

def safe_mean_exp(x, dim=None, keepdim=False):
    """Numerically stable version of mean_exp.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to reduce.
        keepdim (bool): Whether to keep the reduced dimension or not.
    """
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    return max_x + (x - max_x).exp().mean(dim=dim, keepdim=keepdim).log()
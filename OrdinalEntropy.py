"""
Ordinal Entropy regularizer
"""
import torch
import torch.nn.functional as F
import random


def ordinal_entropy(features, gt):
    """
    Features: The last layer's features
    gt: The corresponding ground truth values
    """

    """
    sample in case the training size too large
    """
    # samples = random.sample(range(0, len(gt)-1), 100)  # random sample 100 features
    samples = random.sample(range(0, len(gt)-1), 10)  # random sample 100 features
    features = features[samples]
    gt = gt[samples]

    """
    calculate distances in the feature space, i.e. ||z_{c_i} - z_{c_j}||_2
    """
    p = F.normalize(features, dim=1)
    _distance = euclidean_dist(p, p)
    _distance = up_triu(_distance)

    """
    calculate the distances in the label space, i.e. w_{ij} = ||y_i -y_j||_2
    """
    _weight = euclidean_dist(gt, gt)
    _weight = up_triu(_weight)
    _max = torch.max(_weight)
    _min = torch.min(_weight)
    _weight = ((_weight - _min) / _max)

    """
    L_d = - mean(w_ij ||z_{c_i} - z_{c_j}||_2)
    """
    _distance = _distance * _weight
    L_d = - torch.mean(_distance)

    return L_d


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]
"""

"""
import torch
import torch.nn.functional as F


def ordinalentropy(features, gt,  mask=None):
    """
    Features: a certain layer's features
    gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
    mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
    """
    f_n, f_c, f_h, f_w = features.size()

    features = F.interpolate(features, size=[f_h // 4, f_w // 4], mode='nearest')
    features = features.permute(0, 2, 3, 1)  # n, h, w, c
    features = torch.flatten(features, start_dim=1, end_dim=2)

    gt = F.interpolate(gt, size=[f_h // 4, f_w // 4], mode='nearest')

    loss = 0

    for i in range(f_n):
        """
        mask pixels that without valid values
        """
        _gt = gt[i,:].view(-1)
        _mask = _gt > 0.001
        _mask = _mask.to(torch.bool)
        _gt = _gt[_mask]
        _features = features[i,:]
        _features = _features[_mask,:]

        """
        diverse part
        """
        u_value, u_index, u_counts = torch.unique(_gt, return_inverse=True, return_counts =True)
        center_f = torch.zeros([len(u_value), f_c]).cuda()
        center_f.index_add_(0, u_index, _features)
        u_counts = u_counts.unsqueeze(1)
        center_f = center_f / u_counts

        p = F.normalize(center_f, dim=1)
        _distance = euclidean_dist(p, p)
        _distance = up_triu(_distance)

        u_value = u_value.unsqueeze(1)
        _weight = euclidean_dist(u_value, u_value)
        _weight = up_triu(_weight)
        _max = torch.max(_weight)
        _min = torch.min(_weight)
        _weight = ((_weight - _min) / _max)

        _distance = _distance * _weight

        _entropy = torch.mean(_distance)
        loss = loss - _entropy

        """
        tightness part
        """
        _features = F.normalize(_features, dim=1)
        _features_center = p[u_index, :]
        _features = _features - _features_center
        _features = _features.pow(2)
        _tightness = torch.sum(_features, dim=1)
        _mask = _tightness > 0
        _tightness = _tightness[_mask]

        _tightness = torch.sqrt(_tightness)
        _tightness = torch.mean(_tightness)

        loss = loss + _tightness

    return loss/ f_n

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

import numpy as np
import torch


def padjust_bh(p):
    """Benjamini-Hochberg adjusted p-values"""
    n = len(p)
    i = np.arange(n,0,-1)
    o = np.argsort(p)[::-1]
    ro = np.argsort(o)
    return np.minimum(1, np.minimum.accumulate(np.float(n)/i * np.array(p)[o]))[ro]


#------------------------------------------------------------------------------------------------------
#  Covariance estimators using shrinkage, adapted for PyTorch from  sklearn.covariance:
#  https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/covariance/shrunk_covariance_.py
#------------------------------------------------------------------------------------------------------
def empirical_covariance(X_t, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator
    Parameters
    ----------
    X_t : tensor, shape (n_samples, n_features)
        Data from which to compute the covariance estimate
    assume_centered : boolean
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.
    Returns
    -------
    covariance : 2D tensor, shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).
    """
    if assume_centered:
        M_t = X_t
    else:
        M_t = X_t - X_t.mean(0, keepdim=True)
    covariance_t = torch.mm(torch.transpose(M_t, 1, 0), M_t) / M_t.shape[0]
    return covariance_t


def ledoit_wolf_shrinkage(X_t, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.
    assume_centered : bool
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.
    block_size : int
        Size of the blocks into which the covariance matrix will be split.
    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularized (shrunk) covariance is:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    """
    n_samples, n_features = X_t.shape

    # optionally center data
    if not assume_centered:
        X_t = X_t - X_t.mean(0, keepdim=True)

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2_t = X_t.pow(2)
    emp_cov_trace = X2_t.sum(0) / n_samples
    mu = emp_cov_trace.sum() / n_features
    beta_ = 0.  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            # beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            beta_ += torch.mm(torch.transpose(X2_t[:,rows], 0, 1), X2_t[:,cols]).sum()
            # delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
            delta_ += torch.mm(torch.transpose(X_t[:,rows], 0, 1), X_t[:,cols]).pow(2).sum()
        rows = slice(block_size * i, block_size * (i + 1))
        # beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits:]))
        beta_ += torch.mm(torch.transpose(X2_t[:,rows], 0, 1), X2_t[:, block_size * n_splits:]).sum()
        # delta_ += np.sum(np.dot(X.T[rows], X[:, block_size * n_splits:]) ** 2)
        delta_ += torch.mm(torch.transpose(X_t[:,rows], 0, 1), X_t[:, block_size * n_splits:]).pow(2).sum()
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        # beta_ += np.sum(np.dot(X2.T[block_size * n_splits:], X2[:, cols]))
        beta_ += torch.mm(torch.transpose(X2_t[:,block_size * n_splits:], 0, 1), X2_t[:, cols]).sum()
        # delta_ += np.sum(np.dot(X.T[block_size * n_splits:], X[:, cols]) ** 2)
        delta_ += torch.mm(torch.transpose(X_t[:,block_size * n_splits:], 0, 1), X_t[:, cols]).pow(2).sum()
    # delta_ += np.sum(np.dot(X.T[block_size * n_splits:], X[:, block_size * n_splits:]) ** 2)
    delta_ += torch.mm(torch.transpose(X_t[:,block_size * n_splits:], 0, 1), X_t[:, block_size * n_splits:]).pow(2).sum()
    delta_ /= n_samples ** 2
    # beta_ += np.sum(np.dot(X2.T[block_size * n_splits:], X2[:, block_size * n_splits:]))
    beta_ += torch.mm(torch.transpose(X2_t[:,block_size * n_splits:], 0, 1), X2_t[:, block_size * n_splits:]).sum()
    # use delta_ to compute beta
    beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2. * mu * emp_cov_trace.sum() + n_features * mu ** 2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which whould invert
    # the value of covariances
    beta = torch.min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage


def ledoit_wolf(X_t, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate
    assume_centered : boolean, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.
    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.
    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularized (shrunk) covariance is:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    """
    _, n_features = X_t.shape

    # get Ledoit-Wolf shrinkage
    shrinkage_t = ledoit_wolf_shrinkage(X_t, assume_centered=assume_centered, block_size=block_size)
    emp_cov_t = empirical_covariance(X_t, assume_centered=assume_centered)
    mu_t = torch.trace(emp_cov_t) / n_features
    shrunk_cov_t = (1. - shrinkage_t) * emp_cov_t
    shrunk_cov_t.view(-1)[::n_features + 1] += shrinkage_t * mu_t  # add to diagonal

    return shrunk_cov_t, shrinkage_t

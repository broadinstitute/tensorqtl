"""eigenmt.py: Re-implementation of eigenMT (Davis et al., AJHG, 2016)"""

__author__ = "Francois Aguet"
__copyright__ = "Copyright 2019, The Broad Institute"
__license__ = "BSD3"

import torch
import numpy as np
import pandas as pd
import time
import os
import sys
from collections import OrderedDict

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio
from core import *


def lw_shrink(X_t):
    """
    Estimates the shrunk Ledoit-Wolf covariance matrix

    Args:
      X_t: samples x variants

    Returns:
      shrunk_cov_t: shrunk covariance
      shrinkage_t:  shrinkage coefficient

    Adapted from scikit-learn:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/covariance/shrunk_covariance_.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(X_t.shape) == 2:
        n_samples, n_features = X_t.shape  # samples x variants
        X_t = X_t - X_t.mean(0)
        X2_t = X_t.pow(2)
        emp_cov_trace_sum = X2_t.sum() / n_samples
        delta_ = torch.mm(X_t.t(), X_t).pow(2).sum() / n_samples**2
        beta_ = torch.mm(X2_t.t(), X2_t).sum()
        beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
        delta = delta_ - 1. * emp_cov_trace_sum**2 / n_features
        delta /= n_features
        beta = torch.min(beta, delta)
        shrinkage_t = 0 if beta == 0 else beta / delta
        emp_cov_t = torch.mm(X_t.t(), X_t) / n_samples
        mu_t = torch.trace(emp_cov_t) / n_features
        shrunk_cov_t = (1. - shrinkage_t) * emp_cov_t
        shrunk_cov_t.view(-1)[::n_features + 1] += shrinkage_t * mu_t  # add to diagonal
    else:  # broadcast along first dimension
        n_samples, n_features = X_t.shape[1:]  # samples x variants
        X_t = X_t - X_t.mean(1, keepdim=True)
        X2_t = X_t.pow(2)
        emp_cov_trace_sum = X2_t.sum([1,2]) / n_samples
        delta_ = torch.matmul(torch.transpose(X_t, 1, 2), X_t).pow(2).sum([1,2]) / n_samples**2
        beta_ = torch.matmul(torch.transpose(X2_t, 1, 2), X2_t).sum([1,2])
        beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
        delta = delta_ - 1. * emp_cov_trace_sum**2 / n_features
        delta /= n_features
        beta = torch.min(beta, delta)
        shrinkage_t = torch.where(beta==0, torch.zeros(beta.shape).to(device), beta/delta)
        emp_cov_t = torch.matmul(torch.transpose(X_t, 1, 2), X_t) / n_samples
        mu_t = torch.diagonal(emp_cov_t, dim1=1, dim2=2).sum(1) / n_features
        shrunk_cov_t = (1 - shrinkage_t.reshape([shrinkage_t.shape[0], 1, 1])) * emp_cov_t

        ix = torch.LongTensor(np.array([np.arange(0, n_features**2, n_features+1)+i*n_features**2 for i in range(X_t.shape[0])])).to(device)
        shrunk_cov_t.view(-1)[ix] += (shrinkage_t * mu_t).unsqueeze(-1)  # add to diagonal

    return shrunk_cov_t, shrinkage_t


def find_num_eigs(eigenvalues, variance, var_thresh=0.99):
    """Returns the number of eigenvalues required to reach threshold of variance explained."""
    eigenvalues = np.sort(eigenvalues)[::-1]
    running_sum = 0
    counter = 0
    while running_sum < variance * var_thresh:
        running_sum += eigenvalues[counter]
        counter += 1
    return counter


def compute_tests(genotypes_t, var_thresh=0.99, variant_window=200):
    """determine effective number of independent variants (M_eff)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # break into windows
    windows = torch.split(genotypes_t, variant_window)

    if len(windows)>1:
        shrunk_cov_t, shrinkage_t = lw_shrink(torch.transpose(torch.stack(windows[:-1]), 1, 2))

        n_samples, n_features = windows[0].T.shape
        # indices of diagonals
        ix = torch.LongTensor(np.array([np.arange(0, n_features**2, n_features+1)+i*n_features**2 for i in range(shrunk_cov_t.shape[0])])).to(device)
        shrunk_precision_t = torch.zeros(shrunk_cov_t.shape).to(device)
        shrunk_precision_t.view(-1)[ix] = shrunk_cov_t.view(-1)[ix].pow(-0.5)
        shrunk_cor_t = torch.matmul(torch.matmul(shrunk_precision_t, shrunk_cov_t), shrunk_precision_t)
        # eigenvalues_t,_ = torch.symeig(shrunk_cor_t, eigenvectors=False)  # will be deprecated
        eigenvalues_t = torch.linalg.eigvalsh(shrunk_cor_t)  # ~2x slower than symeig with 1.10.0+cu102 and 2.0.1+cu118

    # last window
    shrunk_cov0_t, shrinkage0_t = lw_shrink(windows[-1].t())
    shrunk_precision0_t = torch.diag(torch.diag(shrunk_cov0_t).pow(-0.5))
    shrunk_cor0_t = torch.mm(torch.mm(shrunk_precision0_t, shrunk_cov0_t), shrunk_precision0_t)
    # eigenvalues0_t,_ = torch.symeig(shrunk_cor0_t, eigenvectors=False)
    eigenvalues0_t = torch.linalg.eigvalsh(shrunk_cor0_t)

    if len(windows) > 1:
        eigenvalues = list(eigenvalues_t.cpu().numpy())
        eigenvalues.append(eigenvalues0_t.cpu().numpy())
    else:
        eigenvalues = [eigenvalues0_t.cpu().numpy()]

    m_eff = 0
    for ev,m in zip(eigenvalues, [i.shape[0] for i in windows]):
        ev[ev < 0] = 0
        m_eff += find_num_eigs(ev, m, var_thresh=var_thresh)

    return m_eff



def run_eigenmt(genotype_df, variant_df, phenotype_df, phenotype_pos_df, interaction_s=None,
                maf_threshold=0, var_thresh=0.99, variant_window=200, window=1000000, verbose=True, logger=None):
    """Standalone function for computing eigenMT correction.

    Returns the number of tests for each gene
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('eigenMT: estimating number of independent variants tested for each phenotype')

    logger.write('cis-QTL mapping: empirical p-values for phenotypes')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    logger.write(f'  * {genotype_df.shape[0]} variants')

    if interaction_s is not None and maf_threshold > 0:
        interaction_mask_t = torch.BoolTensor(interaction_s >= interaction_s.median()).to(device)
    else:
        interaction_mask_t = None

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    start_time = time.time()
    m_eff = OrderedDict()
    for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):

        # copy genotypes to GPU
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        genotypes_t = genotypes_t[:,genotype_ix_t]
        impute_mean(genotypes_t)

        if interaction_s is None:
            mask_t = calculate_maf(genotypes_t) >= maf_threshold
            genotypes_t = genotypes_t[mask_t]
        else:
            genotypes_t, mask_t = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t, maf_threshold_interaction=maf_threshold)

        m_eff[phenotype_id] = compute_tests(genotypes_t, var_thresh=var_thresh, variant_window=variant_window)

    logger.write(f'    time elapsed: {(time.time()-start_time)/60:.2f} min')
    return pd.Series(m_eff)


def padjust_bh(p):
    """Benjamini-Hochberg adjusted p-values"""
    if not np.all(np.isfinite(p)):
        raise ValueError('P values must be finite.')
    n = len(p)
    i = np.arange(n,0,-1)
    o = np.argsort(p)[::-1]
    ro = np.argsort(o)
    return np.minimum(1, np.minimum.accumulate(np.float64(n)/i * np.array(p)[o]))[ro]

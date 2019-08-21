
"""eigenmt.py: Re-implementation of eigenMT (Davis et al., AJHG, 2016)"""

import torch
# from torch.utils.dlpack import to_dlpack, from_dlpack
# import cupy as cp
import numpy as np
import sys
import os

sys.path.insert(1, os.path.dirname(__file__))
import statsfunc


def lw_shrink(genotypes_t):
    """Compute smoothened genotype correlation matrix using Ledoit-Wolf shrinkage"""
    m, n = genotypes_t.shape  # variants x samples
    try:
        shrunk_cov_t, alpha_t = statsfunc.ledoit_wolf(torch.transpose(genotypes_t, 0, 1), assume_centered=False, block_size=1000)
        shrunk_precision_t = torch.diag(torch.diag(shrunk_cov_t).pow(-0.5))
        shrunk_cor_t = torch.mm(torch.mm(shrunk_precision_t, shrunk_cov_t), shrunk_precision_t)
    except:  # Exception for case where variants are all in perfect LD
        shrunk_cor_t = torch.ones(m,m)
        alpha_t = 'NA'
    return shrunk_cor_t, alpha_t


def find_num_eigs(eigenvalues, variance, var_thresh=0.99):
    """Returns the number of eigenvalues required to reach threshold of variance explained."""
    eigenvalues = np.sort(eigenvalues)[::-1]
    running_sum = 0
    counter = 0
    while running_sum < variance * var_thresh:
        running_sum += eigenvalues[counter]
        counter += 1
    return counter


def fit(genotypes_t, var_thresh=0.99, window=200):
    """
    """
    m_eff = 0
    # process genotypes in chunks of 'window' variants
    n_chunks = int(np.ceil(genotypes_t.shape[0] / window))
    for i in range(n_chunks):
        window_t = genotypes_t[i*window:(i+1)*window, :]
        m, _ = window_t.shape  # number of variants
        gen_corr_t, _ = lw_shrink(window_t)
        eigenvalues_t, _ = torch.symeig(gen_corr_t, eigenvectors=False)
        eigenvalues = eigenvalues_t.cpu().numpy()
        # eigenvalues = cp.asnumpy(cp.linalg.eigvalsh(cp.fromDlpack(to_dlpack(gen_corr_t))))
        eigenvalues[eigenvalues < 0] = 0
        m_eff += find_num_eigs(eigenvalues, m, var_thresh=var_thresh)
    return m_eff

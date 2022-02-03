import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(1, os.path.dirname(__file__))
import cis
from core import *


def trc(genotypes_t, counts_t, covariates_t=None, select_covariates=True,
        count_threshold=0, imputation='offset', return_af=False):
    """
    genotypes_t: dosages (variants x samples)
    counts_t: DESeq size factor-normalized read counts
    covariates_t: first column must be intercept
    """
    nonzero_t = counts_t != 0

    if imputation == 'offset':
        log_counts_t = counts_t.log1p()
    elif imputation == 'half_min':
        log_counts_t = counts_t.clone()
        log_counts_t[~nonzero_t] = log_counts_t[nonzero_t].min() / 2
        log_counts_t = log_counts_t.log()

    if covariates_t is not None:
        if select_covariates:
            # select significant covariates
            b_t, b_se_t = linreg(covariates_t[nonzero_t, :], log_counts_t[nonzero_t], dtype=torch.float32)
            tstat_t = b_t / b_se_t
            m = tstat_t.abs() > 2
            m[0] = True  # keep intercept
            sel_covariates_t = covariates_t[:, m]
        else:
            sel_covariates_t = covariates_t

        # Regress out covariates from non-zero counts, and keep zeros.
        # This follows the original mixQTL implementation, but may be
        # problematic when count_threshold is 0.
        residualizer = Residualizer(sel_covariates_t[nonzero_t, 1:])  # exclude intercept
        y_t = counts_t.clone()
        y_t[nonzero_t] = residualizer.transform(log_counts_t[nonzero_t].reshape(1,-1), center=True)
    else:
        y_t = log_counts_t

    m_t = counts_t >= count_threshold
    res = cis.calculate_cis_nominal(genotypes_t[:, m_t] / 2, y_t[m_t], return_af=False)

    if return_af:
        af, ma_samples, ma_counts = get_allele_stats(genotypes_t)
        return *res, af, ma_samples, ma_counts
    else:
        return res

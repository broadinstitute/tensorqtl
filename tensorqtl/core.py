import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize
from scipy.special import loggamma
import sys


class SimpleLogger(object):
    def __init__(self, logfile=None, verbose=True):
        self.console = sys.stdout
        self.verbose = verbose
        if logfile is not None:
            self.log = open(logfile, 'w')
        else:
            self.log = None

    def write(self, message):
        if self.verbose:
            self.console.write(message+'\n')
        if self.log is not None:
            self.log.write(message+'\n')
            self.log.flush()

#------------------------------------------------------------------------------
#  Core classes/functions for mapping associations on GPU
#------------------------------------------------------------------------------
class Residualizer(object):
    def __init__(self, C_t):
        # center and orthogonalize
        self.Q_t, _ = torch.qr(C_t - C_t.mean(0))

    def transform(self, M_t, center=True):
        """Residualize rows of M wrt columns of C"""
        if center:
            M0_t = M_t - M_t.mean(1, keepdim=True)
        else:
            M0_t = M_t
        return M_t - torch.mm(torch.mm(M0_t, self.Q_t), torch.transpose(self.Q_t, 0, 1))  # keep original mean


def calculate_maf(genotype_t, alleles=2):
    """Calculate minor allele frequency"""
    af_t = genotype_t.sum(1) / (alleles * genotype_t.shape[1])
    return torch.where(af_t > 0.5, 1 - af_t, af_t)


def filter_maf(genotypes_t, variant_ids, maf_threshold):
    """Calculate MAF and filter genotypes that don't pass threshold"""
    maf_t = calculate_maf(genotypes_t)
    mask_t = maf_t >= maf_threshold
    genotypes_t = genotypes_t[mask_t]
    variant_ids = variant_ids[mask_t.cpu().numpy().astype(bool)]
    maf_t = maf_t[mask_t]
    return genotypes_t, variant_ids, maf_t


def center_normalize(M_t, dim=0):
    """Center and normalize M"""
    N_t = M_t - M_t.mean(dim=dim, keepdim=True)
    return N_t / torch.sqrt(torch.pow(N_t, 2).sum(dim=dim, keepdim=True))


def calculate_corr(genotype_t, phenotype_t, residualizer, return_sd=False):
    """Calculate correlation between normalized residual genotypes and phenotypes"""
    # residualize
    genotype_res_t = residualizer.transform(genotype_t)  # variants x samples
    phenotype_res_t = residualizer.transform(phenotype_t)  # phenotypes x samples

    if return_sd:
        gstd = genotype_res_t.var(1)
        pstd = phenotype_res_t.var(1)

    # center and normalize
    genotype_res_t = center_normalize(genotype_res_t, dim=1)
    phenotype_res_t = center_normalize(phenotype_res_t, dim=1)

    # correlation
    if return_sd:
        return torch.mm(genotype_res_t, torch.transpose(phenotype_res_t, 0, 1)), torch.sqrt(pstd / gstd)
    else:
        return torch.mm(genotype_res_t, torch.transpose(phenotype_res_t, 0, 1))

#------------------------------------------------------------------------------
#  Functions for beta-approximating empirical p-values
#------------------------------------------------------------------------------
def pval_from_corr(r2, dof):
    tstat2 = dof * r2 / (1 - r2)
    return 2*stats.t.cdf(-np.abs(np.sqrt(tstat2)), dof)


def df_cost(r2, dof):
    """minimize abs(1-alpha) as a function of M_eff"""
    pval = pval_from_corr(r2, dof)
    mean = np.mean(pval)
    var = np.var(pval)
    return mean * (mean * (1.0-mean) / var - 1.0) - 1.0


def beta_log_likelihood(x, shape1, shape2):
    """negative log-likelihood of beta distribution"""
    logbeta = loggamma(shape1) + loggamma(shape2) - loggamma(shape1+shape2)
    return (1.0-shape1)*np.sum(np.log(x)) + (1.0-shape2)*np.sum(np.log(1.0-x)) + len(x)*logbeta


def fit_beta_parameters(r2_perm, dof, tol=1e-4, return_minp=False):
    """
      r2_perm:    array of max. r2 values from permutations
      dof:        degrees of freedom
    """
    try:
        true_dof = scipy.optimize.newton(lambda x: df_cost(r2_perm, x), dof, tol=tol, maxiter=50)
    except:
        print('WARNING: scipy.optimize.newton failed to converge (running scipy.optimize.minimize)')
        res = scipy.optimize.minimize(lambda x: np.abs(df_cost(r2_perm, x)), dof, method='Nelder-Mead', tol=tol)
        true_dof = res.x[0]

    pval = pval_from_corr(r2_perm, true_dof)
    mean, var = np.mean(pval), np.var(pval)
    beta_shape1 = mean * (mean * (1 - mean) / var - 1)
    beta_shape2 = beta_shape1 * (1/mean - 1)
    res = scipy.optimize.minimize(lambda s: beta_log_likelihood(pval, s[0], s[1]), [beta_shape1, beta_shape2], method='Nelder-Mead', tol=tol)
    beta_shape1, beta_shape2 = res.x
    if return_minp:
        return beta_shape1, beta_shape2, true_dof, pval
    else:
        return beta_shape1, beta_shape2, true_dof


def calculate_beta_approx_pval(r2_perm, r2_nominal, dof, tol=1e-4):
    """
      r2_nominal: nominal max. r2 (scalar or array)
      r2_perm:    array of max. r2 values from permutations
      dof:        degrees of freedom
    """
    beta_shape1, beta_shape2, true_dof = fit_beta_parameters(r2_perm, dof, tol)
    pval_true_dof = pval_from_corr(r2_nominal, true_dof)
    pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)
    return pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof

#------------------------------------------------------------------------------
#  i/o functions
#------------------------------------------------------------------------------
def read_phenotype_bed(phenotype_bed):
    """Load phenotype BED file as phenotype and TSS DataFrames"""
    if phenotype_bed.endswith('.bed.gz'):
        phenotype_df = pd.read_csv(phenotype_bed, sep='\t', index_col=3, dtype={'#chr':str, '#Chr':str})
    elif phenotype_bed.endswith('.parquet'):
        phenotype_df = pd.read_parquet(phenotype_bed)
    else:
        raise ValueError('Unsupported file type.')
    phenotype_df = phenotype_df.rename(columns={i:i.lower() for i in phenotype_df.columns[:3]})
    phenotype_pos_df = phenotype_df[['#chr', 'end']].rename(columns={'#chr':'chr', 'end':'tss'})
    phenotype_df = phenotype_df.drop(['#chr', 'start', 'end'], axis=1)
    return phenotype_df, phenotype_pos_df

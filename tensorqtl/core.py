from calendar import c
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize
from scipy.special import loggamma
import sys
import re


output_dtype_dict = {
    'num_var':np.int32,
    'beta_shape1':np.float32,
    'beta_shape2':np.float32,
    'true_df':np.float32,
    'pval_true_df':np.float64,
    'variant_id':str,
    'tss_distance':np.int32,
    'ma_samples':np.int32,
    'ma_count':np.int32,
    'af':np.float32,
    'pval_nominal':np.float64,
    'slope':np.float32,
    'slope_se':np.float32,
    'pval_perm':np.float64,
    'pval_beta':np.float64,
}


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
        self.Q_t, _ = torch.linalg.qr(C_t - C_t.mean(0))
        self.dof = C_t.shape[0] - 2 - C_t.shape[1]
        self.n = C_t.shape[1] + 2

    def transform(self, M_t, center=True):
        """Residualize rows of M wrt columns of C"""
        M0_t = M_t - M_t.mean(1, keepdim=True)
        if center:
            M0_t = M0_t - torch.mm(torch.mm(M0_t, self.Q_t), self.Q_t.t())
        else:
            M0_t = M_t - torch.mm(torch.mm(M0_t, self.Q_t), self.Q_t.t())
        return M0_t


def calculate_maf(genotype_t, alleles=2):
    """Calculate minor allele frequency"""
    af_t = genotype_t.sum(1) / (alleles * genotype_t.shape[1])
    return torch.where(af_t > 0.5, 1 - af_t, af_t)


def get_allele_stats(genotype_t):
    """Returns allele frequency, minor allele samples, and minor allele counts (row-wise)."""
    # allele frequency
    n2 = 2 * genotype_t.shape[1]
    af_t = genotype_t.sum(1) / n2
    # minor allele samples and counts
    ix_t = af_t <= 0.5
    m = genotype_t > 0.5
    a = m.sum(1).int()
    b = (genotype_t < 1.5).sum(1).int()
    ma_samples_t = torch.where(ix_t, a, b)
    a = (genotype_t * m.float()).sum(1).int()
    # a = (genotype_t * m.float()).sum(1).round().int()  # round for missing/imputed genotypes
    ma_count_t = torch.where(ix_t, a, n2-a)
    return af_t, ma_samples_t, ma_count_t


def filter_maf(genotypes_t, variant_ids, maf_threshold, alleles=2):
    """Calculate MAF and filter genotypes that don't pass threshold"""
    af_t = genotypes_t.sum(1) / (alleles * genotypes_t.shape[1])
    maf_t = torch.where(af_t > 0.5, 1 - af_t, af_t)
    if maf_threshold > 0:
        mask_t = maf_t >= maf_threshold
        genotypes_t = genotypes_t[mask_t]
        variant_ids = variant_ids[mask_t.cpu().numpy().astype(bool)]
        af_t = af_t[mask_t]
    return genotypes_t, variant_ids, af_t


def filter_maf_interaction(genotypes_t, interaction_mask_t=None, maf_threshold_interaction=0.05):
    # filter monomorphic sites (to avoid colinearity)
    mask_t = ~((genotypes_t==0).all(1) | (genotypes_t==1).all(1) | (genotypes_t==2).all(1))
    if interaction_mask_t is not None:
        if interaction_mask_t.dim == 1:
            upper_t = calculate_maf(genotypes_t[:, interaction_mask_t]) >= maf_threshold_interaction - 1e-7
            lower_t = calculate_maf(genotypes_t[:,~interaction_mask_t]) >= maf_threshold_interaction - 1e-7
            mask_t = mask_t & upper_t & lower_t

    genotypes_t = genotypes_t[mask_t]
    return genotypes_t, mask_t

def filter_term_samples(genotypes_t, term_mask_t, num_samples_per_gt=3, device='cpu'):

    # filter monomorphic sites (to avoid colinearity)
    mask_t = ~((genotypes_t==0).all(1) | (genotypes_t==1).all(1) | (genotypes_t==2).all(1))

    # terms matrix: ng x n interactions
    terms_mask_t = torch.zeros(genotypes_t.shape[0], term_mask_t.shape[1], dtype=torch.bool).to(device)

    if term_mask_t.dim == 1:
        term_mask_t = term_mask_t.unsqueeze(-1)

    # Determine which terms have at least 'num_samples_per_gt' samples for each genotype
    for i in range(term_mask_t.shape[1]):
        rr = ((genotypes_t[:, term_mask_t[:,i]] == 0).sum(1) >= num_samples_per_gt)
        ra = ((genotypes_t[:, term_mask_t[:,i]] == 1).sum(1) >= num_samples_per_gt)
        aa = ((genotypes_t[:, term_mask_t[:,i]] == 2).sum(1) >= num_samples_per_gt)
        terms_mask_t[:,i] = rr & ra & aa

    # Filter out monomorphic sites and those that do not have enough data per term
    mask_t = mask_t & (terms_mask_t.sum(1) == term_mask_t.shape[1])

    genotypes_t = genotypes_t[mask_t]
    return genotypes_t, mask_t

def impute_mean(genotypes_t):
    """Impute missing genotypes to mean"""
    m = genotypes_t == -1
    ix = torch.nonzero(m, as_tuple=True)[0]
    if len(ix) > 0:
        a = genotypes_t.sum(1)
        b = m.sum(1).float()
        mu = (a + b) / (genotypes_t.shape[1] - b)
        genotypes_t[m] = mu[ix]


def center_normalize(M_t, dim=0):
    """Center and normalize M"""
    N_t = M_t - M_t.mean(dim=dim, keepdim=True)
    return N_t / torch.sqrt(torch.pow(N_t, 2).sum(dim=dim, keepdim=True))


def calculate_corr(genotype_t, phenotype_t, residualizer=None, return_var=False):
    """Calculate correlation between normalized residual genotypes and phenotypes"""

    # residualize
    if residualizer is not None:
        genotype_res_t = residualizer.transform(genotype_t)  # variants x samples
        phenotype_res_t = residualizer.transform(phenotype_t)  # phenotypes x samples
    else:
        genotype_res_t = genotype_t
        phenotype_res_t = phenotype_t

    if return_var:
        genotype_var_t = genotype_res_t.var(1)
        phenotype_var_t = phenotype_res_t.var(1)

    # center and normalize
    genotype_res_t = center_normalize(genotype_res_t, dim=1)
    phenotype_res_t = center_normalize(phenotype_res_t, dim=1)

    # correlation
    if return_var:
        return torch.mm(genotype_res_t, phenotype_res_t.t()), genotype_var_t, phenotype_var_t
    else:
        return torch.mm(genotype_res_t, phenotype_res_t.t())



def calculate_interaction_nominal(genotypes_t, phenotypes_t, design_t, 
                                  g_interaction_terms_t,
                                  residualizer=None, center=False, variant_ids=None):
    """
    Solve y ~ g + i + g:i, where i is an interaction vector or matrix

    Inputs
      genotypes_t:   [num_genotypes x num_samples]
      phenotypes_t:  [num_phenotypes x num_samples]
      design_t: [num_samples x interactions] <- dummy matrix
      g_idx_t: position of genotype column in design matrix
      g_interaction_terms_t: positions of terms with a genetic component
      terms_to_residualize_t: position of terms to center and residualize

    Outputs
    tstat_t, b_t, b_se_t, af_t, ma_samples_t, ma_count_t
    tstat_t, b_t, b_se_t columns: [g, i_1 ... i_n, gi_1, ... gi_n]
                                where n is the number of interactions from 
                                design matrix
    """

    ng, ns = genotypes_t.shape

    X = torch.clone(design_t).unsqueeze(0).repeat([ng, 1, 1])

    # apply genotypes to terms w/ genotype component
    # if g_idx_t > 0:
    #     X[..., g_idx_t] = genotypes_t
    X[..., g_interaction_terms_t] *= genotypes_t.unsqueeze(-1)

    nterms = X.shape[2]

    # center and residualize non-categorical variables and those with genotype component
    # for i in terms_to_residualize_t:
    for i in range(nterms):
        if residualizer is not None:
            X[..., i] = residualizer.transform(X[..., i], center=center)


    Y = phenotypes_t if phenotypes_t.dim() == 2 else phenotypes_t.unsqueeze(0)
    nps = Y.shape[0]
    
    # center and residualize phenotypes matrix
    if residualizer is not None:
        Y = residualizer.transform(Y, center=center)

    #Y = Y.unsqueeze(0).expand([ng, *Y.shape])  # ng x np x ns
    Y = Y.expand([ng, *Y.shape])  # ng x np x ns
    Y = torch.transpose(Y, 1, 2)

    dfe = ns - nterms
    dfm = nterms - 1
    if residualizer is not None:
        dfe -= residualizer.n
        dfm += residualizer.n

    # matrix regession
    XtX = torch.matmul(torch.transpose(X, 1, 2), X)
    XtY = torch.matmul(torch.transpose(X, 1, 2), Y)

    try:
        XtXinv = XtX.inverse()
    except Exception as e:
        if variant_ids is not None and len(e.args) >= 1:
            i = int(re.findall('Batch element (\d+)', str(e))[0])
            e.args = (e.args[0] + f'\n    Likely problematic variant: {variant_ids[i]} ',) + e.args[1:]
        raise

    b_t = torch.matmul(XtXinv, XtY)

    hat_matrix_t = torch.matmul(torch.matmul(X, XtXinv), torch.transpose(X, 1, 2))
    predicted_t = torch.matmul(hat_matrix_t, Y)
    
    resids_t = Y - predicted_t
    sse_t = torch.pow(resids_t, 2).sum(1)
    mse_t = sse_t / dfe

    resids_t = predicted_t - Y.mean(1, keepdims=True)
    ssr_t = torch.pow(resids_t, 2).sum(1)
    msr_t = ssr_t / dfm

    sst_t = sse_t + ssr_t

    b_se_t = torch.sqrt(XtXinv[:, torch.eye(nterms, dtype=torch.bool)].unsqueeze(-1).repeat([1, 1, nps]) * mse_t.unsqueeze(1).repeat([1,nterms,1]))
    tstat_t = b_t/b_se_t

    f_t = (msr_t/mse_t)
    r2_t = (ssr_t/sst_t)

    af_t, ma_samples_t, ma_count_t = get_allele_stats(genotypes_t) # allele freqs are wrong because we have same inidviduals for interaction studies
    
    return sse_t, f_t, r2_t, tstat_t, b_t, b_se_t, af_t, ma_samples_t, ma_count_t

def linreg(X_t, y_t, dtype=torch.float64):
    """
    Robust linear regression. Solves y = Xb, standardizing X.
    The first column of X must be the intercept.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_std_t = X_t.std(0)
    x_mean_t = X_t.mean(0)
    x_std_t[0] = 1
    x_mean_t[0] = 0

    # standardize X
    Xtilde_t = (X_t - x_mean_t) / x_std_t

    # regression
    XtX_t = torch.matmul(Xtilde_t.T, Xtilde_t)
    Xty_t = torch.matmul(Xtilde_t.T, y_t)
    b_t, _ = torch.solve(Xty_t.unsqueeze(-1), XtX_t)
    b_t = b_t.squeeze()

    # compute s.e.
    dof = X_t.shape[0] - X_t.shape[1]
    r_t = y_t - torch.matmul(Xtilde_t, b_t)
    sigma2_t = (r_t*r_t).sum() / dof
    XtX_inv_t, _ = torch.solve(torch.eye(X_t.shape[1], dtype=dtype).to(device), XtX_t)
    var_b_t = sigma2_t * XtX_inv_t
    b_se_t = torch.sqrt(torch.diag(var_b_t))

    # rescale
    b_t /= x_std_t
    b_se_t /= x_std_t

    # adjust intercept
    b_t[0] -= torch.sum(x_mean_t * b_t)
    ms_t = x_mean_t / x_std_t
    b_se_t[0] = torch.sqrt(b_se_t[0]**2 + torch.matmul(torch.matmul(ms_t.T, var_b_t), ms_t))

    return b_t, b_se_t


def filter_covariates(covariates_t, log_counts_t, tstat_threshold=2):
    """
    Inputs:
      covariates0_t: covariates matrix (samples x covariates)
                     including genotype PCs, PEER factors, etc.
                     ** with intercept in first column **
      log_counts_t:  counts vector (samples)
    """
    assert (covariates_t[:,0] == 0).all()
    b_t, b_se_t = linreg(covariates_t, log_counts_t)
    tstat_t = b_t / b_se_t
    m = tstat_t.abs() > tstat_threshold
    m[0] = False
    return covariates_t[:, m]


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


def fit_beta_parameters(r2_perm, dof_init, tol=1e-4, return_minp=False):
    """
      r2_perm:    array of max. r2 values from permutations
      dof_init:   degrees of freedom
    """
    try:
        true_dof = scipy.optimize.newton(lambda x: df_cost(r2_perm, x), dof_init, tol=tol, maxiter=50)
    except:
        print('WARNING: scipy.optimize.newton failed to converge (running scipy.optimize.minimize)')
        res = scipy.optimize.minimize(lambda x: np.abs(df_cost(r2_perm, x)), dof_init, method='Nelder-Mead', tol=tol)
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


def calculate_beta_approx_pval(r2_perm, r2_nominal, dof_init, tol=1e-4):
    """
      r2_nominal: nominal max. r2 (scalar or array)
      r2_perm:    array of max. r2 values from permutations
      dof_init:   degrees of freedom
    """
    beta_shape1, beta_shape2, true_dof = fit_beta_parameters(r2_perm, dof_init, tol)
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
        phenotype_df.set_index(phenotype_df.columns[3], inplace=True)
    else:
        raise ValueError('Unsupported file type.')
    phenotype_df.rename(columns={i:i.lower().replace('#chr','chr') for i in phenotype_df.columns[:3]}, inplace=True)
    # make sure TSS/cis-window is properly defined
    if not (phenotype_df['start']+1 == phenotype_df['end']).all():
        raise ValueError("The BED file must define the TSS/cis-window center, with start+1 == end.")
    phenotype_pos_df = phenotype_df[['chr', 'end']].rename(columns={'end':'tss'})
    phenotype_df.drop(['chr', 'start', 'end'], axis=1, inplace=True)
    return phenotype_df, phenotype_pos_df

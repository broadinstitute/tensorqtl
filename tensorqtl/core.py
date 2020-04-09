import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize
from scipy.special import loggamma
import sys


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
    'maf':np.float32,
    'ref_factor':np.int32,
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
        self.Q_t, _ = torch.qr(C_t - C_t.mean(0))
        self.dof = C_t.shape[0] - 2 - C_t.shape[1]

    def transform(self, M_t, center=True):
        """Residualize rows of M wrt columns of C"""
        if center:
            M0_t = M_t - M_t.mean(1, keepdim=True)
        else:
            M0_t = M_t
        return M_t - torch.mm(torch.mm(M0_t, self.Q_t), self.Q_t.t())  # keep original mean


def calculate_maf(genotype_t, alleles=2):
    """Calculate minor allele frequency"""
    af_t = genotype_t.sum(1) / (alleles * genotype_t.shape[1])
    return torch.where(af_t > 0.5, 1 - af_t, af_t)


def filter_maf(genotypes_t, variant_ids, maf_threshold):
    """Calculate MAF and filter genotypes that don't pass threshold"""
    maf_t = calculate_maf(genotypes_t)
    if maf_threshold > 0:
        mask_t = maf_t >= maf_threshold
        genotypes_t = genotypes_t[mask_t]
        variant_ids = variant_ids[mask_t.cpu().numpy().astype(bool)]
        maf_t = maf_t[mask_t]
    return genotypes_t, variant_ids, maf_t


def filter_maf_interaction(genotypes_t, interaction_mask_t=None, maf_threshold_interaction=0.05):
    # filter monomorphic sites (to avoid colinearity)
    mask_t = ~((genotypes_t==0).all(1) | (genotypes_t==1).all(1) | (genotypes_t==2).all(1))
    if interaction_mask_t is not None:
        upper_t = calculate_maf(genotypes_t[:, interaction_mask_t]) >= maf_threshold_interaction - 1e-7
        lower_t = calculate_maf(genotypes_t[:,~interaction_mask_t]) >= maf_threshold_interaction - 1e-7
        mask_t = mask_t & upper_t & lower_t
    genotypes_t = genotypes_t[mask_t]
    return genotypes_t, mask_t


def impute_mean(genotypes_t):
    """Impute missing genotypes to mean"""
    m = genotypes_t == -1
    a = genotypes_t.sum(1)
    b = m.sum(1).float()
    mu = (a + b) / (genotypes_t.shape[1] - b)
    ix = m.nonzero()
    genotypes_t[m] = mu[ix[:,0]]


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
        return torch.mm(genotype_res_t, phenotype_res_t.t()), torch.sqrt(pstd.reshape(1,-1) / gstd.reshape(-1,1))
    else:
        return torch.mm(genotype_res_t, phenotype_res_t.t())


def calculate_interaction_nominal(genotypes_t, phenotypes_t, interaction_t, residualizer,
                                  return_sparse=False, tstat_threshold=None):
    """
    genotypes_t:   [num_genotypes x num_samples]
    phenotypes_t:   [num_phenotypes x num_samples]
    interaction_t: [1 x num_samples]
    """
    ng, ns = genotypes_t.shape
    nps = phenotypes_t.shape[0]

    # centered inputs
    g0_t = genotypes_t - genotypes_t.mean(1, keepdim=True)
    gi_t = genotypes_t * interaction_t
    gi0_t = gi_t - gi_t.mean(1, keepdim=True)
    i0_t = interaction_t - interaction_t.mean()
    p0_t = phenotypes_t - phenotypes_t.mean(1, keepdim=True)

    # residualize rows
    g0_t = residualizer.transform(g0_t, center=False)
    gi0_t = residualizer.transform(gi0_t, center=False)
    p0_t = residualizer.transform(p0_t, center=False)
    i0_t = residualizer.transform(i0_t, center=False)
    i0_t = i0_t.repeat(ng, 1)

    # regression (in float; loss of precision may occur in edge cases)
    X_t = torch.stack([g0_t, i0_t, gi0_t], 2)  # ng x ns x 3
    Xinv = torch.matmul(torch.transpose(X_t, 1, 2), X_t).inverse() # ng x 3 x 3
    # Xinv = tf.linalg.inv(tf.matmul(X_t, X_t, transpose_a=True))  # ng x 3 x 3
    # p0_tile_t = tf.tile(tf.expand_dims(p0_t, 0), [ng,1,1])  # ng x np x ns
    p0_tile_t = p0_t.unsqueeze(0).expand([ng, *p0_t.shape])  # ng x np x ns

    # calculate b, b_se
    # [(ng x 3 x 3) x (ng x 3 x ns)] x (ng x ns x np) = (ng x 3 x np)
    b_t = torch.matmul(torch.matmul(Xinv, torch.transpose(X_t, 1, 2)), torch.transpose(p0_tile_t, 1, 2))
    dof = residualizer.dof - 2
    if nps==1:
        r_t = torch.matmul(X_t, b_t).squeeze() - p0_t
        rss_t = (r_t*r_t).sum(1)
        b_se_t = torch.sqrt(Xinv[:, torch.eye(3, dtype=torch.uint8).bool()] * rss_t.unsqueeze(1) / dof)
        b_t = b_t.squeeze(2)
        # r_t = tf.squeeze(tf.matmul(X_t, b_t)) - p0_t  # (ng x ns x 3) x (ng x 3 x 1)
        # rss_t = tf.reduce_sum(tf.multiply(r_t, r_t), axis=1)
        # b_se_t = tf.sqrt( tf.matrix_diag_part(Xinv) * tf.expand_dims(rss_t, 1) / dof )
    else:
        # b_t = tf.matmul(p0_tile_t, tf.matmul(Xinv, X_t, transpose_b=True), transpose_b=True)
        # convert to ng x np x 3??
        r_t = torch.matmul(X_t, b_t) - torch.transpose(p0_tile_t, 1, 2)  # (ng x ns x np)
        rss_t = (r_t*r_t).sum(1)  # ng x np
        b_se_t = torch.sqrt(Xinv[:, torch.eye(3, dtype=torch.uint8).bool()].unsqueeze(-1).repeat([1,1,nps]) * rss_t.unsqueeze(1).repeat([1,3,1]) / dof)
        # b_se_t = tf.sqrt(tf.tile(tf.expand_dims(tf.matrix_diag_part(Xinv), 2), [1,1,nps]) * tf.tile(tf.expand_dims(rss_t, 1), [1,3,1]) / dof) # (ng x 3) -> (ng x 3 x np)

    tstat_t = (b_t.double() / b_se_t.double()).float()  # (ng x 3 x np)

    # calculate MAF
    n2 = 2*ns
    af_t = genotypes_t.sum(1) / n2
    ix_t = af_t <= 0.5
    maf_t = torch.where(ix_t, af_t, 1 - af_t)
    # tdist = tfp.distributions.StudentT(np.float64(dof), loc=np.float64(0.0), scale=np.float64(1.0))
    if not return_sparse:
        # calculate pval
        # pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat_t)))  # (ng x 3 x np)

        # calculate MA samples and counts
        m = genotypes_t > 0.5
        a = m.sum(1).int()
        b = (genotypes_t < 1.5).sum(1).int()
        ma_samples_t = torch.where(ix_t, a, b)
        a = (genotypes_t * m.float()).sum(1).round().int()  # round for missing/imputed genotypes
        ma_count_t = torch.where(ix_t, a, n2-a)
        return tstat_t, b_t, b_se_t, maf_t, ma_samples_t, ma_count_t

    else:  # sparse output
        tstat_g_t =  tstat_t[:,0,:]  # genotypes x phenotypes
        tstat_i_t =  tstat_t[:,1,:]
        tstat_gi_t = tstat_t[:,2,:]
        m = tstat_gi_t.abs() >= tstat_threshold
        tstat_g_t = tstat_g_t[m]
        tstat_i_t = tstat_i_t[m]
        tstat_gi_t = tstat_gi_t[m]
        ix = m.nonzero()  # indexes: [genotype, phenotype]
        return tstat_g_t, tstat_i_t, tstat_gi_t, maf_t[ix[:,0]], ix

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
    phenotype_pos_df = phenotype_df[['chr', 'end']].rename(columns={'end':'tss'})
    phenotype_df.drop(['chr', 'start', 'end'], axis=1, inplace=True)
    return phenotype_df, phenotype_pos_df

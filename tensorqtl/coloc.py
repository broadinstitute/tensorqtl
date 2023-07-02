import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import os
import time
import sys
sys.path.insert(1, os.path.dirname(__file__))
import genotypeio, eigenmt
from core import *


def logsumexp(x, dim=0):
    mmax,_ = torch.max(x, dim=dim, keepdim=True)
    return mmax + (x-mmax).exp().sum(dim, keepdim=True).log()


def logdiff(x, y, dim=0):
    xmax,_ = torch.max(x, dim=dim, keepdim=True)
    ymax,_ = torch.max(y, dim=dim, keepdim=True)
    mmax = torch.max(xmax, ymax)
    return mmax + ((x - mmax).exp() - (y - mmax).exp()).log()


def coloc(genotypes1_t, genotypes2_t, phenotype1_t, phenotype2_t,
          residualizer1=None, residualizer2=None, mode='beta',
          p1=1e-4, p2=1e-4, p12=1e-5):
    """COLOC from summary statistics (either beta/sds or p-values and MAF)"""

    assert phenotype1_t.dim() == 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # phenotype 1
    if mode == 'beta':
        r_nominal_t, genotype_var_t, phenotype_var_t = calculate_corr(
            genotypes1_t, phenotype1_t.reshape(1,-1), residualizer1, return_var=True)
        r_nominal_t = r_nominal_t.squeeze()
        var_ratio_t = phenotype_var_t.reshape(1,-1) / genotype_var_t.reshape(-1,1)
    else:
        r_nominal_t = calculate_corr(
            genotypes1_t, phenotype1_t.reshape(1,-1), residualizer1, return_var=False).squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    if residualizer1 is not None:
        dof = residualizer1.dof
    else:
        dof = phenotype1_t.shape[0] - 2

    if mode == 'beta':
        tstat2_t = r2_nominal_t * dof / (1 - r2_nominal_t)
        beta2_t = r2_nominal_t * var_ratio_t.squeeze()
        beta_var_t = beta2_t / tstat2_t
        var_prior = 0.0225 * phenotype_var_t
        r = var_prior / (var_prior + beta_var_t)
        l1 = 0.5 * ((1 - r).log() + r*tstat2_t)
    else:
        # compute p-values and z-score to match COLOC results exactly
        # (instead of directly using t-statistic)
        tstat_t = r_nominal_t * torch.sqrt(dof / (1 - r2_nominal_t))
        p = stats.t.cdf(-np.abs(tstat_t.cpu().numpy()), dof)  # 2 dropped since canceled in isf
        maf_t = calculate_maf(genotypes1_t)
        N = phenotype1_t.shape[0]
        v = 1 / (2 * N * maf_t * (1 - maf_t))
        z2_t = torch.Tensor(stats.norm.isf(p)**2).to(device)
        r = 0.0225 / (0.0225 + v)
        l1 = 0.5 * ((1 - r).log() + r*z2_t)

    # phenotype 2
    if phenotype2_t.dim() == 1:
        num_phenotypes = 1
        num_samples = phenotype2_t.shape[0]
        phenotype2_t = phenotype2_t.reshape(1,-1)
    else:
        num_phenotypes, num_samples = phenotype2_t.shape

    if mode == 'beta':
        r_nominal_t, genotype_var_t, phenotype_var_t = calculate_corr(
            genotypes2_t, phenotype2_t, residualizer2, return_var=True)
        r_nominal_t = r_nominal_t.squeeze()
        var_ratio_t = phenotype_var_t.reshape(1,-1) / genotype_var_t.reshape(-1,1)
    else:
        r_nominal_t = calculate_corr(genotypes2_t, phenotype2_t, residualizer2, return_var=False).squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    if residualizer2 is not None:
        dof = residualizer2.dof
    else:
        dof = num_samples - 2

    if mode == 'beta':
        tstat2_t = r2_nominal_t * dof / (1 - r2_nominal_t)
        beta2_t = r2_nominal_t * var_ratio_t.squeeze()
        beta_var_t = beta2_t / tstat2_t
        var_prior = 0.0225 * phenotype_var_t
        r = var_prior / (var_prior + beta_var_t)
        l2 = 0.5 * ((1 - r).log() + r*tstat2_t)
    else:
        tstat_t = r_nominal_t * torch.sqrt(dof / (1 - r2_nominal_t))
        p = stats.t.cdf(-np.abs(tstat_t.cpu().numpy()), dof)
        maf_t = calculate_maf(genotypes2_t)
        v = 1 / (2 * num_samples * maf_t * (1 - maf_t))
        z2_t = torch.Tensor(stats.norm.isf(p)**2).to(device)
        r = 0.0225 / (0.0225 + v)
        if num_phenotypes > 1:
            r = r.reshape(-1,1)
        l2 = 0.5 * ((1 - r).log() + r*z2_t)

    if num_phenotypes > 1:
        lsum = l1.reshape(-1,1) + l2
        lh0_abf = torch.zeros([1, num_phenotypes]).to(device)
        lh1_abf = np.log(p1) + logsumexp(l1).repeat([1, num_phenotypes])
    else:
        lsum = l1 + l2
        lh0_abf = torch.zeros([1]).to(device)
        lh1_abf = np.log(p1) + logsumexp(l1)
    lh2_abf = np.log(p2) + logsumexp(l2)
    lh3_abf = np.log(p1) + np.log(p2) + logdiff(logsumexp(l1) + logsumexp(l2), logsumexp(lsum))
    lh4_abf = np.log(p12) + logsumexp(lsum)
    all_abf = torch.cat([lh0_abf, lh1_abf, lh2_abf, lh3_abf, lh4_abf])
    return (all_abf - logsumexp(all_abf, dim=0)).exp().squeeze()


def run_pairs(genotype_df, variant_df, phenotype1_df, phenotype2_df, phenotype_pos_df,
              covariates1_df=None, covariates2_df=None, p1=1e-4, p2=1e-4, p12=1e-5, mode='beta',
              maf_threshold=0, window=1000000, batch_size=10000, logger=None, verbose=True):
    """Compute COLOC for all phenotype pairs"""

    assert np.all(phenotype1_df.index == phenotype2_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('Computing COLOC for all pairs of phenotypes')
    logger.write(f'  * {phenotype1_df.shape[0]} phenotypes')
    logger.write(f'  * phenotype group 1: {phenotype1_df.shape[1]} samples')
    logger.write(f'  * phenotype group 2: {phenotype2_df.shape[1]} samples')

    if covariates1_df is not None:
        assert np.all(phenotype1_df.columns == covariates1_df.index)
        logger.write(f'  * phenotype group 1: {covariates1_df.shape[1]} covariates')
        residualizer1 = Residualizer(torch.tensor(covariates1_df.values, dtype=torch.float32).to(device))
    else:
        residualizer1 = None

    if covariates2_df is not None:
        assert np.all(phenotype2_df.columns == covariates2_df.index)
        logger.write(f'  * phenotype group 2: {covariates2_df.shape[1]} covariates')
        residualizer2 = Residualizer(torch.tensor(covariates2_df.values, dtype=torch.float32).to(device))
    else:
        residualizer2 = None

    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter (in at least one cohort)')

    genotype1_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype1_df.columns])
    genotype1_ix_t = torch.from_numpy(genotype1_ix).to(device)
    genotype2_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype2_df.columns])
    genotype2_ix_t = torch.from_numpy(genotype2_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype1_df, phenotype_pos_df, window=window)
    coloc_df = []
    start_time = time.time()
    logger.write('  * Computing pairwise colocalization')
    for phenotype1, genotypes, genotype_range, phenotype_id in igc.generate_data(verbose=verbose):
        phenotype2 = phenotype2_df.loc[phenotype_id]

        # copy to GPU
        phenotype1_t = torch.tensor(phenotype1, dtype=torch.float).to(device)
        phenotype2_t = torch.tensor(phenotype2, dtype=torch.float).to(device)
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        genotypes1_t = genotypes_t[:,genotype1_ix_t]
        genotypes2_t = genotypes_t[:,genotype2_ix_t]
        del genotypes_t

        impute_mean(genotypes1_t)
        impute_mean(genotypes2_t)
        # filter monomorphic sites
        m = ((genotypes1_t==0).all(1) | (genotypes1_t==1).all(1) | (genotypes1_t==2).all(1) |
             (genotypes2_t==0).all(1) | (genotypes2_t==1).all(1) | (genotypes2_t==2).all(1))
        genotypes1_t = genotypes1_t[~m]
        genotypes2_t = genotypes2_t[~m]

        if maf_threshold > 0:
            maf1_t = calculate_maf(genotypes1_t)
            maf2_t = calculate_maf(genotypes2_t)
            mask_t = (maf1_t >= maf_threshold) | (maf2_t >= maf_threshold)
            genotypes1_t = genotypes1_t[mask_t]
            genotypes2_t = genotypes2_t[mask_t]

        coloc_t = coloc(genotypes1_t, genotypes2_t, phenotype1_t, phenotype2_t,
                        residualizer1=residualizer1, residualizer2=residualizer2,
                        p1=p1, p2=p2, p12=p12, mode=mode)
        coloc_df.append(coloc_t.cpu().numpy())
    logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
    coloc_df = pd.DataFrame(coloc_df, columns=[f'pp_h{i}_abf' for i in range(5)], index=phenotype1_df.index)
    logger.write('done.')
    return coloc_df

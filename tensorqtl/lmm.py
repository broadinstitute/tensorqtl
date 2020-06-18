
# LMM-based QTL solver
#
# Adapted from:
#   pyLMM (https://github.com/nickFurlotte/pylmm/)
#   Lippert, Listgarten et al., Nat Meth, 2011

import torch
from torch.utils import data
import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import OrderedDict
import sys
import os
import time

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio
from core import *


def _calculate_rrm(genotypes_t):
    """
    Calculate realized relationship matrix (RRM)

      genotypes_t: variants x samples
    """
    mask = genotypes_t == -1
    a = genotypes_t.sum(1, keepdim=True)
    b = mask.sum(1, keepdim=True).float()
    mu = (a + b) / (genotypes_t.shape[1] - b)  # nanmean
    m,n = genotypes_t.shape
    M_t = genotypes_t - mu
    M_t = M_t / torch.sqrt( m/n * (M_t.pow(2)).sum(1, keepdim=True) )
    M_t[mask] = 0
    return torch.mm(M_t.t(), M_t)


def realized_relationship_matrix(genotype_df, batch_size=100000, logger=None):
    """
    Calculate realized relationship matrix (RRM)

      genotype_df: genotype (dosage) dataframe (variants x samples)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = SimpleLogger()

    nv, ns = genotype_df.shape
    ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size)
    start_time = time.time()
    K_t = torch.zeros(ns, ns).to(device)
    logger.write('  * Computing RRM')
    for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=True), 1):
        # copy genotypes to GPU
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        K_t += _calculate_rrm(genotypes_t)
    logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
    return pd.DataFrame(K_t.cpu().numpy(), index=genotype_df.columns, columns=genotype_df.columns)


def mle_beta(h, X_t, yt_t, K_eval_t):
    """
    fixed effects beta for a single variant
    h: heritability
    X_t: covariates + variant
    yt_t: phenotype
    K_eval_t: kinship eigenvalues
    """
    # TODO: change this to process h x ns x nc tensors
    S = 1 / (h*K_eval_t + (1 - h))
    Xt = X_t.t() * S
    XX = torch.mm(Xt, X_t)
    XX_i = torch.inverse(XX)
    beta = torch.mm(torch.mm(XX_i, Xt), yt_t)
    Rt = yt_t - torch.mm(X_t, beta)
    Q = torch.mm(Rt.t()*S, Rt).squeeze()
    sigma = Q / (X_t.shape[0] - X_t.shape[1])
    return beta, sigma, Q, XX_i, XX


def LL(h, X_t, yt_t, K_eval_t, use_reml=True):
    """
    Computes the log-likelihood for a given heritability (h)
    h: heritability
    X_t: covariates matrix augmented with genotypes (samples x covariates + 1)
    yt_t: phenotype
    """
    n, q = X_t.shape
    beta, sigma, Q, XX_i, XX = mle_beta(h, X_t, yt_t, K_eval_t)
    LL = -0.5 * (n*np.log(2*np.pi) + torch.log(h*K_eval_t + (1-h)).sum() + n + n*torch.log(Q/n))

    if use_reml:
        LL += 0.5 * (q*torch.log(2*np.pi*sigma) + torch.logdet(torch.mm(X_t.t(), X_t)) - torch.logdet(XX))

    return LL, beta, sigma, XX_i


def LL_brent(h, X_t, yt_t, K_eval_t, use_reml=False):
    if h < 0:
        return 1e6
    return float(-LL(h, X_t, yt_t, K_eval_t, use_reml=use_reml)[0])


def get_max(LLs, H, X_t, yt_t, K_eval_t, use_reml=False, verbose=True):
    """
    Brent search for maximum, within candidate regions from grid search (input LLs)
    """
    n = len(LLs)
    HOpt = []
    for i in range(1, n-2):
        if LLs[i-1] < LLs[i] and LLs[i] > LLs[i+1]:
            HOpt.append(scipy.optimize.brent(LL_brent, args=(X_t, yt_t, K_eval_t, use_reml), brack=(H[i-1], H[i+1])))
            if np.isnan(HOpt[-1]):
                HOpt[-1] = H[i-1]

    if len(HOpt) > 1:
        if verbose:
            sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n")
        return HOpt[0]
    elif len(HOpt) == 1:
        return HOpt[0]
    elif LLs[0] > LLs[n-1]:
        return H[0]
    else:
        return H[n-1]


def fit(x_t, X0t_t, yt_t, K_evec_t, K_eval_t, ngrids=100, use_reml=True):
    """
    Fit LMM for a single variant (genotypes x_t)

    X0t_t: rotated covariates matrix
    yt_t:  rotated phenotype vector
    """
    # append genotype vector to covariates matrix
    X0t_t[:, (-1)] = torch.mm(K_evec_t.t(), x_t)[:,0]

    H = np.arange(0, 1, 1/ngrids)
    LLs = np.array([LL(h, X0t_t, yt_t, K_eval_t, use_reml=use_reml)[0] for h in H])
    hmax = get_max(LLs, H, X0t_t, yt_t, K_eval_t, use_reml=use_reml)

    L, beta, sigma, betaVAR = LL(hmax, X0t_t, yt_t, K_eval_t, use_reml=use_reml)

    N,q = X0t_t.shape
    ts = beta[q-1].squeeze() / torch.sqrt(betaVAR[q-1, q-1] * sigma)
    ps = 2.0*stats.t.sf(np.abs(np.float64(ts)), N-q)

    return hmax, beta, sigma, L, LLs, ts, ps


# def map_lmm(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
#             kinship_df, covariates_df, prefix,
#             window=1000000,
#             output_dir='.', logger=None, verbose=True):
#     """cis-QTL mapping with LMM"""
#
#     assert np.all(phenotype_df.columns==covariates_df.index)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     if logger is None:
#         logger = SimpleLogger()
#
#     logger.write('cis-QTL mapping with LMM: nominal associations for all variant-phenotype pairs')
#     logger.write('  * {} samples'.format(phenotype_df.shape[1]))
#     logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
#     logger.write('  * {} covariates'.format(covariates_df.shape[1]))
#     logger.write('  * {} variants'.format(variant_df.shape[0]))
#
#     K_t = torch.tensor(kinship_df.loc[phenotype_df.columns, phenotype_df.columns].values, dtype=torch.float32).to(device)
#     K_eval_t, K_evec_t = torch.symeig(K_t, eigenvectors=True)
#
#     C_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
#     Ct_t = torch.mm(K_evec_t.t(), C_t)
#     Ct0_t = torch.cat([Ct_t, torch.ones(Ct_t.shape[0], 1).to(device)], 1)  # augment with placeholder
#
#     genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
#     genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
#
#     igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
#     # iterate over chromosomes
#     start_time = time.time()
#     k = 0
#     logger.write('  * Computing associations')
#     for chrom in igc.chrs:
#         logger.write('    Mapping chromosome {}'.format(chrom))
#         # allocate arrays
#         n = 0
#         for i in igc.phenotype_pos_df[igc.phenotype_pos_df['chr']==chrom].index:
#             j = igc.cis_ranges[i]
#             n += j[1] - j[0] + 1
#
#         chr_res = OrderedDict()
#         chr_res['phenotype_id'] = []
#         chr_res['variant_id'] = []
#         chr_res['tss_distance'] = np.empty(n, dtype=np.int32)
#         chr_res['maf'] =          np.empty(n, dtype=np.float32)
#         chr_res['ma_samples'] =   np.empty(n, dtype=np.int32)
#         chr_res['ma_count'] =     np.empty(n, dtype=np.int32)
#         chr_res['pval'] =         np.empty(n, dtype=np.float64)
#         chr_res['b'] =            np.empty(n, dtype=np.float32)
#         chr_res['b_se'] =         np.empty(n, dtype=np.float32)
#
#         start = 0
#         for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):
#             # copy genotypes to GPU
#             phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
#             genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
#             genotypes_t = genotypes_t[:,genotype_ix_t]
#             impute_mean(genotypes_t)
#
#             yt_t = np.dot(K_evec_t, phenotype_t)
#
#             for k,x_t in enumerate(genotypes_t, 1):
#                 hmax_t, beta_t, sigma_t, L_t, LLs_t, ts_t, ps_t = lmm.fit(x_t.reshape(-1,1), Ct0_t, yt_t, K_evec_t, K_eval_t, ngrids=100, use_reml=True)
#                 pvals.append(ps_t)
#                 hm.append(hmax_t)
#
#
#
#             variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
#             n = len(variant_ids)
#             tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])
#
#             # res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
#             # tstat, slope, slope_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
#
#             chr_res['phenotype_id'].extend([phenotype_id]*n)
#             chr_res['variant_id'].extend(variant_ids)
#             chr_res['tss_distance'][start:start+n] = tss_distance
#             chr_res['maf'][start:start+n] = maf
#             chr_res['ma_samples'][start:start+n] = ma_samples
#             chr_res['ma_count'][start:start+n] = ma_count
#             # chr_res['pval_nominal'][start:start+n] = tstat
#             # chr_res['slope'][start:start+n] = slope
#             # chr_res['slope_se'][start:start+n] = slope_se
#             start += n  # update pointer
#
#         logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
#
#         # convert to dataframe, compute p-values and write current chromosome
#         if start < len(chr_res['maf']):
#             for x in chr_res:
#                 chr_res[x] = chr_res[x][:start]
#
#         chr_res_df = pd.DataFrame(chr_res)
#         m = chr_res_df['pval_nominal'].notnull()
#         chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
#         print('    * writing output')
#         chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.{}.parquet'.format(prefix, chrom)))
#
#     logger.write('done.')

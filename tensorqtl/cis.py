import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os
import time
from collections import OrderedDict

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio
from core import *


def calculate_cis_nominal(genotypes_t, phenotype_t, residualizer):
    """
    Calculate nominal associations

    genotypes_t: genotypes x samples
    phenotype_t: single phenotype
    covariates_t: covariates matrix, samples x covariates
    """
    p = phenotype_t.reshape(1,-1)
    r_nominal_t, std_ratio_t = calculate_corr(genotypes_t, p, residualizer, return_sd=True)
    r_nominal_t = r_nominal_t.squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    dof = residualizer.dof
    slope_t = r_nominal_t * std_ratio_t
    tstat_t = torch.sqrt((dof * r2_nominal_t) / (1 - r2_nominal_t))
    slope_se_t = slope_t.abs().double() / tstat_t
    # tdist = tfp.distributions.StudentT(np.float64(dof), loc=np.float64(0.0), scale=np.float64(1.0))
    # pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat)))

    # calculate MAF
    n2 = 2 * genotypes_t.shape[1]
    af_t = genotypes_t.sum(1) / n2
    ix_t = af_t <= 0.5
    maf_t = torch.where(ix_t, af_t, 1 - af_t)
    # calculate MA samples and counts
    m = genotypes_t > 0.5
    a = m.sum(1).int()
    b = (genotypes_t < 1.5).sum(1).int()
    ma_samples_t = torch.where(ix_t, a, b)
    a = (genotypes_t * m.float()).sum(1).int()
    ma_count_t = torch.where(ix_t, a, n2-a)

    return tstat_t, slope_t, slope_se_t, maf_t, ma_samples_t, ma_count_t


def calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t):
    """Calculate nominal and empirical correlations"""
    permutations_t = phenotype_t[permutation_ix_t]

    r_nominal_t, std_ratio_t = calculate_corr(genotypes_t, phenotype_t.reshape(1,-1), residualizer, return_sd=True)
    r_nominal_t = r_nominal_t.squeeze()
    corr_t = calculate_corr(genotypes_t, permutations_t, residualizer).pow(2)  # genotypes x permutations
    r2_perm_t,_ = corr_t[~torch.isnan(corr_t).any(1),:].max(0)

    r2_nominal_t = r_nominal_t.pow(2)
    r2_nominal_t[torch.isnan(r2_nominal_t)] = -1  # workaround for nanargmax()
    ix = r2_nominal_t.argmax()
    return r_nominal_t[ix], std_ratio_t[ix], ix, r2_perm_t, genotypes_t[ix]


def map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, prefix,
                interaction_s=None, maf_threshold_interaction=0.05,
                group_s=None, window=1000000, output_dir='.', logger=None, verbose=True):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    """
    assert np.all(phenotype_df.columns==covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()
    if group_s is not None:
        group_dict = group_s.to_dict()

    logger.write('cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(variant_df.shape[0]))
    if interaction_s is not None:
        assert np.all(interaction_s.index==covariates_df.index)
        logger.write('  * including interaction term')
        if maf_threshold_interaction>0:
            logger.write('    * using {:.2f} MAF threshold'.format(maf_threshold_interaction))

    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    residualizer = Residualizer(covariates_t)
    del covariates_t

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    if interaction_s is None:
        dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]
    else:
        dof = phenotype_df.shape[1] - 4 - covariates_df.shape[1]
        interaction_t = torch.tensor(interaction_s.values.reshape(1,-1), dtype=torch.float32).to(device)
        if maf_threshold_interaction>0:
            interaction_mask_t = torch.BoolTensor(interaction_s >= interaction_s.median()).to(device)
        else:
            interaction_mask_t = None

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    # iterate over chromosomes
    best_assoc = []
    start_time = time.time()
    prev_phenotype_id = None
    k = 0
    logger.write('  * Computing associations')
    for chrom in igc.chrs:
        logger.write('    Mapping chromosome {}'.format(chrom))
        chr_res_df = []
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):
            # copy genotypes to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            if interaction_s is None:
                res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                tstat, slope, slope_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                # r = igc.cis_ranges[phenotype_id]
                variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
                tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])
                res_df = pd.DataFrame(OrderedDict([
                    ('phenotype_id', [phenotype_id]*len(variant_ids)),
                    ('variant_id', variant_ids),
                    ('tss_distance', tss_distance),
                    ('maf', maf),
                    ('ma_samples', ma_samples),
                    ('ma_count', ma_count),
                    ('pval_nominal', tstat),  #### replace with pval (currently on CPU, below)
                    ('slope', slope),
                    ('slope_se', slope_se),
                ]))
            else:
                res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer,
                                                    interaction_mask_t=interaction_mask_t,
                                                    maf_threshold_interaction=maf_threshold_interaction,
                                                    return_sparse=False)
                tstat, b, b_se, maf, ma_samples, ma_count, mask = [i.cpu().numpy() for i in res]
                if len(tstat)>0:
                    r = igc.cis_ranges[phenotype_id]
                    variant_ids = variant_df.index[r[0]:r[-1]+1]
                    tss_distance = np.int32(variant_df['pos'].values[r[0]:r[-1]+1] - igc.phenotype_tss[phenotype_id])
                    if mask is not None:
                        variant_ids = variant_ids[mask]
                        tss_distance = tss_distance[mask]
                    nv = len(variant_ids)
                    res_df = pd.DataFrame(OrderedDict([
                        ('phenotype_id', [phenotype_id]*nv),
                        ('variant_id', variant_ids),
                        ('tss_distance', tss_distance),
                        ('maf', maf),
                        ('ma_samples', ma_samples),
                        ('ma_count', ma_count),
                        ('pval_g', tstat[:,0]),
                        ('b_g', b[:,0]),
                        ('b_g_se', b_se[:,0]),
                        ('pval_i', tstat[:,1]),
                        ('b_i', b[:,1]),
                        ('b_i_se', b_se[:,1]),
                        ('pval_gi', tstat[:,2]),
                        ('b_gi', b[:,2]),
                        ('b_gi_se', b_se[:,2]),
                    ]))
                    best_assoc.append(res_df.loc[res_df['pval_gi'].abs().idxmax()])  # top variant only (pval_gi is t-statistic here, hence max)
                else:
                    res_df = None

            if group_s is not None and group_dict[phenotype_id]==group_dict.get(prev_phenotype_id):
                # store the strongest association within each group
                if interaction_s is None:
                    ix = res_df['pval_nominal'] > chr_res_df[-1]['pval_nominal']  # compare t-statistics
                else:
                    ix = res_df['pval_gi'] > chr_res_df[-1]['pval_gi']
                chr_res_df[-1].loc[ix] = res_df.loc[ix]
            else:
                chr_res_df.append(res_df)
            prev_phenotype_id = phenotype_id
        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # compute p-values and write current chromosome
        chr_res_df = pd.concat(chr_res_df, copy=False)
        if interaction_s is None:
            m = chr_res_df['pval_nominal'].notnull()
            chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
        else:
            m = chr_res_df['pval_gi'].notnull()
            chr_res_df.loc[m, 'pval_g'] =  2*stats.t.cdf(-chr_res_df.loc[m, 'pval_g'].abs(), dof)
            chr_res_df.loc[m, 'pval_i'] =  2*stats.t.cdf(-chr_res_df.loc[m, 'pval_i'].abs(), dof)
            chr_res_df.loc[m, 'pval_gi'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_gi'].abs(), dof)
        print('  * writing output')
        chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.{}.parquet'.format(prefix, chrom)))

    if interaction_s is not None:
        best_assoc = pd.concat(best_assoc, axis=1, sort=False).T.set_index('phenotype_id').infer_objects()
        m = best_assoc['pval_g'].notnull()
        best_assoc.loc[m, 'pval_g'] =  2*stats.t.cdf(-best_assoc.loc[m, 'pval_g'].abs(), dof)
        best_assoc.loc[m, 'pval_i'] =  2*stats.t.cdf(-best_assoc.loc[m, 'pval_i'].abs(), dof)
        best_assoc.loc[m, 'pval_gi'] = 2*stats.t.cdf(-best_assoc.loc[m, 'pval_gi'].abs(), dof)
        best_assoc.to_csv(os.path.join(output_dir, '{}.cis_qtl_top_assoc.txt.gz'.format(prefix)),
                          sep='\t', float_format='%.6g')
    logger.write('done.')


def prepare_cis_output(r_nominal, r2_perm, std_ratio, g, num_var, dof, variant_id, tss_distance, phenotype_id, nperm=10000):
    """Return nominal p-value, allele frequencies, etc. as pd.Series"""
    r2_nominal = r_nominal*r_nominal
    pval_perm = (np.sum(r2_perm>=r2_nominal)+1) / (nperm+1)

    slope = r_nominal * std_ratio
    tstat2 = dof * r2_nominal / (1 - r2_nominal)
    slope_se = np.abs(slope) / np.sqrt(tstat2)

    n2 = 2*len(g)
    maf = np.sum(g) / n2
    if maf <= 0.5:
        ref_factor = 1
        ma_samples = np.sum(g>0.5)
        ma_count = np.sum(g[g>0.5])
    else:
        maf = 1-maf
        ref_factor = -1
        ma_samples = np.sum(g<1.5)
        ma_count = n2 - np.sum(g[g>0.5])

    res_s = pd.Series(OrderedDict([
        ('num_var', num_var),
        ('beta_shape1', np.NaN),
        ('beta_shape2', np.NaN),
        ('true_df', np.NaN),
        ('pval_true_df', np.NaN),
        ('variant_id', variant_id),
        ('tss_distance', tss_distance),
        ('ma_samples', ma_samples),
        ('ma_count', ma_count),
        ('maf', maf),
        ('ref_factor', ref_factor),
        ('pval_nominal', pval_from_corr(r2_nominal, dof)),
        ('slope', slope),
        ('slope_se', slope_se),
        ('pval_perm', pval_perm),
        ('pval_beta', np.NaN),
    ]), name=phenotype_id)
    return res_s


def _process_group_permutations(buf, variant_df, tss, dof, group_id, nperm=10000):
    """
    Merge results for grouped phenotypes

    buf: [r_nominal, std_ratio, var_ix, r2_perm, g, num_var, phenotype_id]
    """
    # select phenotype with strongest nominal association
    max_ix = np.argmax(np.abs([b[0] for b in buf]))
    r_nominal, std_ratio, var_ix = buf[max_ix][:3]
    g, num_var, phenotype_id = buf[max_ix][4:]
    # select best phenotype correlation for each permutation
    r2_perm = np.max([b[3] for b in buf], 0)
    # return r_nominal, std_ratio, var_ix, r2_perm, g, num_var, phenotype_id
    variant_id = variant_df.index[var_ix]
    tss_distance = variant_df['pos'].values[var_ix] - tss
    res_s = prepare_cis_output(r_nominal, r2_perm, std_ratio, g, num_var, dof, variant_id, tss_distance, phenotype_id, nperm=nperm)
    res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof)
    res_s['group_id'] = group_id
    res_s['group_size'] = len(buf)
    return res_s


def map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
            group_s=None, beta_approx=True, nperm=10000,
            window=1000000, logger=None, seed=None, verbose=True):
    """Run cis-QTL mapping"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert np.all(phenotype_df.columns==covariates_df.index)
    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: empirical p-values for phenotypes')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    if group_s is not None:
        logger.write('  * {} phenotype groups'.format(len(group_s.unique())))
        group_dict = group_s.to_dict()
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(genotype_df.shape[0]))

    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    residualizer = Residualizer(covariates_t)
    del covariates_t

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    if seed is not None:
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(np.array([np.random.permutation(ix) for i in range(nperm)])).to(device)

    res_df = []
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    start_time = time.time()
    logger.write('  * computing permutations')
    if group_s is None:
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)

            res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
            r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
            var_ix = genotype_range[var_ix]
            variant_id = variant_df.index[var_ix]
            tss_distance = variant_df['pos'].values[var_ix] - igc.phenotype_tss[phenotype_id]
            res_s = prepare_cis_output(r_nominal, r2_perm, std_ratio, g, genotypes.shape[0], dof, variant_id, tss_distance, phenotype_id, nperm=nperm)
            if beta_approx:
                res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof)
            res_df.append(res_s)
    else:
        for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(group_s=group_s, verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            # iterate over phenotypes
            buf = []
            for phenotype in phenotypes:
                phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
                res = [i.cpu().numpy() for i in res]  # r_nominal, std_ratio, var_ix, r2_perm, g
                res[2] = genotype_range[var_ix]
                buf.append(res + [genotypes.shape[0], phenotype_id])
            res_s = _process_group_permutations(buf, variant_df, igc.phenotype_tss[phenotype_ids[0]], dof, group_id, nperm=nperm)
            res_df.append(res_s)

    res_df = pd.concat(res_df, axis=1, sort=False).T
    res_df.index.name = 'phenotype_id'
    logger.write('  Time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
    logger.write('done.')
    return res_df.astype(output_dtype_dict).infer_objects()


def map_independent(genotype_df, variant_df, cis_df, phenotype_df, phenotype_pos_df, covariates_df,
                    group_s=None, fdr=0.05, fdr_col='qval', nperm=10000, 
                    window=1000000, logger=None, seed=None, verbose=True):
    """
    Run independent cis-QTL mapping (forward-backward regression)

    cis_df: output from map_cis, annotated with q-values (calculate_qvalues)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert np.all(phenotype_df.index==phenotype_pos_df.index)
    if logger is None:
        logger = SimpleLogger()

    signif_df = cis_df[cis_df[fdr_col]<=fdr].copy()
    cols = [
        'num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df',
        'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf', 'ref_factor',
        'pval_nominal', 'slope', 'slope_se', 'pval_perm', 'pval_beta',
    ]
    if group_s is not None:
        cols += ['group_id', 'group_size']
    signif_df = signif_df[cols]
    signif_threshold = signif_df['pval_beta'].max()
    # subset significant phenotypes
    if group_s is None:
        ix = signif_df.index[signif_df.index.isin(phenotype_df.index)]
    else:
        ix = group_s[group_s.isin(signif_df['group_id'])].index
        ix = ix[ix.isin(phenotype_df.index)]
    phenotype_df = phenotype_df.loc[ix]
    phenotype_pos_df = phenotype_pos_df.loc[ix]

    logger.write('cis-QTL mapping: conditionally independent variants')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} significant phenotypes'.format(signif_df.shape[0]))
    if group_s is not None:
        logger.write('  * {} phenotype groups'.format(len(group_s.unique())))
        group_dict = group_s.to_dict()
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(genotype_df.shape[0]))
    # print('Significance threshold: {}'.format(signif_threshold))

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]
    ix_dict = {i:k for k,i in enumerate(genotype_df.index)}

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    if seed is not None:
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(np.array([np.random.permutation(ix) for i in range(nperm)])).to(device)

    res_df = []
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    logger.write('  * computing independent QTLs')
    start_time = time.time()
    if group_s is None:
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            # 1) forward pass
            forward_df = [signif_df.loc[phenotype_id]]  # initialize results with top variant
            covariates = covariates_df.values.copy()  # initialize covariates
            dosage_dict = {}
            while True:
                # add variant to covariates
                variant_id = forward_df[-1]['variant_id']
                ig = genotype_df.values[ix_dict[variant_id], genotype_ix]
                dosage_dict[variant_id] = ig
                covariates = np.hstack([covariates, ig.reshape(-1,1)]).astype(np.float32)
                dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                covariates_t = torch.tensor(covariates, dtype=torch.float32).to(device)
                residualizer = Residualizer(covariates_t)
                del covariates_t

                res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
                r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
                x = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof)
                # add to list if empirical p-value passes significance threshold
                if x[0] <= signif_threshold:
                    var_ix = genotype_range[var_ix]
                    variant_id = variant_df.index[var_ix]
                    tss_distance = variant_df['pos'].values[var_ix] - igc.phenotype_tss[phenotype_id]
                    res_s = prepare_cis_output(r_nominal, r2_perm, std_ratio, g, genotypes.shape[0], dof, variant_id, tss_distance, phenotype_id, nperm=nperm)
                    res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = x
                    forward_df.append(res_s)
                else:
                    break
            forward_df = pd.concat(forward_df, axis=1, sort=False).T
            dosage_df = pd.DataFrame(dosage_dict)

            # 2) backward pass
            if forward_df.shape[0]>1:
                back_df = []
                variant_set = set()
                for k,i in enumerate(forward_df['variant_id'], 1):
                    covariates = np.hstack([
                        covariates_df.values,
                        dosage_df[np.setdiff1d(forward_df['variant_id'], i)].values,
                    ])
                    dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                    covariates_t = torch.tensor(covariates, dtype=torch.float32).to(device)
                    residualizer = Residualizer(covariates_t)
                    del covariates_t

                    res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
                    r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
                    var_ix = genotype_range[var_ix]
                    variant_id = variant_df.index[var_ix]
                    x = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof)
                    if x[0] <= signif_threshold and variant_id not in variant_set:
                        tss_distance = variant_df['pos'].values[var_ix] - igc.phenotype_tss[phenotype_id]
                        res_s = prepare_cis_output(r_nominal, r2_perm, std_ratio, g, genotypes.shape[0], dof, variant_id, tss_distance, phenotype_id, nperm=nperm)
                        res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = x
                        res_s['rank'] = k
                        back_df.append(res_s)
                        variant_set.add(variant_id)
                if len(back_df)>0:
                    res_df.append(pd.concat(back_df, axis=1, sort=False).T)
            else:  # single independent variant
                forward_df['rank'] = 1
                res_df.append(forward_df)

    else:  # grouped phenotypes
        for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(group_s=group_s, verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            # 1) forward pass
            forward_df = [signif_df[signif_df['group_id']==group_id].iloc[0]]  # initialize results with top variant
            covariates = covariates_df.values.copy()  # initialize covariates
            dosage_dict = {}
            while True:
                # add variant to covariates
                variant_id = forward_df[-1]['variant_id']
                ig = genotype_df.values[ix_dict[variant_id], genotype_ix]
                dosage_dict[variant_id] = ig
                covariates = np.hstack([covariates, ig.reshape(-1,1)]).astype(np.float32)
                dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                covariates_t = torch.tensor(covariates, dtype=torch.float32).to(device)
                residualizer = Residualizer(covariates_t)
                del covariates_t

                # iterate over phenotypes
                buf = []
                for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                    phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                    res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
                    res = [i.cpu().numpy() for i in res]  # r_nominal, std_ratio, var_ix, r2_perm, g
                    res[2] = genotype_range[res[2]]
                    buf.append(res + [genotypes.shape[0], phenotype_id])
                res_s = _process_group_permutations(buf, variant_df, igc.phenotype_tss[phenotype_ids[0]], dof, group_id, nperm=nperm)

                # add to list if significant
                if res_s['pval_beta'] <= signif_threshold:
                    forward_df.append(res_s)
                else:
                    break
            forward_df = pd.concat(forward_df, axis=1, sort=False).T
            dosage_df = pd.DataFrame(dosage_dict)

            # 2) backward pass
            if forward_df.shape[0]>1:
                back_df = []
                variant_set = set()
                for k,variant_id in enumerate(forward_df['variant_id'], 1):
                    covariates = np.hstack([
                        covariates_df.values,
                        dosage_df[np.setdiff1d(forward_df['variant_id'], variant_id)].values,
                    ])
                    dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                    covariates_t = torch.tensor(covariates, dtype=torch.float32).to(device)
                    residualizer = Residualizer(covariates_t)
                    del covariates_t

                    # iterate over phenotypes
                    buf = []
                    for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                        res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
                        res = [i.cpu().numpy() for i in res]  # r_nominal, std_ratio, var_ix, r2_perm, g
                        res[2] = genotype_range[res[2]]
                        buf.append(res + [genotypes.shape[0], phenotype_id])
                    res_s = _process_group_permutations(buf, variant_df, igc.phenotype_tss[phenotype_ids[0]], dof, group_id, nperm=nperm)

                    if res_s['pval_beta'] <= signif_threshold and variant_id not in variant_set:
                        res_s['rank'] = k
                        back_df.append(res_s)
                        variant_set.add(variant_id)
                if len(back_df)>0:
                    res_df.append(pd.concat(back_df, axis=1, sort=False).T)
            else:  # single independent variant
                forward_df['rank'] = 1
                res_df.append(forward_df)

    res_df = pd.concat(res_df, axis=0, sort=False)
    res_df.index.name = 'phenotype_id'
    logger.write('  Time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
    logger.write('done.')
    return res_df.reset_index().astype(output_dtype_dict)

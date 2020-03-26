import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os
import time
from collections import OrderedDict

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio, eigenmt
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
    slope_t = r_nominal_t * std_ratio_t.squeeze()
    tstat_t = r_nominal_t * torch.sqrt(dof / (1 - r2_nominal_t))
    slope_se_t = (slope_t.double() / tstat_t).float()
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
    r_nominal_t = r_nominal_t.squeeze(dim=-1)
    std_ratio_t = std_ratio_t.squeeze(dim=-1)
    corr_t = calculate_corr(genotypes_t, permutations_t, residualizer).pow(2)  # genotypes x permutations
    corr_t = corr_t[~torch.isnan(corr_t).any(1),:]
    if corr_t.shape[0] == 0:
        raise ValueError('All correlations resulted in NaN. Please check phenotype values.')
    r2_perm_t,_ = corr_t.max(0)  # maximum correlation across permutations

    r2_nominal_t = r_nominal_t.pow(2)
    r2_nominal_t[torch.isnan(r2_nominal_t)] = -1  # workaround for nanargmax()
    ix = r2_nominal_t.argmax()
    return r_nominal_t[ix], std_ratio_t[ix], ix, r2_perm_t, genotypes_t[ix]


def map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, prefix,
                interaction_s=None, maf_threshold_interaction=0.05,
                group_s=None, window=1000000, run_eigenmt=False,
                output_dir='.', write_top=True, write_stats=True, logger=None, verbose=True):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet

    If interaction_s is provided, the top association per phenotype is
    written to <output_dir>/<prefix>.cis_qtl_top_assoc.txt.gz unless
    write_top is set to False, in which case it is returned as a DataFrame
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
        if maf_threshold_interaction > 0:
            interaction_mask_t = torch.BoolTensor(interaction_s >= interaction_s.median()).to(device)
        else:
            interaction_mask_t = None

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
    # iterate over chromosomes
    best_assoc = []
    start_time = time.time()
    k = 0
    logger.write('  * Computing associations')
    for chrom in igc.chrs:
        logger.write('    Mapping chromosome {}'.format(chrom))
        # allocate arrays
        n = 0
        if group_s is None:
            for i in igc.phenotype_pos_df[igc.phenotype_pos_df['chr']==chrom].index:
                j = igc.cis_ranges[i]
                n += j[1] - j[0] + 1
        else:
            for i in igc.group_s[igc.phenotype_pos_df['chr']==chrom].drop_duplicates().index:
                j = igc.cis_ranges[i]
                n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['tss_distance'] = np.empty(n, dtype=np.int32)
        chr_res['maf'] =          np.empty(n, dtype=np.float32)
        chr_res['ma_samples'] =   np.empty(n, dtype=np.int32)
        chr_res['ma_count'] =     np.empty(n, dtype=np.int32)
        if interaction_s is None:
            chr_res['pval_nominal'] = np.empty(n, dtype=np.float64)
            chr_res['slope'] =        np.empty(n, dtype=np.float32)
            chr_res['slope_se'] =     np.empty(n, dtype=np.float32)
        else:
            chr_res['pval_g'] =  np.empty(n, dtype=np.float64)
            chr_res['b_g'] =     np.empty(n, dtype=np.float32)
            chr_res['b_g_se'] =  np.empty(n, dtype=np.float32)
            chr_res['pval_i'] =  np.empty(n, dtype=np.float64)
            chr_res['b_i'] =     np.empty(n, dtype=np.float32)
            chr_res['b_i_se'] =  np.empty(n, dtype=np.float32)
            chr_res['pval_gi'] = np.empty(n, dtype=np.float64)
            chr_res['b_gi'] =    np.empty(n, dtype=np.float32)
            chr_res['b_gi_se'] = np.empty(n, dtype=np.float32)

        start = 0
        if group_s is None:
            for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):
                # copy genotypes to GPU
                phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                genotypes_t = genotypes_t[:,genotype_ix_t]
                impute_mean(genotypes_t)

                variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
                tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])

                if interaction_s is None:
                    res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                    tstat, slope, slope_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                    n = len(variant_ids)
                else:
                    genotypes_t, mask_t = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t,
                                                                 maf_threshold_interaction=maf_threshold_interaction)
                    if genotypes_t.shape[0]>0:
                        res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer,
                                                            return_sparse=False)
                        tstat, b, b_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                        mask = mask_t.cpu().numpy()
                        variant_ids = variant_ids[mask]
                        tss_distance = tss_distance[mask]
                        n = len(variant_ids)

                        # top association
                        ix = np.nanargmax(np.abs(tstat[:,2]))
                        top_s = pd.Series([phenotype_id, variant_ids[ix], tss_distance[ix], maf[ix], ma_samples[ix], ma_count[ix],
                                           tstat[ix,0], b[ix,0], b_se[ix,0],
                                           tstat[ix,1], b[ix,1], b_se[ix,1],
                                           tstat[ix,2], b[ix,2], b_se[ix,2]], index=chr_res.keys())
                        if run_eigenmt:  # compute eigenMT correction
                            top_s['tests_emt'] = eigenmt.compute_tests(genotypes_t, var_thresh=0.99, variant_window=200)
                        best_assoc.append(top_s)
                    else:  # all genotypes in window were filtered out
                        n = 0

                if n > 0:
                    chr_res['phenotype_id'].extend([phenotype_id]*n)
                    chr_res['variant_id'].extend(variant_ids)
                    chr_res['tss_distance'][start:start+n] = tss_distance
                    chr_res['maf'][start:start+n] = maf
                    chr_res['ma_samples'][start:start+n] = ma_samples
                    chr_res['ma_count'][start:start+n] = ma_count
                    if interaction_s is None:
                        chr_res['pval_nominal'][start:start+n] = tstat
                        chr_res['slope'][start:start+n] = slope
                        chr_res['slope_se'][start:start+n] = slope_se
                    else:
                        chr_res['pval_g'][start:start+n]  = tstat[:,0]
                        chr_res['b_g'][start:start+n]     = b[:,0]
                        chr_res['b_g_se'][start:start+n]  = b_se[:,0]
                        chr_res['pval_i'][start:start+n]  = tstat[:,1]
                        chr_res['b_i'][start:start+n]     = b[:,1]
                        chr_res['b_i_se'][start:start+n]  = b_se[:,1]
                        chr_res['pval_gi'][start:start+n] = tstat[:,2]
                        chr_res['b_gi'][start:start+n]    = b[:,2]
                        chr_res['b_gi_se'][start:start+n] = b_se[:,2]
                start += n  # update pointer
        else:  # groups
            for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):

                # copy genotypes to GPU
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                genotypes_t = genotypes_t[:,genotype_ix_t]
                impute_mean(genotypes_t)

                variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
                # assuming that the TSS for all grouped phenotypes is the same
                tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_ids[0]])

                if interaction_s is not None:
                    genotypes_t, mask_t = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t,
                                                                 maf_threshold_interaction=maf_threshold_interaction)
                    mask = mask_t.cpu().numpy()
                    variant_ids = variant_ids[mask]
                    tss_distance = tss_distance[mask]

                n = len(variant_ids)

                if genotypes_t.shape[0]>0:
                    # process first phenotype in group
                    phenotype_id = phenotype_ids[0]
                    phenotype_t = torch.tensor(phenotypes[0], dtype=torch.float).to(device)

                    if interaction_s is None:
                        res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                        tstat, slope, slope_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                    else:
                        res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer, return_sparse=False)
                        tstat, b, b_se, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                    px = [phenotype_id]*n

                    # iterate over remaining phenotypes in group
                    for phenotype, phenotype_id in zip(phenotypes[1:], phenotype_ids[1:]):
                        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                        if interaction_s is None:
                            res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
                            tstat0, slope0, slope_se0, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                        else:
                            res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer, return_sparse=False)
                            tstat0, b0, b_se0, maf, ma_samples, ma_count = [i.cpu().numpy() for i in res]

                        # find associations that are stronger for current phenotype
                        if interaction_s is None:
                            ix = np.where(np.abs(tstat0) > np.abs(tstat))[0]
                        else:
                            ix = np.where(np.abs(tstat0[:,2]) > np.abs(tstat[:,2]))[0]

                        # update relevant positions
                        for j in ix:
                            px[j] = phenotype_id
                        if interaction_s is None:
                            tstat[ix] = tstat0[ix]
                            slope[ix] = slope0[ix]
                            slope_se[ix] = slope_se0[ix]
                        else:
                            tstat[ix] = tstat0[ix]
                            b[ix] = b0[ix]
                            b_se[ix] = b_se0[ix]

                    chr_res['phenotype_id'].extend(px)
                    chr_res['variant_id'].extend(variant_ids)
                    chr_res['tss_distance'][start:start+n] = tss_distance
                    chr_res['maf'][start:start+n] = maf
                    chr_res['ma_samples'][start:start+n] = ma_samples
                    chr_res['ma_count'][start:start+n] = ma_count
                    if interaction_s is None:
                        chr_res['pval_nominal'][start:start+n] = tstat
                        chr_res['slope'][start:start+n] = slope
                        chr_res['slope_se'][start:start+n] = slope_se
                    else:
                        chr_res['pval_g'][start:start+n]  = tstat[:,0]
                        chr_res['b_g'][start:start+n]     = b[:,0]
                        chr_res['b_g_se'][start:start+n]  = b_se[:,0]
                        chr_res['pval_i'][start:start+n]  = tstat[:,1]
                        chr_res['b_i'][start:start+n]     = b[:,1]
                        chr_res['b_i_se'][start:start+n]  = b_se[:,1]
                        chr_res['pval_gi'][start:start+n] = tstat[:,2]
                        chr_res['b_gi'][start:start+n]    = b[:,2]
                        chr_res['b_gi_se'][start:start+n] = b_se[:,2]

                    # top association for the group
                    if interaction_s is not None:
                        ix = np.nanargmax(np.abs(tstat[:,2]))
                        top_s = pd.Series([chr_res['phenotype_id'][start:start+n][ix], variant_ids[ix], tss_distance[ix], maf[ix], ma_samples[ix], ma_count[ix],
                                           tstat[ix,0], b[ix,0], b_se[ix,0],
                                           tstat[ix,1], b[ix,1], b_se[ix,1],
                                           tstat[ix,2], b[ix,2], b_se[ix,2]], index=chr_res.keys())
                        top_s['num_phenotypes'] = len(phenotype_ids)
                        if run_eigenmt:  # compute eigenMT correction
                            top_s['tests_emt'] = eigenmt.compute_tests(genotypes_t, var_thresh=0.99, variant_window=200)
                        best_assoc.append(top_s)

                start += n  # update pointer

        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # convert to dataframe, compute p-values and write current chromosome
        if start < len(chr_res['maf']):
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]

        if write_stats:
            chr_res_df = pd.DataFrame(chr_res)
            if interaction_s is None:
                m = chr_res_df['pval_nominal'].notnull()
                chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
            else:
                m = chr_res_df['pval_gi'].notnull()
                chr_res_df.loc[m, 'pval_g'] =  2*stats.t.cdf(-chr_res_df.loc[m, 'pval_g'].abs(), dof)
                chr_res_df.loc[m, 'pval_i'] =  2*stats.t.cdf(-chr_res_df.loc[m, 'pval_i'].abs(), dof)
                chr_res_df.loc[m, 'pval_gi'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_gi'].abs(), dof)
            print('    * writing output')
            chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.{}.parquet'.format(prefix, chrom)))

    if interaction_s is not None and len(best_assoc) > 0:
        best_assoc = pd.concat(best_assoc, axis=1, sort=False).T.set_index('phenotype_id').infer_objects()
        m = best_assoc['pval_g'].notnull()
        best_assoc.loc[m, 'pval_g'] =  2*stats.t.cdf(-best_assoc.loc[m, 'pval_g'].abs(), dof)
        best_assoc.loc[m, 'pval_i'] =  2*stats.t.cdf(-best_assoc.loc[m, 'pval_i'].abs(), dof)
        best_assoc.loc[m, 'pval_gi'] = 2*stats.t.cdf(-best_assoc.loc[m, 'pval_gi'].abs(), dof)
        if run_eigenmt:
            if group_s is None:
                best_assoc['pval_emt'] = np.minimum(best_assoc['tests_emt']*best_assoc['pval_gi'], 1)
            else:
                best_assoc['pval_emt'] = np.minimum(best_assoc['num_phenotypes']*best_assoc['tests_emt']*best_assoc['pval_gi'], 1)
            best_assoc['pval_adj_bh'] = eigenmt.padjust_bh(best_assoc['pval_emt'])
        if write_top:
            best_assoc.to_csv(os.path.join(output_dir, '{}.cis_qtl_top_assoc.txt.gz'.format(prefix)),
                              sep='\t', float_format='%.6g')
        else:
            return best_assoc
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
    res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof*0.25)
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
        logger.write('  * using seed {}'.format(seed))
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(np.array([np.random.permutation(ix) for i in range(nperm)])).to(device)

    res_df = []
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
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
        for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            # iterate over phenotypes
            buf = []
            for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                res = calculate_cis_permutations(genotypes_t, phenotype_t, residualizer, permutation_ix_t)
                res = [i.cpu().numpy() for i in res]  # r_nominal, std_ratio, var_ix, r2_perm, g
                res[2] = genotype_range[res[2]]
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
    assert np.all(covariates_df.index==phenotype_df.columns)
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
        ix = phenotype_df.index[phenotype_df.index.isin(signif_df.index)]
    else:
        ix = group_s[phenotype_df.index].loc[group_s[phenotype_df.index].isin(signif_df['group_id'])].index

    logger.write('cis-QTL mapping: conditionally independent variants')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    if group_s is None:
        logger.write('  * {}/{} significant phenotypes'.format(signif_df.shape[0], cis_df.shape[0]))
    else:
        logger.write('  * {}/{} significant groups'.format(signif_df.shape[0], cis_df.shape[0]))
        logger.write('    {}/{} phenotypes'.format(len(ix), phenotype_df.shape[0]))
        group_dict = group_s.to_dict()
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(genotype_df.shape[0]))
    # print('Significance threshold: {}'.format(signif_threshold))
    phenotype_df = phenotype_df.loc[ix]
    phenotype_pos_df = phenotype_pos_df.loc[ix]

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]
    ix_dict = {i:k for k,i in enumerate(genotype_df.index)}

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    if seed is not None:
        logger.write('  * using seed {}'.format(seed))
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(np.array([np.random.permutation(ix) for i in range(nperm)])).to(device)

    res_df = []
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
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
        for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(verbose=verbose), 1):
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

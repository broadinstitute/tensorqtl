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
    r2_nominal_t = r_nominal_t.pow(2)

    dof = residualizer.dof
    slope_t = r_nominal_t * std_ratio_t
    tstat_t = torch.sqrt((dof * r2_nominal_t) / (1 - r2_nominal_t))
    slope_se_t = slope_t.abs() / tstat_t
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


def map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, prefix,
                window=1000000, group_s=None, output_dir='.', logger=None):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    """
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

    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    residualizer = Residualizer(covariates_t)
    del covariates_t

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    # iterate over chromosomes
    start_time = time.time()
    prev_phenotype_id = ''
    k = 0
    for chrom in igc.chrs:
        logger.write('  Mapping chromosome {}'.format(chrom))
        chr_res_df = []
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom), k+1):
            # copy genotypes to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)


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


            if group_s is not None and group_dict[phenotype_id]==group_dict.get(prev_phenotype_id):
                # ix = res[0] > chr_res_df[-1][0]  # compare T-statistics
                # chr_res_df[-1] = [list(np.where(ix, i, j)) for i,j in zip(res, chr_res_df[-1])]
                ix = res_df['pval_nominal'] > chr_res_df[-1]['pval_nominal']
                chr_res_df[-1].loc[ix] = res_df.loc[ix]
            else:
                chr_res_df.append(res_df)
            prev_phenotype_id = phenotype_id

            print('\r    computing associations for phenotype {}/{}'.format(k, igc.n_phenotypes), end='')
        print()
        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
        print('  * writing output')
        chr_res_df = pd.concat(chr_res_df, copy=False)
        chr_res_df['pval_nominal'] = 2*stats.t.cdf(-chr_res_df['pval_nominal'].abs(), dof)
        chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.{}.parquet'.format(prefix, chrom)))
    logger.write('done.')


def map_nominal_interaction(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, interaction_s,
                                prefix, maf_threshold_interaction=0.05, best_only=False, group_s=None,
                                window=1000000, output_dir='.', logger=None):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    """
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
    logger.write('  * including interaction term')
    if maf_threshold_interaction>0:
        logger.write('  * using {:.2f} MAF threshold'.format(maf_threshold_interaction))

    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    residualizer = Residualizer(covariates_t)
    del covariates_t

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    dof = phenotype_df.shape[1] - 4 - covariates_df.shape[1]
    interaction_t = torch.tensor(interaction_s.values.reshape(1,-1), dtype=torch.float32).to(device)
    if maf_threshold_interaction>0:
        interaction_mask_t = torch.ByteTensor(interaction_s >= interaction_s.median()).to(device)
    else:
        interaction_mask_t = None

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    # iterate over chromosomes
    start_time = time.time()
    best_assoc = []
    prev_phenotype_id = None
    for chrom in igc.chrs:
        logger.write('  Mapping chromosome {}'.format(chrom))
        chr_res_df = []
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom), 1):
            # copy genotypes to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device).unsqueeze(0)
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            res = calculate_interaction_nominal(genotypes_t, phenotype_t, interaction_t, residualizer,
                                                interaction_mask_t=interaction_mask_t,
                                                maf_threshold_interaction=maf_threshold_interaction,
                                                return_sparse=False)
            # res: tstat, b, b_se, maf, ma_samples, ma_count, mask
            res = [i.cpu().numpy() for i in res]
            # res[-1] = variant_ids[res[-1].astype(bool)]
            tstat, b, b_se, maf, ma_samples, ma_count, mask = [i.cpu().numpy() for i in res]
            if len(tstat)>0:
                r = igc.cis_ranges[phenotype_id]
                variant_ids = variant_df.index[r[0]:r[-1]+1]
                tss_distance = np.int32(variant_df['pos'].values[r[0]:r[-1]+1] - igc.phenotype_tss[phenotype_id])
                if mask is not None:
                    variant_ids = variant_ids[mask]
                    tss_distance = tss_distance[mask]
                nv = len(variant_ids)
                df = pd.DataFrame(OrderedDict([
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
                best_assoc.append(df.loc[df['pval_gi'].idxmin()])


                if group_s is not None and group_dict[phenotype_id]==group_dict.get(prev_phenotype_id):
                    ix = df['pval_gi'] > chr_res_df[-1]['pval_gi']
                    chr_res_df[-1].loc[ix] = df.loc[ix]
                else:
                    chr_res_df.append(df)
                prev_phenotype_id = phenotype_id
                print('\r    computing associations for phenotype {}/{}'.format(i, igc.n_phenotypes), end='')
            print()
            logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

            if not best_only:
                print('  * writing output')
                chr_res_df = pd.concat(chr_res_df, copy=False)
                chr_res_df['pval_g'] = 2*stats.t.cdf(-chr_res_df['pval_g'].abs(), dof)
                chr_res_df['pval_i'] = 2*stats.t.cdf(-chr_res_df['pval_i'].abs(), dof)
                chr_res_df['pval_gi'] = 2*stats.t.cdf(-chr_res_df['pval_gi'].abs(), dof)
                pd.concat(chr_res_df, copy=False).to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairsX.{}.parquet'.format(prefix, chrom)))

    best_assoc = pd.concat(best_assoc, axis=1).T.set_index('phenotype_id').infer_objects()
    # TODO: compute pval
    best_assoc.to_csv(os.path.join(output_dir, '{}.cis_qtl_top_assoc.txt.gz'.format(prefix)),
                      sep='\t', float_format='%.6g')
    logger.write('done.')

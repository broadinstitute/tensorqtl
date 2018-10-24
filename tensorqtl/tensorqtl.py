#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import tensorflow as tf
import numpy as np
import gzip
import time
from collections import OrderedDict
from datetime import datetime
import sys
import os
import glob
import argparse
import scipy.optimize
import scipy.stats as stats
from scipy.special import loggamma

sys.path.append(os.path.dirname(__file__))
import genotypeio


class SimpleLogger(object):
    def __init__(self, logfile=None):
        self.console = sys.stdout
        if logfile is not None:
            self.log = open(logfile, 'w')
        else:
            self.log = None

    def write(self, message):
        self.console.write(message+'\n')
        if self.log is not None:
            self.log.write(message+'\n')
            self.log.flush()


#------------------------------------------------------------------------------
#  Core functions for mapping associations on GPU
#------------------------------------------------------------------------------
def residualize(M_t, C_t):
    """Residualize M wrt columns of C"""

    # center and orthogonalize
    Q_t, _ = tf.qr(C_t - tf.reduce_mean(C_t, 0), full_matrices=False, name='qr')

    # residualize M relative to C
    M0_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
    return M_t - tf.matmul(tf.matmul(M0_t, Q_t), Q_t, transpose_b=True)  # keep original mean


def center_normalize(M_t, axis=0):
    """Center and normalize M"""
    if axis == 0:
        N_t = M_t - tf.reduce_mean(M_t, 0)
        return tf.divide(N_t, tf.sqrt(tf.reduce_sum(tf.pow(N_t, 2), 0)))
    elif axis == 1:
        N_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
        return tf.divide(N_t, tf.sqrt(tf.reduce_sum(tf.pow(N_t, 2), axis=1, keepdims=True)))


def calculate_maf(genotype_t):
    """Calculate minor allele frequency"""
    af_t = tf.reduce_sum(genotype_t,1) / (2*genotype_t.shape[1].value)
    return tf.where(af_t>0.5, 1-af_t, af_t)


def _calculate_corr(genotype_t, phenotype_t, covariates_t, return_sd=False):
    """Calculate squared correlation between normalized residual genotypes and phenotypes"""
    # residualize
    genotype_res_t = residualize(genotype_t, covariates_t)  # variants x samples
    phenotype_res_t = residualize(phenotype_t, covariates_t)  # phenotypes x samples

    if return_sd:
        _, gstd = tf.nn.moments(genotype_res_t, axes=1)
        _, pstd = tf.nn.moments(phenotype_res_t, axes=1)

    # center and normalize
    genotype_res_t = center_normalize(genotype_res_t, axis=1)
    phenotype_res_t = center_normalize(phenotype_res_t, axis=1)

    # correlation
    if return_sd:
        return tf.pow(tf.squeeze(tf.matmul(genotype_res_t, phenotype_res_t, transpose_b=True)), 2), tf.sqrt(pstd / gstd)
    else:
        return tf.pow(tf.squeeze(tf.matmul(genotype_res_t, phenotype_res_t, transpose_b=True)), 2)


def calculate_pval(r2_t, dof, maf_t=None, return_sparse=True, r2_threshold=0):
    """Calculate p-values from squared correlations"""
    dims = r2_t.get_shape()
    if return_sparse:
        ix = tf.where(r2_t>=r2_threshold, name='threshold_r2')
        r2_t = tf.gather_nd(r2_t, ix)

    r2_t = tf.cast(r2_t, tf.float64)

    tstat = tf.sqrt(tf.divide(tf.scalar_mul(dof, r2_t), 1 - r2_t), name='tstat')
    tdist = tf.contrib.distributions.StudentT(np.float64(dof), loc=np.float64(0.0), scale=np.float64(1.0))

    if return_sparse:
        pval_t = tf.SparseTensor(ix, tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat))), dims)
        if maf_t is not None:
            maf_t = tf.gather(maf_t, ix[:,0])
    else:
        pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat)))

    if maf_t is not None:
        return pval_t, maf_t
    else:
        return pval_t


def _interaction_assoc_row(genotype_t, phenotype_t, icovariates_t):
    """genotype_t must be a 1D tensor"""
    gi_covariates_t = tf.concat([icovariates_t, tf.reshape(genotype_t, [-1,1])], axis=1)
    ix_t = tf.reshape(tf.multiply(genotype_t, icovariates_t[:,-1]), [1,-1])  # must be 1 x N
    return _calculate_corr(ix_t, phenotype_t, gi_covariates_t)[0]


def calculate_association(genotype_t, phenotype_t, covariates_t, interaction_t=None, return_sparse=True, r2_threshold=None):
    """Calculate genotype-phenotype associations"""
    maf_t = calculate_maf(genotype_t)

    if interaction_t is None:
        r2_t = _calculate_corr(genotype_t, phenotype_t, covariates_t)
        dof = genotype_t.shape[1].value - 2 - covariates_t.shape[1].value
    else:
        icovariates_t = tf.concat([covariates_t, tf.reshape(interaction_t, [-1,1])], axis=1)
        r2_t = tf.map_fn(lambda x: _interaction_assoc_row(x, phenotype_t, icovariates_t), genotype_t, infer_shape=False)
        dof = genotype_t.shape[1].value - 4 - covariates_t.shape[1].value

    return calculate_pval(r2_t, dof, maf_t, return_sparse=return_sparse, r2_threshold=r2_threshold)


def get_sample_indexes(vcf_sample_ids, phenotype_df):
    """Get index of sample IDs in VCF"""
    return tf.constant([vcf_sample_ids.index(i) for i in phenotype_df.columns])


def initialize_data(phenotype_df, covariates_df, batch_size, interaction_s=None, dtype=tf.float32):
    """Generate placeholders"""
    num_samples = phenotype_df.shape[1]
    genotype_t = tf.placeholder(dtype, shape=[batch_size, num_samples])
    phenotype_t = tf.constant(phenotype_df.values, dtype=dtype)
    phenotype_t = tf.reshape(phenotype_t, shape=[-1, num_samples])
    covariates_t = tf.constant(covariates_df.values, dtype=dtype)
    covariates_t = tf.reshape(covariates_t, shape=[-1, covariates_df.shape[1]])
    if interaction_s is None:
        return genotype_t, phenotype_t, covariates_t
    else:
        interaction_t = tf.constant(interaction_s.values, dtype=dtype)
        interaction_t = tf.reshape(interaction_t, [-1,1])
        return genotype_t, phenotype_t, covariates_t, interaction_t

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

def calculate_beta_approx_pval(r2_perm, r2_nominal, dof, tol=1e-4, return_minp=False):
    """
      r2_nominal: nominal max. r2 (scalar or array)
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

    pval_true_dof = pval_from_corr(r2_nominal, true_dof)
    pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)
    if return_minp:
        return pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof, pval
    else:
        return pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof

#------------------------------------------------------------------------------
#  Top-level functions for running cis-/trans-QTL mapping
#------------------------------------------------------------------------------
def calculate_cis_permutations(genotypes_t, range_t, phenotype_t, covariates_t, permutation_ix_t, dof, tdist):
    """Calculate nominal and empirical correlations"""
    permutations_t = tf.gather(phenotype_t, permutation_ix_t)

    r2_nominal_t, std_ratio_t = _calculate_corr(genotypes_t, tf.reshape(phenotype_t, [1,-1]), covariates_t, return_sd=True)

    corr_t = _calculate_corr(genotypes_t, permutations_t, covariates_t)
    corr_t.set_shape([None,None])
    r2_perm_t = tf.cast(tf.reduce_max(tf.boolean_mask(corr_t, ~tf.reduce_any(tf.is_nan(corr_t), 1)), axis=0), tf.float64)

    ix = tf.argmax(r2_nominal_t)
    return r2_nominal_t[ix], std_ratio_t[ix], range_t[ix], r2_perm_t, genotypes_t[ix], tf.shape(r2_nominal_t)[0]


def process_cis_permutations(r2_perm, r2_nominal, std_ratio, g, ng, dof, n_samples, nperm=10000):
    """Calculate beta-approximated empirical p-value and annotate phenotype"""
    pval_perm = (np.sum(r2_perm>=r2_nominal)+1) / (nperm+1)
    pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof = calculate_beta_approx_pval(r2_perm, r2_nominal, dof)

    maf = np.sum(g) / (2*n_samples)
    if maf <= 0.5:
        ref_factor = 1
        ma_samples = np.sum(g>0.5)
        ma_count = np.sum(g[g>0.5])
    else:
        maf = 1-maf
        ref_factor = -1
        ma_samples = np.sum(g<1.5)
        ma_count = np.sum(g[g<1.5])

    slope = np.sqrt(r2_nominal) * std_ratio
    tstat2 = dof * r2_nominal / (1 - r2_nominal)
    slope_se = np.abs(slope) / np.sqrt(tstat2)

    return pd.Series(OrderedDict([
        ('num_var', ng),
        ('beta_shape1', beta_shape1),
        ('beta_shape2', beta_shape2),
        ('true_df', true_dof),
        ('pval_true_df', pval_true_dof),
        ('variant_id', np.NaN),
        ('tss_distance', np.NaN),
        ('ma_samples', ma_samples),
        ('ma_count', ma_count),
        ('maf', maf),
        ('ref_factor', ref_factor),
        ('pval_nominal', pval_from_corr(r2_nominal, dof)),
        ('slope', slope),
        ('slope_se', slope_se),
        ('pval_perm', pval_perm),
        ('pval_beta', pval_beta),
    ]))


def map_cis(plink_reader, phenotype_df, phenotype_pos_df, covariates_df, nperm=10000, logger=None):
    """Run cis-QTL mapping"""
    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: empirical p-values for phenotypes')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(plink_reader.bed.shape[0]))

    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]
    tdist = tf.contrib.distributions.StudentT(np.float64(dof), loc=np.float64(0.0), scale=np.float64(1.0))

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    permutation_ix_t = tf.convert_to_tensor(np.array([np.random.permutation(ix) for i in range(nperm)]))

    # placeholders
    covariates_t = tf.constant(covariates_df.values, dtype=tf.float32)
    genotype_t = tf.placeholder(dtype=tf.float32, shape=(None))
    phenotype_t = tf.placeholder(dtype=tf.float32, shape=(None))

    # iterate over chromosomes
    res_df = []
    with tf.Session() as sess:
        for chrom in phenotype_pos_df.loc[phenotype_df.index, 'chr'].unique():
            logger.write('  Mapping chromosome {}'.format(chrom))
            igc = genotypeio.InputGeneratorCis(plink_reader, phenotype_df.loc[phenotype_pos_df['chr']==chrom], phenotype_pos_df)

            dataset = tf.data.Dataset.from_generator(igc.generate_data, output_types=(tf.float32, tf.float32, tf.int32, tf.string))
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            next_phenotype, next_genotypes, next_range, next_id = iterator.get_next()

            r2_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t, ng_t = calculate_cis_permutations(
                next_genotypes, next_range, next_phenotype, covariates_t, permutation_ix_t, dof, tdist)

            for i in range(1, igc.n_phenotypes+1):
                r2_nom, s_r, var_ix, r2_perm, g, ng, nid = sess.run([r2_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t, ng_t, next_id])

                # post-processing (on CPU)
                res_s = process_cis_permutations(r2_perm, r2_nom, s_r, g, ng, dof, phenotype_df.shape[1], nperm=nperm)
                res_s.name = nid.decode()
                res_s['variant_id'] = igc.chr_variant_pos.index[var_ix]
                res_s['tss_distance'] = igc.chr_variant_pos[res_s['variant_id']] - igc.phenotype_tss[res_s.name]
                res_df.append(res_s)
                print('\r  * computing permutations for phenotype {}/{}'.format(i, igc.n_phenotypes), end='')
            print()
    res_df = pd.concat(res_df, axis=1).T
    res_df.index.name = 'phenotype_id'
    logger.write('done.')
    return res_df


def calculate_cis_nominal(genotypes_t, phenotype_t, covariates_t, dof):
    """Calculate nominal associations"""
    r2_nominal_t, std_ratio_t = _calculate_corr(genotypes_t, tf.reshape(phenotype_t, [1,-1]), covariates_t, return_sd=True)
    pval_t = calculate_pval(r2_nominal_t, dof, maf_t=None, return_sparse=False)
    slope_t = tf.multiply(tf.sqrt(r2_nominal_t), std_ratio_t)
    slope_se_t = tf.divide(tf.abs(slope_t), tf.sqrt(tf.divide(tf.scalar_mul(dof, r2_nominal_t), 1 - r2_nominal_t)))

    n = covariates_t.shape[0].value
    n2 = 2*n
    af_t = tf.reduce_sum(genotypes_t,1) / n2
    ix_t = af_t<=0.5
    maf_t = tf.where(ix_t, af_t, 1-af_t)

    m = tf.cast(genotypes_t>0.5, tf.float32)
    a = tf.reduce_sum(m, 1)
    b = tf.reduce_sum(tf.cast(genotypes_t<1.5, tf.float32), 1)
    ma_samples_t = tf.where(ix_t, a, b)
    m = tf.multiply(m, genotypes_t)
    a = tf.reduce_sum(m, 1)
    ma_count_t = tf.where(ix_t, a, n2-a)

    return pval_t, slope_t, slope_se_t, maf_t, ma_samples_t, ma_count_t


def map_cis_nominal(plink_reader, phenotype_df, phenotype_pos_df, covariates_df, prefix, output_dir='.', interaction_s=None, logger=None):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.variant_phenotype_pairs.<chr>.parquet
    """
    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(plink_reader.bed.shape[0]))
    if interaction_s is not None:
        logger.write('  * including interaction term')

    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]
    tdist = tf.contrib.distributions.StudentT(np.float64(dof), loc=np.float64(0.0), scale=np.float64(1.0))

    # placeholders
    covariates_t = tf.constant(covariates_df.values, dtype=tf.float32)
    genotype_t = tf.placeholder(dtype=tf.float32, shape=(None))
    phenotype_t = tf.placeholder(dtype=tf.float32, shape=(None))

    with tf.Session() as sess:
        # iterate over chromosomes
        for chrom in phenotype_pos_df.loc[phenotype_df.index, 'chr'].unique():
            logger.write('  Mapping chromosome {}'.format(chrom))
            igc = genotypeio.InputGeneratorCis(plink_reader, phenotype_df.loc[phenotype_pos_df['chr']==chrom], phenotype_pos_df)

            dataset = tf.data.Dataset.from_generator(igc.generate_data, output_types=(tf.float32, tf.float32, tf.int32, tf.string))
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            next_phenotype, next_genotypes, _, next_id = iterator.get_next()
            pval_t, slope_t, slope_se_t, maf_t, ma_samples_t, ma_count_t = calculate_cis_nominal(next_genotypes, next_phenotype, covariates_t, dof)

            chr_res_df = []
            for i in range(1, igc.n_phenotypes+1):
                pval_nominal, slope, slope_se, maf, ma_samples, ma_count, phenotype_id = sess.run([pval_t, slope_t, slope_se_t, maf_t, ma_samples_t, ma_count_t, next_id])
                phenotype_id = phenotype_id.decode()
                r = igc.cis_ranges[phenotype_id]
                variant_ids = plink_reader.variant_pos[chrom].index[r[0]:r[1]+1]
                nv = len(variant_ids)
                tss_distance = np.int32(plink_reader.variant_pos[chrom].values[r[0]:r[1]+1] - igc.phenotype_tss[phenotype_id])

                chr_res_df.append(pd.DataFrame(OrderedDict([
                    ('phenotype_id', [phenotype_id]*nv),
                    ('variant_id', variant_ids),
                    ('tss_distance', tss_distance),
                    ('maf', maf),
                    ('ma_samples', np.int32(ma_samples)),
                    ('ma_count', np.int32(ma_count)),
                    ('pval_nominal', pval_nominal),
                    ('slope', slope),
                    ('slope_se', slope_se),
                ])))
                print('\r    computing associations for phenotype {}/{}'.format(i, igc.n_phenotypes), end='')
            print()
            print('  * writing output')
            pd.concat(chr_res_df, copy=False).to_parquet(os.path.join(output_dir, '{}.variant_phenotype_pairs.{}.parquet'.format(prefix, chrom)))
    logger.write('done.')


def map_trans(genotype_df, phenotype_df, covariates_df, interaction_s=None, return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05, batch_size=20000, logger=None):
    """Run trans-QTL mapping from genotypes in memory"""
    if logger is None:
        logger = SimpleLogger()
    assert np.all(phenotype_df.columns==covariates_df.index)

    variant_ids = genotype_df.index.tolist()
    variant_dict = {i:j for i,j in enumerate(variant_ids)}
    n_variants = len(variant_ids)
    n_samples = phenotype_df.shape[1]

    logger.write('trans-QTL mapping')
    logger.write('  * {} samples'.format(n_samples))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(n_variants))
    if interaction_s is not None:
        logger.write('  * including interaction term')

    # with tf.device('/cpu:0'):
    ggt = genotypeio.GenotypeGeneratorTrans(genotype_df.values, batch_size=batch_size, dtype=np.float32)
    dataset_genotypes = tf.data.Dataset.from_generator(ggt.generate_data, output_types=tf.float32)
    dataset_genotypes = dataset_genotypes.prefetch(10)
    iterator = dataset_genotypes.make_one_shot_iterator()
    next_element = iterator.get_next()

    # index of VCF samples corresponding to phenotypes
    ix_t = get_sample_indexes(genotype_df.columns.tolist(), phenotype_df)
    next_element = tf.gather(next_element, ix_t, axis=1)

    # calculate correlation threshold for sparse output
    if return_sparse:
        dof = n_samples - 2 - covariates_df.shape[1]
        t = stats.t.ppf(pval_threshold/2, dof)**2 / dof
        r2_threshold = t / (1+t)
    else:
        r2_threshold = None

    if interaction_s is None:
        genotypes, phenotypes, covariates = initialize_data(phenotype_df, covariates_df, batch_size=batch_size, dtype=tf.float32)
        # with tf.device('/gpu:0'):
        p_values, maf = calculate_association(genotypes, phenotypes, covariates, return_sparse=return_sparse, r2_threshold=r2_threshold)
    else:
        genotypes, phenotypes, covariates, interaction = initialize_data(phenotype_df, covariates_df, batch_size=batch_size, interaction_s=interaction_s)
        p_values, maf = calculate_association(genotypes, phenotypes, covariates, interaction_t=interaction, return_sparse=return_sparse, r2_threshold=r2_threshold)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start_time = time.time()
    print('  Mapping batches')
    with tf.Session() as sess:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # writer = tf.summary.FileWriter('logs', sess.graph, session=sess)
        sess.run(init_op)
        pval_list = []
        maf_list = []
        for i in range(1, ggt.num_batches+1):
            sys.stdout.write('\r  * processing batch {}/{}'.format(i, ggt.num_batches))
            sys.stdout.flush()

            g_iter = sess.run(next_element)
            p_ = sess.run([p_values, maf], feed_dict={genotypes:g_iter})#, options=run_options, run_metadata=run_metadata)
            # writer.add_run_metadata(run_metadata, 'batch{}'.format(i))

            pval_list.append(p_[0])
            maf_list.append(p_[1])

        if return_sparse:
            pval = tf.sparse_concat(0, pval_list).eval()
        else:
            pval = tf.concat(pval_list, 0).eval()
        maf = tf.concat(maf_list, 0).eval()
        print()
        # writer.close()

    end_time = time.time()
    logger.write('    time elapsed: {:.2f} min'.format((end_time-start_time)/60))

    if return_sparse:
        ix = pval.indices[:,0]<n_variants  # truncate last batch
        v = [variant_dict[i] for i in pval.indices[ix,0]]
        if phenotype_df.shape[0]>1:
            phenotype_ids = phenotype_df.index[pval.indices[ix,1]]
        else:
            phenotype_ids = phenotype_df.index.tolist()*len(pval.values)
        pval_df = pd.DataFrame(
            np.array([v, phenotype_ids, pval.values[ix], maf[ix]]).T,
            columns=['variant_id', 'phenotype_id', 'pval', 'maf']
        )
        pval_df['pval'] = pval_df['pval'].astype(np.float64)
        pval_df['maf'] = pval_df['maf'].astype(np.float32)
    else:
        # truncate last batch
        pval = pval[:n_variants]
        maf = maf[:n_variants]
        # add indices
        pval_df = pd.DataFrame(pval, index=variant_ids, columns=[i for i in phenotype_df.index])
        pval_df['maf'] = maf
        pval_df.index.name = 'variant_id'

    if maf_threshold is not None and maf_threshold>0:
        logger.write('  * filtering output by MAF >= {}'.format(maf_threshold))
        pval_df = pval_df[pval_df['maf']>=maf_threshold]

    logger.write('done.')
    return pval_df


def map_trans_tfrecord(vcf_tfrecord, phenotype_df, covariates_df, interaction_s=None, return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05, batch_size=50000, logger=None):
    """Run trans-QTL mapping from genotypes in tfrecord"""
    if logger is None:
        logger = SimpleLogger()
    assert np.all(phenotype_df.columns==covariates_df.index)

    with open(vcf_tfrecord+'.samples') as f:
        vcf_sample_ids = f.read().strip().split('\n')
    n_samples_vcf = len(vcf_sample_ids)

    with gzip.open(vcf_tfrecord+'.variants.gz', 'rt') as f:
        variant_ids = f.read().strip().split('\n')
    variant_dict = {i:j for i,j in enumerate(variant_ids)}
    n_variants = len(variant_ids)

    # index of VCF samples corresponding to phenotypes
    ix_t = get_sample_indexes(vcf_sample_ids, phenotype_df)
    n_samples = phenotype_df.shape[1]

    # batched_dataset = dataset.apply(tf.contrib.data.padded_batch(batch_size, padded_shapes=[[batch_size], [None]]))
    # batched_dataset = dataset.padded_batch(batch_size, padded_shapes=(batch_size,n_samples), padding_values=0)
    with tf.device('/cpu:0'):
        batched_dataset = genotypeio.make_batched_dataset(vcf_tfrecord, batch_size, ix_t=ix_t)

        iterator = batched_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        next_element = genotypeio.pad_up_to(next_element, [batch_size, n_samples])  # not clear if right/best way to do this

    logger.write('  * {} samples'.format(n_samples))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(n_variants))
    if interaction_s is not None:
        logger.write('  * including interaction term')

    num_batches = int(np.ceil(np.true_divide(n_variants, batch_size)))

    # calculate correlation threshold for sparse output
    if return_sparse:
        dof = n_samples - 2 - covariates_df.shape[1]
        t = stats.t.ppf(pval_threshold/2, dof)**2 / dof
        r2_threshold = t / (1+t)
    else:
        r2_threshold = None

    if interaction_s is None:
        genotypes, phenotypes, covariates = initialize_data(phenotype_df, covariates_df, batch_size=batch_size)
        with tf.device('/gpu:0'):
            p_values, maf = calculate_association(genotypes, phenotypes, covariates, return_sparse=return_sparse, r2_threshold=r2_threshold)
    else:
        genotypes, phenotypes, covariates, interaction = initialize_data(phenotype_df, covariates_df, batch_size=batch_size, interaction_s=interaction_s)
        p_values, maf = calculate_association(genotypes, phenotypes, covariates, interaction_t=interaction, return_sparse=return_sparse, r2_threshold=r2_threshold)

    # g = _parse_function(next_element, batch_size, n_samples, ix_t)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start_time = time.time()
    with tf.Session() as sess:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        # writer = tf.summary.FileWriter('logs', sess.graph, session=sess)
        sess.run(init_op)
        pval_list = []
        maf_list = []
        for i in range(num_batches):
            sys.stdout.write('\rProcessing batch {}/{}'.format(i+1, num_batches))
            sys.stdout.flush()

            g_iter = sess.run(next_element)
            # g_iter = sess.run(g)
            p_ = sess.run([p_values, maf], feed_dict={genotypes:g_iter})#, options=run_options, run_metadata=run_metadata)
            # writer.add_run_metadata(run_metadata, 'batch%d' % i)

            pval_list.append(p_[0])
            maf_list.append(p_[1])

        if return_sparse:
            pval = tf.sparse_concat(0, pval_list).eval()
        else:
            pval = tf.concat(pval_list, 0).eval()
        maf = tf.concat(maf_list, 0).eval()
        print()
        # writer.close()

    end_time = time.time()
    logger.write('Time elapsed: {:.2f} min'.format((end_time-start_time)/60))

    if return_sparse:
        ix = pval.indices[:,0]<n_variants  # truncate last batch
        v = [variant_dict[i] for i in pval.indices[ix,0]]
        pval_df = pd.DataFrame(
            np.array([v, phenotype_df.index[pval.indices[ix,1]], pval.values[ix], maf[ix]]).T,
            columns=['variant_id', 'phenotype_id', 'pval', 'maf']
        )
        pval_df['pval'] = pval_df['pval'].astype(np.float64)
        pval_df['maf'] = pval_df['maf'].astype(np.float32)
    else:
        # truncate last batch
        pval = pval[:n_variants]
        maf = maf[:n_variants]
        # add indices
        pval_df = pd.DataFrame(pval, index=variant_ids, columns=[i for i in phenotype_df.index])
        pval_df['maf'] = maf
        pval_df.index.name = 'variant_id'

    if maf_threshold is not None and maf_threshold>0:
        logger.write('  * filtering output by MAF >= {}'.format(maf_threshold))
        pval_df = pval_df[pval_df['maf']>=maf_threshold]

    return pval_df


def _in_cis(chrom, pos, gene_id, tss_dict, window=1000000):
    """Test if a variant-gene pair is in cis"""
    if chrom==tss_dict[gene_id]['chr']:
        tss = tss_dict[gene_id]['tss']
        if pos>=tss-window and pos<=tss+window:
            return True
        else:
            return False
    else:
        return False


def filter_cis(pval_df, tss_dict, window=1000000):
    """Filter out cis-QTLs"""
    drop_ix = []
    for k,gene_id,variant_id in zip(pval_df['phenotype_id'].index, pval_df['phenotype_id'], pval_df['variant_id']):
        chrom, pos = variant_id.split('_',2)[:2]
        pos = int(pos)
        if _in_cis(chrom, pos, gene_id, tss_dict, window=window):
            drop_ix.append(k)
    return pval_df.drop(drop_ix)

#------------------------------------------------------------------------------
#  Input parsers
#------------------------------------------------------------------------------
def read_phenotype_bed(phenotype_bed):
    """Load phenotype BED file as phenotype and TSS DataFrames"""
    phenotype_df = pd.read_csv(phenotype_bed, sep='\t', index_col=3, dtype={'#chr':str, '#Chr':str})
    phenotype_df = phenotype_df.rename(columns={i:i.lower() for i in phenotype_df.columns[:3]})
    phenotype_pos_df = phenotype_df[['#chr', 'end']].rename(columns={'#chr':'chr', 'end':'tss'})
    phenotype_df = phenotype_df.drop(['#chr', 'start', 'end'], axis=1)
    return phenotype_df, phenotype_pos_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run TensorQTL')
    parser.add_argument('genotype_path', help='Genotypes/dosages stored in PLINK or tfrecord format')
    parser.add_argument('phenotype_bed', help='Phenotypes in BED format')
    parser.add_argument('prefix', help='Prefix for output file names')
    parser.add_argument('--mode', default='cis', choices=['cis', 'cis_nominal', 'trans'])
    parser.add_argument('--covariates', default=None, help='Covariates [covariates x samples]')
    parser.add_argument('--permutations', default=10000, help='Number of permutations. Default: 10000')
    parser.add_argument('--interaction', default=None, type=str, help='Interaction term')
    parser.add_argument('--window', default=1000000, type=np.int32, help='Cis-window size, in bases. Default: 1Mb')
    parser.add_argument('--pval_threshold', default=None, type=np.float64, help='Output only significant phenotype-variant pairs with a p-value below threshold. Default: 1e-5 for trans-QTL')
    parser.add_argument('--maf_threshold', default=None, type=np.float64, help='Include only genotypes with minor allele frequency >=maf_threshold. Default: 0')
    parser.add_argument('--return_dense', action='store_true', help='Return dense output for trans-QTL')
    parser.add_argument('--output_text', action='store_true', help='Write output in txt.gz format instead of parquet (trans-QTL mode only)')
    parser.add_argument('--batch_size', type=int, default=50000, help='Batch size')
    # parser.add_argument('--fdr', default=0.05, type=np.float64)
    # parser.add_argument('--qvalue_lambda', default=None, help='lambda parameter for pi0est in qvalue.')
    parser.add_argument('-o', '--output_dir', default='.', help='Output directory')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = SimpleLogger(os.path.join(args.output_dir, args.prefix+'.tensorQTL.log'))
    logger.write('[{}] Running TensorQTL: {}-QTL mapping'.format(datetime.now().strftime("%b %d %H:%M:%S"), args.mode.split('_')[0]))

    # load inputs
    logger.write('  * reading phenotypes ({})'.format(args.phenotype_bed))
    phenotype_df, phenotype_pos_df = read_phenotype_bed(args.phenotype_bed)

    tss_dict = phenotype_pos_df.T.to_dict()
    if args.covariates is not None:
        logger.write('  * reading covariates ({})'.format(args.covariates))
        covariates_df = pd.read_csv(args.covariates, sep='\t', index_col=0).T
        assert np.all(phenotype_df.columns==covariates_df.index)
    if args.interaction is not None:
        logger.write('  * reading interaction term ({})'.format(args.interaction))
        interaction_s = pd.read_csv(args.interaction, sep='\t', index_col=0, header=None, squeeze=True)
        assert np.all(interaction_s.index==covariates_df.index)
    else:
        interaction_s = None

    if args.maf_threshold is None:
        if args.mode=='trans':
            maf_threshold = 0.05
        else:
            maf_threshold = 0
    else:
        maf_threshold = args.maf_threshold

    if args.mode=='cis' or args.mode=='cis_nominal':
        pr = genotypeio.PlinkReader(args.genotype_path, select_samples=phenotype_df.columns)
        if args.mode=='cis':
            res_df = map_cis(pr, phenotype_df, phenotype_pos_df, covariates_df, nperm=args.permutations, logger=logger)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir, args.prefix+'.cis_qtl_phenotypes.txt.gz')
            res_df.to_csv(out_file, sep='\t', float_format='%.6g', compression='gzip')
        else:
            map_cis_nominal(pr, phenotype_df, phenotype_pos_df, covariates_df, args.prefix,
                            output_dir=args.output_dir, interaction_s=interaction_s, logger=logger)
    elif args.mode=='trans':
        return_sparse = not args.return_dense
        pval_threshold = args.pval_threshold
        if pval_threshold is None and return_sparse:
            pval_threshold = 1e-5
            logger.write('  * p-value threshold: {:.2g}'.format(pval_threshold))

        genotype_df = genotypeio.load_genotypes(args.genotype_path)
        pval_df = map_trans(genotype_df, phenotype_df, covariates_df, interaction_s=interaction_s,
                  return_sparse=return_sparse, pval_threshold=pval_threshold,
                  maf_threshold=maf_threshold, batch_size=args.batch_size, logger=logger)

        logger.write('  * filtering out cis-QTLs')
        pval_df = filter_cis(pval_df, tss_dict, window=1000000)

        logger.write('  * writing output')
        if not args.output_text:
            pval_df.to_parquet(os.path.join(args.output_dir, args.prefix+'.trans_qtl_pairs.parquet'))
        else:
            out_file = os.path.join(args.output_dir, args.prefix+'.trans_qtl_pairs.txt.gz')
            pval_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g', compression='gzip')

    logger.write('[{}] Finished mapping'.format(datetime.now().strftime("%b %d %H:%M:%S")))

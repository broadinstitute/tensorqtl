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


def map_trans(genotype_df, phenotype_df, covariates_df, interaction_s=None,
              return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05,
              alleles=2, return_r2=False, batch_size=20000,
              logger=None, verbose=True):
    """Run trans-QTL mapping from genotypes in memory"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger(verbose=verbose)
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

    phenotypes_t = torch.tensor(phenotype_df.values, dtype=torch.float32).to(device)
    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    residualizer = Residualizer(covariates_t)
    del covariates_t

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # calculate correlation threshold for sparse output
    dof = n_samples - 2 - covariates_df.shape[1]
    if return_sparse:
        tstat_threshold = -stats.t.ppf(pval_threshold/2, dof)
        r2_threshold = tstat_threshold**2 / (dof + tstat_threshold**2)
    else:
        tstat_threshold = None
        r2_threshold = None

    if interaction_s is None:
        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size, dtype=np.int8)
        start_time = time.time()
        res = []
        n_variants = 0
        for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            # TODO: add imputation?

            # filter by MAF
            genotypes_t, variant_ids, maf_t = filter_maf(genotypes_t[:, genotype_ix_t], variant_ids, maf_threshold)
            n_variants += genotypes_t.shape[0]

            r2_t = calculate_corr(genotypes_t, phenotypes_t, residualizer).pow(2)
            del genotypes_t

            maf = maf_t.cpu()
            if return_sparse:
                m = r2_t >= r2_threshold
                r2_t = r2_t.masked_select(m)
                ix = m.nonzero().cpu()  # sparse index
                r2_t = r2_t.type(torch.float64)
                tstat = torch.sqrt(dof * r2_t / (1 - r2_t))
                if not return_r2:
                    res.append(np.c_[variant_ids[ix[:,0]], phenotype_df.index[ix[:,1]], tstat.cpu(), maf[ix[:,0]]])
                else:
                    res.append(np.c_[variant_ids[ix[:,0]], phenotype_df.index[ix[:,1]], tstat.cpu(), maf[ix[:,0]], r2_t.cpu()])
            else:
                r2_t = r2_t.type(torch.float64)
                tstat = torch.sqrt(dof * r2_t / (1 - r2_t))
                res.append(tstat)
        logger.write('    elapsed time: {:.2f} min'.format((time.time()-start_time)/60))
        del phenotypes_t
        del residualizer

        # post-processing: concatenate batches
        res = np.concatenate(res)
        if return_sparse:
            res[:,2] = 2*stats.t.cdf(-np.abs(res[:,2].astype(np.float64)), dof)
            cols = ['variant_id', 'phenotype_id', 'pval', 'maf']
            if return_r2:
                cols += ['r2']
            pval_df = pd.DataFrame(res, columns=cols)
            pval_df['pval'] = pval_df['pval'].astype(np.float64)
            pval_df['maf'] = pval_df['maf'].astype(np.float32)
            if return_r2:
                pval_df['r2'] = pval_df['r2'].astype(np.float32)
        else:
            pval = 2*stats.t.cdf(-np.abs(res.astype(np.float64)), dof)
            pval_df = pd.DataFrame(pval, index=variant_ids, columns=phenotype_df.index)
            pval_df.index.name = 'variant_id'

        logger.write('  * {} variants passed MAF >= {:.2f} filtering'.format(n_variants, maf_threshold))
        logger.write('done.')
        return pval_df

    else:  # interaction model
        dof = n_samples - 4 - covariates_df.shape[1]
        interaction_t = torch.tensor(interaction_s.values.reshape(1,-1), dtype=torch.float32).to(device)
        interaction_mask_t = torch.ByteTensor(interaction_s >= interaction_s.median()).to(device)

        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size, dtype=np.float32)
        start_time = time.time()
        if return_sparse:
            tstat_g_list = []
            tstat_i_list = []
            tstat_gi_list = []
            maf_list = []
            ix0 = []
            ix1 = []
            for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=verbose), 1):
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                res = calculate_interaction_nominal(genotypes_t[:, genotype_ix_t], phenotypes_t, interaction_t, residualizer,
                                                    interaction_mask_t=interaction_mask_t,
                                                    maf_threshold_interaction=maf_threshold,
                                                    return_sparse=return_sparse,
                                                    tstat_threshold=tstat_threshold)
                # res: tstat_g, tstat_i, tstat_gi, maf, mask, ix
                tstat_g, tstat_i, tstat_gi, maf, mask, ix = [i.cpu().numpy() for i in res]
                # convert sparse indexes
                if len(ix)>0:
                    variant_ids = variant_ids[mask.astype(bool)]
                    tstat_g_list.append(tstat_g)
                    tstat_i_list.append(tstat_i)
                    tstat_gi_list.append(tstat_gi)
                    maf_list.append(maf)
                    ix0.extend(variant_ids[ix[:,0]].tolist())
                    ix1.extend(phenotype_df.index[ix[:,1]].tolist())

            logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

            # concatenate
            pval_g =  2*stats.t.cdf(-np.abs(np.concatenate(tstat_g_list)), dof)
            pval_i =  2*stats.t.cdf(-np.abs(np.concatenate(tstat_i_list)), dof)
            pval_gi = 2*stats.t.cdf(-np.abs(np.concatenate(tstat_gi_list)), dof)
            maf = np.concatenate(maf_list)

            pval_df = pd.DataFrame(np.c_[ix0, ix1, pval_g, pval_i, pval_gi, maf], 
                                   columns=['variant_id', 'phenotype_id', 'pval_g', 'pval_i', 'pval_gi', 'maf']
                                   ).astype({'pval_g':np.float64, 'pval_i':np.float64, 'pval_gi':np.float64, 'maf':np.float32})
            return pval_df
        else:  # dense output
            output_list = []
            for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=verbose), 1):
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                res = calculate_interaction_nominal(genotypes_t[:, genotype_ix_t], phenotypes_t, interaction_t, residualizer,
                                                    interaction_mask_t=interaction_mask_t,
                                                    maf_threshold_interaction=maf_threshold,
                                                    return_sparse=return_sparse)
                # res: tstat, b, b_se, maf, ma_samples, ma_count, mask
                res = [i.cpu().numpy() for i in res]
                res[-1] = variant_ids[res[-1].astype(bool)]
                output_list.append(res)
            logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

            # concatenate outputs
            tstat = np.concatenate([i[0] for i in output_list])
            pval = 2*stats.t.cdf(-np.abs(tstat), dof)
            b = np.concatenate([i[1] for i in output_list])
            b_se = np.concatenate([i[2] for i in output_list])
            maf = np.concatenate([i[3] for i in output_list])
            ma_samples = np.concatenate([i[4] for i in output_list])
            ma_count = np.concatenate([i[5] for i in output_list])
            variant_ids = np.concatenate([i[6] for i in output_list])

            pval_g_df = pd.DataFrame(pval[:,0,:], index=variant_ids, columns=phenotype_df.index)
            pval_i_df = pd.DataFrame(pval[:,1,:], index=variant_ids, columns=phenotype_df.index)
            pval_gi_df = pd.DataFrame(pval[:,2,:], index=variant_ids, columns=phenotype_df.index)
            maf_s = pd.Series(maf, index=variant_ids, name='maf').astype(np.float32)
            ma_samples_s = pd.Series(ma_samples, index=variant_ids, name='maf').astype(np.int32)
            ma_count_s = pd.Series(ma_count, index=variant_ids, name='maf').astype(np.int32)
            return pval_g_df, pval_i_df, pval_gi_df, maf_s, ma_samples_s, ma_count_s


def map_permutations(genotype_df, covariates_df, permutations=None,
                     chr_s=None, nperms=10000, maf_threshold=0.05,
                     batch_size=20000, logger=None, seed=None, verbose=True):
    """


    Warning: this function assumes that all phenotypes are normally distributed,
             e.g., inverse normal transformed
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()
    assert covariates_df.index.isin(genotype_df.columns).all()
    sample_ids = covariates_df.index.values

    variant_ids = genotype_df.index.tolist()

    # index of VCF samples corresponding to phenotypes
    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in sample_ids])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    n_variants = len(variant_ids)
    n_samples = len(sample_ids)
    dof = n_samples - 2 - covariates_df.shape[1]

    logger.write('trans-QTL mapping (permutations)')
    logger.write('  * {} samples'.format(n_samples))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(n_variants))

    if permutations is None:  # generate permutations assuming normal distribution
        q = stats.norm.ppf(np.arange(1,n_samples+1)/(n_samples+1))
        qv = np.tile(q,[nperms,1])
        if seed is not None:
            np.random.seed(seed)
        for i in np.arange(nperms):
            np.random.shuffle(qv[i,:])
    else:
        assert permutations.shape[1]==n_samples
        nperms = permutations.shape[0]
        qv = permutations
        logger.write('  * {} permutations'.format(nperms))

    permutations_t = torch.tensor(permutations, dtype=torch.float32).to(device)
    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    residualizer = Residualizer(covariates_t)
    del covariates_t

    if chr_s is not None:
        start_time = time.time()
        n_variants = 0
        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size, chr_s=chr_s, dtype=np.float32)
        total_batches = np.sum([len(ggt.chr_batch_indexes[c]) for c in ggt.chroms])

        chr_max_r2 = OrderedDict()
        k = 0
        for chrom in ggt.chroms:
            max_r2_t = torch.FloatTensor(nperms).fill_(0).to(device)
            for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(chrom=chrom, verbose=verbose, enum_start=k+1), k+1):
                genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
                genotypes_t, variant_ids, maf_t = filter_maf(genotypes_t[:, genotype_ix_t], variant_ids, maf_threshold)
                n_variants += genotypes_t.shape[0]
                r2_t = calculate_corr(genotypes_t, permutations_t, residualizer).pow(2)
                del genotypes_t
                m,_ = r2_t.max(0)
                max_r2_t = torch.max(m, max_r2_t)
            chr_max_r2[chrom] = max_r2_t.cpu()
        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
        chr_max_r2 = pd.DataFrame(chr_max_r2)

        # leave-one-out max
        max_r2 = OrderedDict()
        for c in chr_max_r2:
            max_r2[c] = chr_max_r2[np.setdiff1d(chr_max_r2.columns, c)].max(1)
        max_r2 = pd.DataFrame(max_r2)  # nperms x chrs

        # empirical p-values
        tstat = np.sqrt( dof*max_r2 / (1-max_r2) )
        minp_empirical = pd.DataFrame(2*stats.t.cdf(-np.abs(tstat), dof), columns=tstat.columns)  # nperms x chrs

        beta_shape1 = OrderedDict()
        beta_shape2 = OrderedDict()
        true_dof = OrderedDict()
        minp_vec = OrderedDict()
        for c in max_r2:
            beta_shape1[c], beta_shape2[c], true_dof[c], minp_vec[c] = fit_beta_parameters(max_r2[c], dof, return_minp=True)

        beta_df = pd.DataFrame(OrderedDict([
            ('beta_shape1', beta_shape1),
            ('beta_shape2', beta_shape2),
            ('true_df', true_dof),
            ('minp_true_df', minp_vec),
            ('minp_empirical', {c:minp_empirical[c].values for c in minp_empirical}),
        ]))
        return beta_df

    else:  # not split_chr
        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df, batch_size=batch_size, dtype=np.float32)
        start_time = time.time()
        max_r2_t = torch.FloatTensor(nperms).fill_(0).to(device)
        n_variants = 0
        for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(verbose=verbose), 1):
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t, variant_ids, maf_t = filter_maf(genotypes_t[:, genotype_ix_t], variant_ids, maf_threshold)
            n_variants += genotypes_t.shape[0]
            r2_t = calculate_corr(genotypes_t, permutations_t, residualizer).pow(2)
            del genotypes_t
            m,_ = r2_t.max(0)
            max_r2_t = torch.max(m, max_r2_t)
        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))
        max_r2 = max_r2_t.cpu()
        tstat = np.sqrt( dof*max_r2 / (1-max_r2) )
        minp_empirical = 2*stats.t.cdf(-np.abs(tstat), dof)
        beta_shape1, beta_shape2, true_dof, minp_vec = fit_beta_parameters(max_r2, dof, tol=1e-4, return_minp=True)

        beta_s = pd.Series([n_samples, dof, beta_shape1, beta_shape2, true_dof, minp_vec, minp_empirical],
            index=['num_samples', 'df', 'beta_shape1', 'beta_shape2', 'true_df', 'minp_true_df', 'minp_empirical'])
        return beta_s


def apply_permutations(res, pairs_df):
    """
      res: output from map_permutations()
      pairs_df: output from map_trans()
    """

    if isinstance(res, pd.Series):  # chrs not split
        nperms = len(res['minp_true_df'])
        for k in ['beta_shape1', 'beta_shape2', 'true_df']:
            pairs_df[k] = res[k]
        pairs_df['pval_true_dof'] = pval_from_corr(pairs_df['r2'], pairs_df['true_df'])
        pairs_df['pval_perm'] = np.array([(np.sum(res['minp_empirical']<=p)+1)/(nperms+1) for p in pairs_df['pval']])
        pairs_df['pval_beta'] = stats.beta.cdf(pairs_df['pval_true_dof'], pairs_df['beta_shape1'], pairs_df['beta_shape2'])

    elif isinstance(res, pd.DataFrame):  #  chrs split
        nperms = len(res['minp_empirical'][0])
        for k in ['beta_shape1', 'beta_shape2', 'true_df']:
            pairs_df[k] = res.loc[pairs_df['phenotype_chr'], k].values
        pairs_df['pval_true_df'] = pval_from_corr(pairs_df['r2'], pairs_df['true_df'])
        pairs_df['pval_perm'] = [(np.sum(pe<=p)+1)/(nperms+1) for p,pe in zip(pairs_df['pval'], res.loc[pairs_df['phenotype_chr'], 'minp_empirical'])]
        # pval_perm = np.array([(np.sum(minp_empirical[chrom]<=p)+1)/(nperms+1) for p, chrom in zip(pval_df['pval'], pval_df['phenotype_chr'])])
        # pval_perm = np.array([(np.sum(minp_empirical<=p)+1)/(nperms+1) for p in minp_nominal])
        pairs_df['pval_beta'] = stats.beta.cdf(pairs_df['pval_true_df'], pairs_df['beta_shape1'], pairs_df['beta_shape2'])

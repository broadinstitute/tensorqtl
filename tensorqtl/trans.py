import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
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
        raise NotImplementedError 
        logger.write('  * including interaction term')

    # calculate correlation threshold for sparse output
    dof = n_samples - 2 - covariates_df.shape[1]
    if return_sparse:
        t = stats.t.ppf(pval_threshold/2, dof)**2 / dof
        r2_threshold = t / (1+t)
    else:
        r2_threshold = None

    phenotypes_t = torch.tensor(phenotype_df.values, dtype=torch.float32).cuda()
    covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).cuda()
    residualizer = Residualizer(covariates_t)

    ggt = genotypeio.GenotypeGeneratorTrans2(genotype_df, batch_size=batch_size, dtype=np.float32)
    start_time = time.time()
    res = []
    n_variants = 0
    for k, (genotypes, variant_ids) in enumerate(ggt.generate_data(), 1):
        if verbose:
            sys.stdout.write('\r  * processing batch {}/{}'.format(k, ggt.num_batches))
            sys.stdout.flush()

        # copy genotypes to GPU
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).cuda()

        # filter by MAF
        maf_t = calculate_maf(genotypes_t)
        mask_t = maf_t >= maf_threshold
        genotypes_t = genotypes_t[mask_t]
        variant_ids = variant_ids[mask_t.cpu().numpy().astype(bool)]
        maf_t = maf_t[mask_t]
        n_variants += genotypes_t.shape[0]

        r2_t = torch.pow(calculate_corr(genotypes_t, phenotypes_t, residualizer), 2)

        maf = maf_t.cpu()
        if return_sparse:
            m = r2_t >= r2_threshold
            r2_t = r2_t.masked_select(m)
            ix = m.nonzero().cpu()
            r2_t = r2_t.type(torch.float64)
            tstat = torch.sqrt(dof * r2_t / (1 - r2_t))
            if not return_r2:
                res.append(np.c_[variant_ids[ix[:,0]], phenotype_df.index[ix[:,1]], tstat.cpu(), maf[ix[:,0]]])
            else:
                res.append(np.c_[variant_ids[ix[:,0]], phenotype_df.index[ix[:,1]], tstat.cpu(), maf[ix[:,0]], r2_t.cpu()])
        else:
            tstat = torch.sqrt(dof * r2_t / (1 - r2_t))
            res.append(tstat)
    print()
    logger.write('    elapsed time: {:.2f} min'.format((time.time()-start_time)/60))

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

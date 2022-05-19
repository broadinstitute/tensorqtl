from bdb import Breakpoint
from types import DynamicClassAttribute
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os
import time
from collections import OrderedDict

from patsy import dmatrices
import six

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio, eigenmt
from core import *

import importlib as imp
import core
imp.reload(core)
from core import *
imp.reload(eigenmt)

def calculate_cis_nominal(genotypes_t, phenotype_t, residualizer=None, return_af=True):
    """
    Calculate nominal associations

    genotypes_t: genotypes x samples
    phenotype_t: single phenotype
    residualizer: Residualizer object (see core.py)
    """
    p = phenotype_t.reshape(1,-1)
    r_nominal_t, genotype_var_t, phenotype_var_t = calculate_corr(genotypes_t, p, residualizer=residualizer, return_var=True)
    std_ratio_t = torch.sqrt(phenotype_var_t.reshape(1,-1) / genotype_var_t.reshape(-1,1))
    r_nominal_t = r_nominal_t.squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    if residualizer is not None:
        dof = residualizer.dof
    else:
        dof = p.shape[1] - 2
    slope_t = r_nominal_t * std_ratio_t.squeeze()
    tstat_t = r_nominal_t * torch.sqrt(dof / (1 - r2_nominal_t))
    slope_se_t = (slope_t.double() / tstat_t).float()
    # tdist = tfp.distributions.StudentT(np.float64(dof), loc=np.float64(0.0), scale=np.float64(1.0))
    # pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat)))

    if return_af:
        af_t, ma_samples_t, ma_count_t = get_allele_stats(genotypes_t)
        return tstat_t, slope_t, slope_se_t, af_t, ma_samples_t, ma_count_t
    else:
        return tstat_t, slope_t, slope_se_t


def calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                               residualizer=None, random_tiebreak=False):
    """Calculate nominal and empirical correlations"""
    permutations_t = phenotype_t[permutation_ix_t]

    r_nominal_t, genotype_var_t, phenotype_var_t = calculate_corr(genotypes_t, phenotype_t.reshape(1,-1),
                                                                  residualizer=residualizer, return_var=True)
    std_ratio_t = torch.sqrt(phenotype_var_t.reshape(1,-1) / genotype_var_t.reshape(-1,1))
    r_nominal_t = r_nominal_t.squeeze(dim=-1)
    std_ratio_t = std_ratio_t.squeeze(dim=-1)
    corr_t = calculate_corr(genotypes_t, permutations_t, residualizer=residualizer).pow(2)  # genotypes x permutations
    corr_t = corr_t[~torch.isnan(corr_t).any(1),:]
    if corr_t.shape[0] == 0:
        raise ValueError('All correlations resulted in NaN. Please check phenotype values.')
    r2_perm_t,_ = corr_t.max(0)  # maximum correlation across permutations

    r2_nominal_t = r_nominal_t.pow(2)
    r2_nominal_t[torch.isnan(r2_nominal_t)] = -1  # workaround for nanargmax()
    if not random_tiebreak:
        ix = r2_nominal_t.argmax()
    else:
        ix = torch.nonzero(r2_nominal_t == r2_nominal_t.max(), as_tuple=True)[0]
        ix = ix[torch.randint(0, len(ix), [1])[0]]
    return r_nominal_t[ix], std_ratio_t[ix], ix, r2_perm_t, genotypes_t[ix]


def calculate_association(genotype_df, phenotype_s, covariates_df=None,
                          interaction_s=None, maf_threshold_interaction=0.05,
                          window=1000000, verbose=True):
    """
    Standalone helper function for computing the association between
    a set of genotypes and a single phenotype.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert genotype_df.columns.equals(phenotype_s.index)

    # copy to GPU
    phenotype_t = torch.tensor(phenotype_s.values, dtype=torch.float).to(device)
    genotypes_t = torch.tensor(genotype_df.values, dtype=torch.float).to(device)
    impute_mean(genotypes_t)

    dof = phenotype_s.shape[0] - 2

    if covariates_df is not None:
        assert phenotype_s.index.equals(covariates_df.index)
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        dof -= covariates_df.shape[1]
    else:
        residualizer = None

    if interaction_s is None:
        res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer)
        tstat, slope, slope_se, af, ma_samples, ma_count = [i.cpu().numpy() for i in res]
        df = pd.DataFrame({
            'pval_nominal':2*stats.t.cdf(-np.abs(tstat), dof),
            'slope':slope, 'slope_se':slope_se,
            'tstat':tstat, 'af':af, 'ma_samples':ma_samples, 'ma_count':ma_count,
        }, index=genotype_df.index)
    else:
        interaction_t = torch.tensor(interaction_s.values.reshape(1,-1), dtype=torch.float32).to(device)
        if maf_threshold_interaction > 0:
            mask_s = pd.Series(True, index=interaction_s.index)
            mask_s[interaction_s.sort_values(kind='mergesort').index[:interaction_s.shape[0]//2]] = False
            interaction_mask_t = torch.BoolTensor(mask_s).to(device)
        else:
            interaction_mask_t = None

        genotypes_t, mask_t = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t,
                                                     maf_threshold_interaction=maf_threshold_interaction)
        res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), interaction_t, residualizer,
                                            return_sparse=False)
        tstat, b, b_se, af, ma_samples, ma_count = [i.cpu().numpy() for i in res]
        mask = mask_t.cpu().numpy()
        dof -= 2

        df = pd.DataFrame({
            'pval_g':2*stats.t.cdf(-np.abs(tstat[:,0]), dof), 'b_g':b[:,0], 'b_g_se':b_se[:,0],
            'pval_i':2*stats.t.cdf(-np.abs(tstat[:,1]), dof), 'b_i':b[:,1], 'b_i_se':b_se[:,1],
            'pval_gi':2*stats.t.cdf(-np.abs(tstat[:,2]), dof), 'b_gi':b[:,2], 'b_gi_se':b_se[:,2],
            'af':af, 'ma_samples':ma_samples, 'ma_count':ma_count,
        }, index=genotype_df.index[mask])
    if df.index.str.startswith('chr').all():  # assume chr_pos_ref_alt_build format
        df['position'] = df.index.map(lambda x: int(x.split('_')[1]))
    return df


def map_nominal_interactions(genotype_df, variant_df, phenotype_df, phenotype_pos_df, 
                            sample_info_df, formula='p ~ g - 1', covariates_df=None, 
                            center=False, window=1_000_000, batch_size=500, write_output=False,
                            prefix='qtl', output_dir='.', logger=None, verbose=True, debug_n=-1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL interaction mapping: nominal associations for all variant-phenotype pairs')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    
    dof = phenotype_df.shape[0]
    if covariates_df is not None:
        assert np.all(phenotype_df.columns==covariates_df.index)
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        dof = residualizer.dof
    else:
        residualizer = None
        dof = phenotype_df.shape[1]

    logger.write(f'  * {variant_df.shape[0]} variants')

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    design_mat = dmatrices(formula, data=sample_info_df.assign(p=1, g=1))[1]
    design_info = design_mat.design_info

    ni = len(design_info.column_names)

    g_interaction_terms = np.zeros(ni, dtype=bool)
    categorical_terms = np.zeros(ni, dtype=bool)

    # todo: clean this code up
    for term_name, span in six.iteritems(design_info.term_name_slices):
        if (term_name == 'g') or ('g:' in term_name):
            g_interaction_terms[span.start:span.stop] = True
 
        if (':' not in term_name) and ('C(' in term_name):
            categorical_terms[span.start:span.stop] = True
            
    # print(f'Genotype interaction columns: {g_interaction_terms}')
    # print(f'Categorical terms columns: {categorical_terms}')

    categorial_terms_t = torch.tensor(categorical_terms, dtype=torch.bool).to(device)
    g_interaction_terms_t = torch.tensor(g_interaction_terms, dtype=torch.bool).to(device)

    # design matrix template
    design_t = torch.tensor(np.asarray(design_mat), dtype=torch.float32).to(device)
    filter_term_mask_t = design_t[...,categorial_terms_t].bool()

    logger.write(f'  * {formula}')
    logger.write(f'     - {len(design_info.term_names)} terms')
    logger.write(f'  * {ni} interactions:')
    for term in design_info.column_names:
        t = term.replace('T.', '')
        logger.write(f'     - {t}')

    dfe = design_mat.shape[0] - ni
    dfm = ni - 1
    if residualizer is not None:
        dfe -= residualizer.n
        dfm += residualizer.n

    start_time = time.time()

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    k = 0
    res_dfs = []

    logger.write('  * Computing associations')
    for chrom in igc.chrs:
        logger.write(f'    Mapping chromosome {chrom}')

        n = 0  # number of pairs
        for i in igc.phenotype_pos_df[igc.phenotype_pos_df['chr'] == chrom].index:
            j = igc.cis_ranges[i]
            n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['tss_distance'] = np.empty(n, dtype=np.int32)
        chr_res['af'] =           np.empty(n, dtype=np.float32)
        chr_res['ma_samples'] =   np.empty(n, dtype=np.int32)
        chr_res['ma_count'] =     np.empty(n, dtype=np.int32)
        chr_res['pval_i'] =       np.empty([n, ni], dtype=np.float64)
        chr_res['b_i'] =          np.empty([n, ni], dtype=np.float32)
        chr_res['b_se_i'] =       np.empty([n, ni], dtype=np.float32)
        chr_res['f'] =            np.empty(n, dtype=np.float32)
        chr_res['r2'] =           np.empty(n, dtype=np.float32)
        chr_res['sse'] =          np.empty(n, dtype=np.float32)
        chr_res['dfe'] =          np.empty(n, dtype=np.float32)
        chr_res['m_eff'] =        np.empty(n, dtype=np.int32)

        res_start = 0 # start of phenotype
        res_chunk_start = 0
        res_chunk_end = 0

        k = 0
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):

            res_start = res_chunk_start

            # copy phenotype to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float32).to(device)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
            tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])

            # Chunk genotypes
            chunked_genotypes = [genotypes[i:i+batch_size,:] for i in range(0, len(genotypes), batch_size)]

            chunk_start = 0
            for chunk in chunked_genotypes:
                
                chunk_size = len(chunk)

                # copy genotypes to GPU
                genotypes_t = torch.tensor(chunk, dtype=torch.float32).to(device)
                genotypes_t = genotypes_t[:,genotype_ix_t]
                impute_mean(genotypes_t)

                chunk_variant_ids = variant_ids[chunk_start:chunk_start+chunk_size]
                chunk_tss_distance = tss_distance[chunk_start:chunk_start+chunk_size]

                assert len(chunk_variant_ids) == len(chunk)
        
                # filter variants here
                genotypes_t, mask_t = filter_term_samples(genotypes_t, filter_term_mask_t, device=device)

                if genotypes_t.shape[0] > 0:

                    mask = mask_t.cpu().numpy().astype(bool)
                    chunk_variant_ids = chunk_variant_ids[mask]
                    chunk_tss_distance = chunk_tss_distance[mask]

                    try:
                        # res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), design_t,
                        #                                 g_idx_t, g_interaction_terms_t, terms_to_residualize_t,
                        #                                 residualizer=residualizer, variant_ids=variant_ids)
                        res = calculate_interaction_nominal(genotypes_t, phenotype_t.unsqueeze(0), design_t,
                                                        g_interaction_terms_t, center=center,
                                                        residualizer=residualizer, variant_ids=variant_ids)
                        sse, f, r2, tstat, b, b_se, af, ma_samples, ma_count = [i.cpu().numpy() for i in res]

                        res_chunk_end = res_chunk_start + len(chunk_variant_ids)

                        chr_res['phenotype_id'].extend([phenotype_id]*len(chunk_variant_ids))
                        chr_res['variant_id'].extend(chunk_variant_ids)
                        chr_res['tss_distance'][res_chunk_start:res_chunk_end] = chunk_tss_distance
                        chr_res['af'][res_chunk_start:res_chunk_end] = af
                        chr_res['ma_samples'][res_chunk_start:res_chunk_end] = ma_samples
                        chr_res['ma_count'][res_chunk_start:res_chunk_end] = ma_count
                        chr_res['pval_i'][res_chunk_start:res_chunk_end]  = tstat[...,0]
                        chr_res['b_i'][res_chunk_start:res_chunk_end]     = b[...,0]
                        chr_res['b_se_i'][res_chunk_start:res_chunk_end]  = b_se[...,0]
                        chr_res['f'][res_chunk_start:res_chunk_end]  = f[...,0]
                        chr_res['r2'][res_chunk_start:res_chunk_end]  = r2[...,0]
                        chr_res['sse'][res_chunk_start:res_chunk_end]  = sse[...,0]
                        chr_res['dfe'][res_chunk_start:res_chunk_end] = dfe
                        
                        res_chunk_start = res_chunk_end
                    
                    except Exception as e:
                        print(e)

                chunk_start = chunk_start + chunk_size
            
            # compte m_eff
            if res_chunk_end-res_start>0:

                genotypes_t = torch.tensor(genotypes, dtype=torch.float32).to(device)
                genotypes_t = genotypes_t[:,genotype_ix_t]
                impute_mean(genotypes_t)

                genotypes_t, mask_t = filter_term_samples(genotypes_t, filter_term_mask_t, device=device)

                chr_res['m_eff'][res_start:res_chunk_end] = eigenmt.compute_tests(genotypes_t, var_thresh=0.99, variant_window=200)

            if debug_n>0 and k>debug_n:
                break

        logger.write(f'    time elapsed: {(time.time()-start_time)/60:.2f} min')

        # convert to dataframe, compute p-values and write current chromosome
        for x in chr_res:
            chr_res[x] = chr_res[x][:res_chunk_end]
        
        # split columns
        cols = ['pval_i', 'b_i', 'b_se_i']
        for i in range(0, ni):  # fix order
            for k in cols:
                chr_res[k.replace('i', f"i{i}")] = None
        for k in cols:
            for i in range(0, ni):
                chr_res[k.replace('i', f"i{i}")] = chr_res[k][:,i]
            del chr_res[k]

        chr_res_df = pd.DataFrame(chr_res)

        for i in range(0, ni):
            chr_res_df.loc[:, f'pval_i{i}'] =  2*stats.t.cdf(-chr_res_df.loc[:, f'pval_i{i}'].abs(), dfe)

        chr_res_df = chr_res_df.assign(pval_f=stats.f.sf(chr_res_df.loc[:, 'f'], dfm, dfe))

        var_dict = []
        for v, i in six.iteritems(design_info.column_name_indexes):
            t = v.replace('T.', '')
            for c in ['pval_i', 'b_i', 'b_se_i']:
                var_dict.append((c.replace('_i', f'_i{i}'), c.replace('_i', f'_{t}')))
        var_dict = dict(var_dict)

        # substitute column headers
        chr_res_df.rename(columns=var_dict, inplace=True)
        
        if write_output:
            print('    * writing output')
            chr_res_df.to_parquet(os.path.join(output_dir, f'{prefix}.cis_qtl_pairs.{chrom}.parquet'))
        else:
            res_dfs.append(chr_res_df)
     
    logger.write('done.')

    if not write_output:
        return pd.concat(res_dfs, axis=0)



def map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, prefix='qtl',
                covariates_df=None, maf_threshold=0, window=1_000_000, 
                output_dir='.', logger=None, verbose=True):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs
    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    If interaction_df is provided, the top association per phenotype is
    written to <output_dir>/<prefix>.cis_qtl_top_assoc.txt.gz unless
    write_top is set to False, in which case it is returned as a DataFrame
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    if covariates_df is not None:
        assert np.all(phenotype_df.columns==covariates_df.index)
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        dof = residualizer.dof - 2
    else:
        residualizer = None
        dof = phenotype_df.shape[1] - 2
    logger.write(f'  * {variant_df.shape[0]} variants')

    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
    # iterate over chromosomes
    start_time = time.time()
    k = 0
    logger.write('  * Computing associations')

    for chrom in igc.chrs:
        logger.write(f'    Mapping chromosome {chrom}')
        # allocate arrays
        n = 0  # number of pairs

        for i in igc.phenotype_pos_df[igc.phenotype_pos_df['chr'] == chrom].index:
            j = igc.cis_ranges[i]
            n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['tss_distance'] = np.empty(n, dtype=np.int32)
        chr_res['af'] =           np.empty(n, dtype=np.float32)
        chr_res['ma_samples'] =   np.empty(n, dtype=np.int32)
        chr_res['ma_count'] =     np.empty(n, dtype=np.int32)
        chr_res['pval_nominal'] = np.empty(n, dtype=np.float64)
        chr_res['slope'] =        np.empty(n, dtype=np.float32)
        chr_res['slope_se'] =     np.empty(n, dtype=np.float32)
    
        start = 0
    
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(chrom=chrom, verbose=verbose), k+1):
            # copy genotypes to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
            tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igc.phenotype_tss[phenotype_id])

            if maf_threshold > 0:
                maf_t = calculate_maf(genotypes_t)
                mask_t = maf_t >= maf_threshold
                genotypes_t = genotypes_t[mask_t]
                mask = mask_t.cpu().numpy().astype(bool)
                variant_ids = variant_ids[mask]
                tss_distance = tss_distance[mask]

            if genotypes_t.shape[0] > 0:
                res = calculate_cis_nominal(genotypes_t, phenotype_t, residualizer=residualizer)
                tstat, slope, slope_se, af, ma_samples, ma_count = [i.cpu().numpy() for i in res]
                n = len(variant_ids)
            else:
                n = 0

            if n > 0:
                chr_res['phenotype_id'].extend([phenotype_id]*n)
                chr_res['variant_id'].extend(variant_ids)
                chr_res['tss_distance'][start:start+n] = tss_distance
                chr_res['af'][start:start+n] = af
                chr_res['ma_samples'][start:start+n] = ma_samples
                chr_res['ma_count'][start:start+n] = ma_count
                chr_res['pval_nominal'][start:start+n] = tstat
                chr_res['slope'][start:start+n] = slope
                chr_res['slope_se'][start:start+n] = slope_se
            
            start += n  # update pointer

        logger.write(f'    time elapsed: {(time.time()-start_time)/60:.2f} min')

        # convert to dataframe, compute p-values and write current chromosome
        if start < len(chr_res['af']):
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]
        
        chr_res_df = pd.DataFrame(chr_res)
        m = chr_res_df['pval_nominal'].notnull()
        chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
    
        print('    * writing output')
        chr_res_df.to_parquet(os.path.join(output_dir, f'{prefix}.cis_qtl_pairs.{chrom}.parquet'))
        
    logger.write('done.')

def prepare_cis_output(r_nominal, r2_perm, std_ratio, g, num_var, dof, variant_id, tss_distance, phenotype_id, nperm=10000):
    """Return nominal p-value, allele frequencies, etc. as pd.Series"""
    r2_nominal = r_nominal*r_nominal
    pval_perm = (np.sum(r2_perm>=r2_nominal)+1) / (nperm+1)

    slope = r_nominal * std_ratio
    tstat2 = dof * r2_nominal / (1 - r2_nominal)
    slope_se = np.abs(slope) / np.sqrt(tstat2)

    n2 = 2*len(g)
    af = np.sum(g) / n2
    if af <= 0.5:
        ma_samples = np.sum(g>0.5)
        ma_count = np.sum(g[g>0.5])
    else:
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
        ('af', af),
        ('pval_nominal', pval_from_corr(r2_nominal, dof)),
        ('slope', slope),
        ('slope_se', slope_se),
        ('pval_perm', pval_perm),
        ('pval_beta', np.NaN),
    ]), name=phenotype_id)
    return res_s


def _process_group_permutations(buf, variant_df, tss, dof, group_id, nperm=10000, beta_approx=True):
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
    if beta_approx:
        res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof*0.25)
    res_s['group_id'] = group_id
    res_s['group_size'] = len(buf)
    return res_s


def map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df=None,
            group_s=None, maf_threshold=0, beta_approx=True, nperm=10000,
            window=1000000, random_tiebreak=False, logger=None, seed=None,
            verbose=True, warn_monomorphic=True):
    """Run cis-QTL mapping"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: empirical p-values for phenotypes')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    if group_s is not None:
        logger.write(f'  * {len(group_s.unique())} phenotype groups')
        group_dict = group_s.to_dict()
    if covariates_df is not None:
        assert np.all(phenotype_df.columns==covariates_df.index), 'Sample names in phenotype matrix columns and covariate matrix rows do not match!'
        assert ~(covariates_df.isnull().any().any()), f'Missing or null values in covariates matrix, in columns {",".join(covariates_df.columns[covariates_df.isnull().any(axis=0)].astype(str))}'
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]
    else:
        residualizer = None
        dof = phenotype_df.shape[1] - 2
    logger.write(f'  * {genotype_df.shape[0]} variants')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')
    if random_tiebreak:
        logger.write(f'  * randomly selecting top variant in case of ties')

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    if seed is not None:
        logger.write(f'  * using seed {seed}')
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(np.array([np.random.permutation(ix) for i in range(nperm)])).to(device)

    res_df = []
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    start_time = time.time()
    logger.write('  * computing permutations')
    if group_s is None:
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            if maf_threshold > 0:
                maf_t = calculate_maf(genotypes_t)
                mask_t = maf_t >= maf_threshold
                genotypes_t = genotypes_t[mask_t]
                mask = mask_t.cpu().numpy().astype(bool)
                genotype_range = genotype_range[mask]

            # filter monomorphic variants
            mono_t = (genotypes_t == genotypes_t[:, [0]]).all(1)
            if mono_t.any():
                genotypes_t = genotypes_t[~mono_t]
                genotype_range = genotype_range[~mono_t.cpu()]
                if warn_monomorphic:
                    logger.write(f'    * WARNING: excluding {mono_t.sum()} monomorphic variants')

            if genotypes_t.shape[0] == 0:
                logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
                continue

            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)

            res = calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                                             residualizer=residualizer, random_tiebreak=random_tiebreak)
            r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
            var_ix = genotype_range[var_ix]
            variant_id = variant_df.index[var_ix]
            tss_distance = variant_df['pos'].values[var_ix] - igc.phenotype_tss[phenotype_id]
            res_s = prepare_cis_output(r_nominal, r2_perm, std_ratio, g, genotypes_t.shape[0], dof, variant_id, tss_distance, phenotype_id, nperm=nperm)
            if beta_approx:
                res_s[['pval_beta', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df']] = calculate_beta_approx_pval(r2_perm, r_nominal*r_nominal, dof)
            res_df.append(res_s)
    else:  # grouped mode
        for k, (phenotypes, genotypes, genotype_range, phenotype_ids, group_id) in enumerate(igc.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            if maf_threshold > 0:
                maf_t = calculate_maf(genotypes_t)
                mask_t = maf_t >= maf_threshold
                genotypes_t = genotypes_t[mask_t]
                mask = mask_t.cpu().numpy().astype(bool)
                genotype_range = genotype_range[mask]

            # filter monomorphic variants
            mono_t = (genotypes_t == genotypes_t[:, [0]]).all(1)
            if mono_t.any():
                genotypes_t = genotypes_t[~mono_t]
                genotype_range = genotype_range[~mono_t.cpu()]
                if warn_monomorphic:
                    logger.write(f'    * WARNING: excluding {mono_t.sum()} monomorphic variants')

            if genotypes_t.shape[0] == 0:
                logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
                continue

            # iterate over phenotypes
            buf = []
            for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                res = calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                                                 residualizer=residualizer, random_tiebreak=random_tiebreak)
                res = [i.cpu().numpy() for i in res]  # r_nominal, std_ratio, var_ix, r2_perm, g
                res[2] = genotype_range[res[2]]
                buf.append(res + [genotypes_t.shape[0], phenotype_id])
            res_s = _process_group_permutations(buf, variant_df, igc.phenotype_tss[phenotype_ids[0]], dof,
                                                group_id, nperm=nperm, beta_approx=beta_approx)
            res_df.append(res_s)

    res_df = pd.concat(res_df, axis=1, sort=False).T
    res_df.index.name = 'phenotype_id'
    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    logger.write('done.')
    return res_df.astype(output_dtype_dict).infer_objects()


def map_independent(genotype_df, variant_df, cis_df, phenotype_df, phenotype_pos_df, covariates_df,
                    group_s=None, maf_threshold=0, fdr=0.05, fdr_col='qval', nperm=10000,
                    window=1000000, random_tiebreak=False, logger=None, seed=None, verbose=True):
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
        'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'af',
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
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    if group_s is None:
        logger.write(f'  * {signif_df.shape[0]}/{cis_df.shape[0]} significant phenotypes')
    else:
        logger.write(f'  * {signif_df.shape[0]}/{cis_df.shape[0]} significant groups')
        logger.write(f'    {len(ix)}/{phenotype_df.shape[0]} phenotypes')
        group_dict = group_s.to_dict()
    logger.write(f'  * {covariates_df.shape[1]} covariates')
    logger.write(f'  * {genotype_df.shape[0]} variants')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')
    if random_tiebreak:
        logger.write(f'  * randomly selecting top variant in case of ties')
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
        logger.write(f'  * using seed {seed}')
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(np.array([np.random.permutation(ix) for i in range(nperm)])).to(device)

    res_df = []
    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=group_s, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    logger.write('  * computing independent QTLs')
    start_time = time.time()
    if group_s is None:
        for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
            # copy genotypes to GPU
            phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
            genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
            genotypes_t = genotypes_t[:,genotype_ix_t]
            impute_mean(genotypes_t)

            if maf_threshold > 0:
                maf_t = calculate_maf(genotypes_t)
                mask_t = maf_t >= maf_threshold
                genotypes_t = genotypes_t[mask_t]
                mask = mask_t.cpu().numpy().astype(bool)
                genotype_range = genotype_range[mask]

            # 1) forward pass
            forward_df = [signif_df.loc[phenotype_id]]  # initialize results with top variant
            covariates = covariates_df.values.copy()  # initialize covariates
            dosage_dict = {}
            while True:
                # add variant to covariates
                variant_id = forward_df[-1]['variant_id']
                ig = genotype_df.values[ix_dict[variant_id], genotype_ix].copy()
                m = ig == -1
                ig[m] = ig[~m].mean()
                dosage_dict[variant_id] = ig
                covariates = np.hstack([covariates, ig.reshape(-1,1)]).astype(np.float32)
                dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                residualizer = Residualizer(torch.tensor(covariates, dtype=torch.float32).to(device))

                res = calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                                                 residualizer=residualizer, random_tiebreak=random_tiebreak)
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
                    residualizer = Residualizer(torch.tensor(covariates, dtype=torch.float32).to(device))

                    res = calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                                                     residualizer=residualizer, random_tiebreak=random_tiebreak)
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

            if maf_threshold > 0:
                maf_t = calculate_maf(genotypes_t)
                mask_t = maf_t >= maf_threshold
                genotypes_t = genotypes_t[mask_t]
                mask = mask_t.cpu().numpy().astype(bool)
                genotype_range = genotype_range[mask]

            # 1) forward pass
            forward_df = [signif_df[signif_df['group_id']==group_id].iloc[0]]  # initialize results with top variant
            covariates = covariates_df.values.copy()  # initialize covariates
            dosage_dict = {}
            while True:
                # add variant to covariates
                variant_id = forward_df[-1]['variant_id']
                ig = genotype_df.values[ix_dict[variant_id], genotype_ix].copy()
                m = ig == -1
                ig[m] = ig[~m].mean()
                dosage_dict[variant_id] = ig
                covariates = np.hstack([covariates, ig.reshape(-1,1)]).astype(np.float32)
                dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                residualizer = Residualizer(torch.tensor(covariates, dtype=torch.float32).to(device))

                # iterate over phenotypes
                buf = []
                for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                    phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                    res = calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                                                     residualizer=residualizer, random_tiebreak=random_tiebreak)
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
                    residualizer = Residualizer(torch.tensor(covariates, dtype=torch.float32).to(device))

                    # iterate over phenotypes
                    buf = []
                    for phenotype, phenotype_id in zip(phenotypes, phenotype_ids):
                        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
                        res = calculate_cis_permutations(genotypes_t, phenotype_t, permutation_ix_t,
                                                         residualizer=residualizer, random_tiebreak=random_tiebreak)
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
    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    logger.write('done.')
    return res_df.reset_index().astype(output_dtype_dict)

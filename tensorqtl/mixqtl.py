import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import os
from collections import OrderedDict
import torch
import tensorqtl
from tensorqtl import genotypeio, cis, SimpleLogger


class InputGeneratorMix(object):
    """
    Input generator for cis-mapping with mixQTL model

    Inputs:
      hap1_df:  genotype dosages for 1st haplotype (DataFrame; genotypes x samples)
      hap2_df:  genotype dosages for 2nd haplotype (DataFrame; genotypes x samples)
      variant_df:         DataFrame mapping variant_id (index) to chrom, pos
      log_counts_imp_df:  log-transformed, imputed read counts (phenotypes x samples)
      counts_df:          raw read counts (phenotypes x samples)
      ref_df:  
      alt_df:  
      phenotype_pos_df: DataFrame defining position of each phenotype, with columns 'chr' and 'tss'
      window:           cis-window (selects variants within +- cis-window from TSS)

    Generates: phenotype array, genotype array (2D), cis-window indices, phenotype ID
    """
    def __init__(self, hap1_df, hap2_df, variant_df, log_counts_imp_df, counts_df, ref_df, alt_df, phenotype_pos_df, window=1000000):
        assert (hap1_df.index==variant_df.index).all()
        assert np.all(counts_df.index==log_counts_imp_df.index)
        assert np.all(counts_df.index==ref_df.index)
        self.hap1_df = hap1_df
        self.hap2_df = hap2_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(variant_df.shape[0])
        self.n_samples = counts_df.shape[1]
        self.log_counts_imp_df = log_counts_imp_df
        self.counts_df = counts_df
        self.ref_df = ref_df
        self.alt_df = alt_df
        self.phenotype_pos_df = phenotype_pos_df
        # check for constant phenotypes and drop
        # m = np.all(phenotype_df.values == phenotype_df.values[:,[0]], 1)
        # if m.any():
        #     print('    ** dropping {} constant phenotypes'.format(np.sum(m)))
        #     self.phenotype_df = self.phenotype_df.loc[~m]
        #     self.phenotype_pos_df = self.phenotype_pos_df.loc[~m]
        self.group_s = None
        self.window = window

        self.n_phenotypes =  phenotype_pos_df.shape[0]
        self.phenotype_tss = phenotype_pos_df['tss'].to_dict()
        self.phenotype_chr = phenotype_pos_df['chr'].to_dict()
        self.chrs = phenotype_pos_df['chr'].unique()
        self.chr_variant_dfs = {c:g[['pos', 'index']] for c,g in self.variant_df.groupby('chrom')}

        # check phenotypes & calculate genotype ranges
        # get genotype indexes corresponding to cis-window of each phenotype
        valid_ix = []
        self.cis_ranges = {}
        for k,phenotype_id in enumerate(phenotype_pos_df.index,1):
            if np.mod(k, 1000) == 0:
                print('\r  * checking phenotypes: {}/{}'.format(k, phenotype_pos_df.shape[0]), end='')

            tss = self.phenotype_tss[phenotype_id]
            chrom = self.phenotype_chr[phenotype_id]
            # r = self.chr_variant_dfs[chrom]['index'].values[
            #     (self.chr_variant_dfs[chrom]['pos'].values >= tss - self.window) &
            #     (self.chr_variant_dfs[chrom]['pos'].values <= tss + self.window)
            # ]
            # r = [r[0],r[-1]]

            m = len(self.chr_variant_dfs[chrom]['pos'].values)
            lb = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, tss - self.window)
            ub = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, tss + self.window, side='right')
            if lb != ub:
                r = self.chr_variant_dfs[chrom]['index'].values[[lb, ub - 1]]
            else:
                r = []

            if len(r) > 0:
                valid_ix.append(phenotype_id)
                self.cis_ranges[phenotype_id] = r

        print('\r  * checking phenotypes: {}/{}'.format(k, phenotype_pos_df.shape[0]))
        if len(valid_ix)!=phenotype_pos_df.shape[0]:
            print('    ** dropping {} phenotypes without variants in cis-window'.format(
                  phenotype_pos_df.shape[0]-len(valid_ix)))
            self.log_counts_imp_df = self.log_counts_imp_df.loc[valid_ix]
            self.counts_df = self.counts_df.loc[valid_ix]
            self.ref_df = self.ref_df.loc[valid_ix]
            self.alt_df = self.alt_df.loc[valid_ix]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[valid_ix]
            self.n_phenotypes = self.phenotype_pos_df.shape[0]
            self.phenotype_tss = phenotype_pos_df['tss'].to_dict()
            self.phenotype_chr = phenotype_pos_df['chr'].to_dict()
        # if group_s is not None:
        #     self.group_s = group_s.loc[self.phenotype_df.index].copy()
        #     self.n_groups = self.group_s.unique().shape[0]


    @genotypeio.background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False):
        """
        Generate batches from genotype data

        Returns: phenotype array, genotype matrix, genotype index, phenotype ID(s), [group ID]
        """
        if chrom is None:
            phenotype_ids = self.phenotype_pos_df.index
            chr_offset = 0
        else:
            phenotype_ids = self.phenotype_pos_df[self.phenotype_pos_df['chr']==chrom].index
            if self.group_s is None:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'], return_index=True))}
            else:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'][self.group_s.drop_duplicates().index], return_index=True))}
            chr_offset = offset_dict[chrom]

        index_dict = {j:i for i,j in enumerate(self.phenotype_pos_df.index)}

        if self.group_s is None:
            for k,phenotype_id in enumerate(phenotype_ids, chr_offset+1):
                if verbose:
                    genotypeio.print_progress(k, self.n_phenotypes, 'phenotype')

                c0 = self.counts_df.values[index_dict[phenotype_id]]
                c = self.log_counts_imp_df.values[index_dict[phenotype_id]]
                ref = self.ref_df.values[index_dict[phenotype_id]]
                alt = self.alt_df.values[index_dict[phenotype_id]]
                r = self.cis_ranges[phenotype_id]
                yield c0, c, ref, alt, self.hap1_df.values[r[0]:r[-1]+1], self.hap2_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), phenotype_id

        else:
            raise NotImplementedError
            # gdf = self.group_s[phenotype_ids].groupby(self.group_s, sort=False)
            # for k,(group_id,g) in enumerate(gdf, chr_offset+1):
            #     if verbose:
            #         _print_progress(k, self.n_groups, 'phenotype group')
            #     # check that ranges are the same for all phenotypes within group
            #     assert np.all([self.cis_ranges[g.index[0]][0] == self.cis_ranges[i][0] and self.cis_ranges[g.index[0]][1] == self.cis_ranges[i][1] for i in g.index[1:]])
            #     group_phenotype_ids = g.index.tolist()
            #     # p = self.phenotype_df.loc[group_phenotype_ids].values
            #     p = self.phenotype_df.values[[index_dict[i] for i in group_phenotype_ids]]
            #     r = self.cis_ranges[g.index[0]]
            #     yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), group_phenotype_ids, group_id


def linreg(X_t, y_t):
    """
    Solve y = Xb, standardizing X

    The first column of X_t must be the intercept
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_std_t = X_t.std(axis=0)
    x_mean_t = X_t.mean(axis=0)
    x_std_t[0] = 1
    x_mean_t[0] = 0

    # standardize X
    Xtilde_t = (X_t - x_mean_t) / x_std_t
    XtX_t = torch.matmul(Xtilde_t.T, Xtilde_t)
    Xty_t = torch.matmul(Xtilde_t.T, y_t)
    b_t, _ = torch.solve(Xty_t.unsqueeze(-1), XtX_t)
    b_t = b_t.squeeze()

    dof = X_t.shape[0] - X_t.shape[1]
    r_t = y_t - torch.matmul(Xtilde_t, b_t)
    sigma2_t = (r_t*r_t).sum() / dof

    # compute s.e.
    XtX_inv_t, _ = torch.solve(torch.eye(X_t.shape[1]).to(device), XtX_t)
    var_b_t = sigma2_t * XtX_inv_t
    b_se_t = torch.sqrt(torch.diag(var_b_t))

    # rescale
    b_t = b_t / x_std_t
    b_se_t = b_se_t / x_std_t

    # adjust intercept
    b_t[0] = b_t[0] - torch.sum(x_mean_t * b_t)
    ms_t = x_mean_t / x_std_t
    b_se_t[0] = torch.sqrt(b_se_t[0]*b_se_t[0] + torch.matmul(torch.matmul(ms_t.T, var_b_t), ms_t))

    return b_t, b_se_t


def filter_covariates(covariates0_t, log_counts_t, tstat_threshold=2):
    """
    Inputs:
      covariates0_t: covariates matrix (samples x covariates)
                     including genotype PCs, PEER factors, etc.
                     ** with intercept in first column **
      log_counts_t:  counts vector (samples)
    """
    b_t, b_se_t = linreg(covariates0_t, log_counts_t)
    tstat_t = b_t / b_se_t
    m = tstat_t.abs() > tstat_threshold
    m[0] = False
    return covariates0_t[:, m]


def trc_calc(genotypes_t, log_counts_t, raw_counts_t, covariates0_t,
             count_threshold=100, select_covariates=True):
    """
    Inputs:
      genotypes_t:  genotype dosages (variants x samples)
      log_counts_t: log(counts/(2*libsize)) --> TODO: use size factor normalized counts instead
      raw_counts_t: raw RNA-seq counts
      covariates0_t: covariates matrix (samples x covariates)
                     including genotype PCs, PEER factors, etc.
                     ** with intercept in first column **
      count_threshold: minimum read count to include a sample
    """
    # # only use samples that pass count threshold
    # mask_t = raw_counts_t >= count_threshold
    # if select_covariates:
    #     covariates_t = filter_covariates(covariates0_t[mask_t], log_counts_t[mask_t])
    # else:
    #     covariates_t = covariates0_t[mask_t, 1:]
    #
    # residualizer = tensorqtl.Residualizer(covariates_t)
    # res = cis.calculate_cis_nominal(genotypes_t[:, mask_t] / 2, log_counts_t[mask_t].reshape(1,-1), residualizer)
    # # [tstat, beta, beta_se, maf, ma_samples, ma_count], samples, dof
    # return res, covariates_t.shape[0], residualizer.dof

    if select_covariates:
        covariates_t = filter_covariates(covariates0_t, log_counts_t)
    else:
        covariates_t = covariates0_t[:, 1:]

    mask_t = raw_counts_t >= count_threshold
    residualizer = tensorqtl.Residualizer(covariates_t[mask_t])
    res = cis.calculate_cis_nominal(genotypes_t[:, mask_t] / 2, log_counts_t[mask_t].reshape(1,-1), residualizer)
    # [tstat, beta, beta_se, maf, ma_samples, ma_count], samples
    return res, int(mask_t.sum()), residualizer.dof


def asc_calc(hap1_t, hap2_t, ref_t, alt_t, ase_threshold=50, ase_max=1000, weight_cap=100):
    """
    Inputs:
      hap1_t: genotypes for haplotype 1 (variants x samples)
      hap2_t: genotypes for haplotype 2 (variants x samples)
      ref_t:  ASE counts for REF allele
      alt_t:  ASE counts for ALT allele
      ase_threshold: minimum read count for each allele to include a sample
      ase_max:       maximum read count for each allele to include a sample

    Only samples for which both REF and ALT counts are >= ase_threshold
    and <= ase_max are used in the calculation.

    Returns:
      [beta_t, beta_se_t]: beta, standard error on beta
      samples:             number of samples that passed filtering steps
      dof:                 degrees of freedom
    """

    mask_t = ((ref_t >= ase_threshold) &
              (alt_t >= ase_threshold) &
              (ref_t <= ase_max) &
              (alt_t <= ase_max))

    X_t = hap1_t - hap2_t
    X_t = X_t[:, mask_t]

    n = X_t.shape[1]
    if n > 0:
        mref_t = ref_t[mask_t]
        malt_t = alt_t[mask_t]
        y_t = torch.log(mref_t / malt_t)
        w2_t = 1 / (1/mref_t + 1/malt_t)
        weight_cutoff = torch.min(w2_t) * np.minimum(weight_cap, np.floor(X_t.shape[1] / 10))
        w2_t[w2_t > weight_cutoff] = weight_cutoff

        xwy_t = torch.einsum('ij->i', X_t * w2_t * y_t)  # faster than (X_t * w2_t * y_t).sum(1)
        xwx_t = torch.einsum('ij->i', X_t * w2_t * X_t)
        b_t = xwy_t / xwx_t

        y2_t = (y_t * w2_t * y_t).sum()
        rss_t = b_t*xwx_t*b_t - 2*b_t*xwy_t + y2_t
        dof = n - 1
        b_se_t = torch.sqrt(rss_t / xwx_t / dof)

        tstat_t = b_t / b_se_t

        return [tstat_t, b_t, b_se_t], n, dof
    else:
        return None, 0, np.NaN


def map_nominal(hap1_df, hap2_df, variant_df, log_counts_imp_df, counts_df, ref_df, alt_df,
                phenotype_pos_df, covariates_df, prefix, select_covariates=True,
                window=1000000, output_dir='.', write_stats=True, logger=None, verbose=True):
    """
    cis-QTL mapping: mixQTL model, nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.mixQTL.<chr>.parquet
    """
    assert np.all(counts_df.columns==covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(counts_df.shape[1]))
    logger.write('  * {} phenotypes'.format(counts_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(variant_df.shape[0]))

    genotype_ix = np.array([hap1_df.columns.tolist().index(i) for i in counts_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # add intercept to covariates
    covariates0_t = torch.tensor(np.c_[np.ones(covariates_df.shape[0]), covariates_df],
                                 dtype=torch.float32).to(device)

    igm = InputGeneratorMix(hap1_df, hap2_df, variant_df, log_counts_imp_df, counts_df, 
                            ref_df, alt_df, phenotype_pos_df, window=window)
    # iterate over chromosomes
    start_time = time.time()
    k = 0
    logger.write('  * Computing associations')
    for chrom in igm.chrs:
        logger.write('    Mapping chromosome {}'.format(chrom))
        # allocate arrays
        n = 0
        for i in igm.phenotype_pos_df[igm.phenotype_pos_df['chr']==chrom].index:
            j = igm.cis_ranges[i]
            n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['tss_distance'] =   np.empty(n, dtype=np.int32)
        chr_res['maf_trc'] =        np.empty(n, dtype=np.float32)
        chr_res['ma_samples_trc'] = np.empty(n, dtype=np.int32)
        chr_res['ma_count_trc'] =   np.empty(n, dtype=np.int32)
        chr_res['beta_trc'] =       np.empty(n, dtype=np.float32)
        chr_res['beta_se_trc'] =    np.empty(n, dtype=np.float32)
        chr_res['tstat_trc'] =      np.empty(n, dtype=np.float32)
        chr_res['pval_trc'] =       np.empty(n, dtype=np.float64)
        chr_res['samples_trc'] =    np.empty(n, dtype=np.int32)
        chr_res['dof_trc'] =        np.empty(n, dtype=np.int32)
        chr_res['beta_asc'] =       np.empty(n, dtype=np.float32)
        chr_res['beta_se_asc'] =    np.empty(n, dtype=np.float32)
        chr_res['tstat_asc'] =      np.empty(n, dtype=np.float32)
        chr_res['pval_asc'] =       np.empty(n, dtype=np.float64)
        chr_res['samples_asc'] =    np.empty(n, dtype=np.int32)
        chr_res['dof_asc'] =        np.empty(n, dtype=np.int32)
        chr_res['beta_meta'] =      np.empty(n, dtype=np.float32)
        chr_res['beta_se_meta'] =   np.empty(n, dtype=np.float32)
        chr_res['tstat_meta'] =     np.empty(n, dtype=np.float32)
        chr_res['pval_meta'] =      np.empty(n, dtype=np.float64)

        start = 0
        for k, (raw_counts, log_counts, ref, alt, hap1, hap2, genotype_range, phenotype_id) in enumerate(igm.generate_data(chrom=chrom, verbose=verbose), k+1):
            # copy data to GPU
            hap1_t = torch.tensor(hap1, dtype=torch.float).to(device)
            hap2_t = torch.tensor(hap2, dtype=torch.float).to(device)
            # subset samples
            hap1_t = hap1_t[:,genotype_ix_t]
            hap2_t = hap2_t[:,genotype_ix_t]

            ref_t = torch.tensor(ref, dtype=torch.float).to(device)
            alt_t = torch.tensor(alt, dtype=torch.float).to(device)
            raw_counts_t = torch.tensor(raw_counts, dtype=torch.float).to(device)
            log_counts_t = torch.tensor(log_counts, dtype=torch.float).to(device)

            genotypes_t = hap1_t + hap2_t
            genotypes_t[genotypes_t==-2] = -1
            tensorqtl.impute_mean(genotypes_t)
            tensorqtl.impute_mean(hap1_t)
            tensorqtl.impute_mean(hap2_t)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1]
            tss_distance = np.int32(variant_df['pos'].values[genotype_range[0]:genotype_range[-1]+1] - igm.phenotype_tss[phenotype_id])

            res_trc, samples_trc, dof_trc = trc_calc(genotypes_t, log_counts_t, raw_counts_t,
                                                     covariates0_t, count_threshold=100, select_covariates=select_covariates)
            # res = [tstat, beta, beta_se, maf, ma_samples, ma_count]

            res_asc, samples_asc, dof_asc = asc_calc(hap1_t, hap2_t, ref_t, alt_t)
            # res = [tstat_t, beta_t, beta_se_t]

            n = len(variant_ids)
            [tstat_trc, beta_trc, beta_se_trc, maf_trc, ma_samples_trc, ma_count_trc] = res_trc

            chr_res['phenotype_id'].extend([phenotype_id]*n)
            chr_res['variant_id'].extend(variant_ids)
            chr_res['tss_distance'][start:start+n] = tss_distance
            chr_res['maf_trc'][start:start+n] = maf_trc.cpu().numpy()
            chr_res['ma_samples_trc'][start:start+n] = ma_samples_trc.cpu().numpy()
            chr_res['ma_count_trc'][start:start+n] = ma_count_trc.cpu().numpy()
            chr_res['beta_trc'][start:start+n] = beta_trc.cpu().numpy()
            chr_res['beta_se_trc'][start:start+n] = beta_se_trc.cpu().numpy()
            chr_res['tstat_trc'][start:start+n] = tstat_trc.cpu().numpy()
            chr_res['samples_trc'][start:start+n] = samples_trc
            chr_res['dof_trc'][start:start+n] = dof_trc

            if res_asc is not None:
                [tstat_asc, beta_asc, beta_se_asc] = res_asc
                chr_res['beta_asc'][start:start+n] = beta_asc.cpu().numpy()
                chr_res['beta_se_asc'][start:start+n] = beta_se_asc.cpu().numpy()
                chr_res['tstat_asc'][start:start+n] = tstat_asc.cpu().numpy()
                chr_res['samples_asc'][start:start+n] = samples_asc
                chr_res['dof_asc'][start:start+n] = dof_asc
                # meta-analysis
                d = 1/beta_se_trc**2 + 1/beta_se_asc**2
                beta_meta_t = (beta_asc/beta_se_asc**2 + beta_trc/beta_se_trc**2) / d
                beta_se_meta_t = 1 / torch.sqrt(d)
                tstat_meta_t = beta_meta_t / beta_se_meta_t
                chr_res['beta_meta'][start:start+n] = beta_meta_t.cpu().numpy()
                chr_res['beta_se_meta'][start:start+n] = beta_se_meta_t.cpu().numpy()
                chr_res['tstat_meta'][start:start+n] = tstat_meta_t.cpu().numpy()

            start += n  # update pointer

        logger.write('    time elapsed: {:.2f} min'.format((time.time()-start_time)/60))

        # convert to dataframe, compute p-values and write current chromosome
        if start < len(chr_res['maf_trc']):
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]

        if write_stats:
            chr_res_df = pd.DataFrame(chr_res)
            # torch.distributions.StudentT.cdf is still not implemented --> use scipy
            # m = chr_res_df['pval_nominal'].notnull()
            # chr_res_df.loc[m, 'pval_nominal'] = 2*stats.t.cdf(-chr_res_df.loc[m, 'pval_nominal'].abs(), dof)
            chr_res_df['pval_trc'] = 2*stats.t.cdf(-chr_res_df['tstat_trc'].abs(), chr_res_df['dof_trc'])
            chr_res_df['pval_asc'] = 2*stats.t.cdf(-chr_res_df['tstat_asc'].abs(), chr_res_df['dof_asc'])
            chr_res_df['pval_meta'] = 2*stats.norm.cdf(-chr_res_df['tstat_meta'].abs())

            print('    * writing output')
            chr_res_df.to_parquet(os.path.join(output_dir, '{}.cis_qtl_pairs.mixQTL.{}.parquet'.format(prefix, chrom)))

    logger.write('done.')


def calculate_replication(res_df, hap1_df, hap2_df, log_counts_imp_df, counts_df, ref_df, alt_df,
                          covariates_df, select_covariates=True, count_threshold=100, lambda_qvalue=None):
    """res_df: DataFrame with 'variant_id' column and phenotype IDs as index"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genotype_ix = np.array([hap1_df.columns.tolist().index(i) for i in counts_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # add intercept to covariates
    covariates0_t = torch.tensor(np.c_[np.ones(covariates_df.shape[0]), covariates_df],
                                 dtype=torch.float32).to(device)

    # copy data to GPU
    hap1_t = torch.tensor(hap1_df.loc[res_df['variant_id']].values, dtype=torch.float).to(device)
    hap2_t = torch.tensor(hap2_df.loc[res_df['variant_id']].values, dtype=torch.float).to(device)
    hap1_t = hap1_t[:,genotype_ix_t]
    hap2_t = hap2_t[:,genotype_ix_t]

    ref_t = torch.tensor(ref_df.loc[res_df.index].values, dtype=torch.float).to(device)
    alt_t = torch.tensor(alt_df.loc[res_df.index].values, dtype=torch.float).to(device)
    raw_counts_t = torch.tensor(counts_df.loc[res_df.index].values, dtype=torch.float).to(device)
    log_counts_t = torch.tensor(log_counts_imp_df.loc[res_df.index].values, dtype=torch.float).to(device)

    genotypes_t = hap1_t + hap2_t
    genotypes_t[genotypes_t==-2] = -1
    tensorqtl.impute_mean(genotypes_t)
    tensorqtl.impute_mean(hap1_t)
    tensorqtl.impute_mean(hap2_t)

    res = OrderedDict()
    n = res_df.shape[0]
    res['phenotype_id'] =   res_df.index.values
    res['variant_id'] =     res_df['variant_id'].values
    res['maf_trc'] =        np.empty(n, dtype=np.float32)
    res['ma_samples_trc'] = np.empty(n, dtype=np.int32)
    res['ma_count_trc'] =   np.empty(n, dtype=np.int32)
    res['beta_trc'] =       np.empty(n, dtype=np.float32)
    res['beta_se_trc'] =    np.empty(n, dtype=np.float32)
    res['tstat_trc'] =      np.empty(n, dtype=np.float32)
    res['pval_trc'] =       np.empty(n, dtype=np.float64)
    res['samples_trc'] =    np.empty(n, dtype=np.int32)
    res['dof_trc'] =        np.empty(n, dtype=np.int32)
    res['beta_asc'] =       np.empty(n, dtype=np.float32)
    res['beta_se_asc'] =    np.empty(n, dtype=np.float32)
    res['tstat_asc'] =      np.empty(n, dtype=np.float32)
    res['pval_asc'] =       np.empty(n, dtype=np.float64)
    res['samples_asc'] =    np.empty(n, dtype=np.int32)
    res['dof_asc'] =        np.empty(n, dtype=np.int32)
    res['beta_meta'] =      np.empty(n, dtype=np.float32)
    res['beta_se_meta'] =   np.empty(n, dtype=np.float32)
    res['tstat_meta'] =     np.empty(n, dtype=np.float32)
    res['pval_meta'] =      np.empty(n, dtype=np.float64)

    # due to covariate selection and masking, each pair needs to run separately
    for i in range(res_df.shape[0]):
        print('\r{}/{}'.format(i+1, res_df.shape[0]), end='')
        res_trc, samples_trc, dof_trc = trc_calc(genotypes_t[[i],:], log_counts_t[i,:], raw_counts_t[i,:], covariates0_t,
                                                 count_threshold=count_threshold, select_covariates=select_covariates)
        res_asc, samples_asc, dof_asc = asc_calc(hap1_t[[i],:], hap2_t[[i],:], ref_t[i,:], alt_t[i,:])

        res_trc = [i.cpu().numpy().squeeze() for i in res_trc]
        [tstat_trc, beta_trc, beta_se_trc, maf_trc, ma_samples_trc, ma_count_trc] = res_trc
        res['maf_trc'][i] = maf_trc
        res['ma_samples_trc'][i] = ma_samples_trc
        res['ma_count_trc'][i] = ma_count_trc
        res['beta_trc'][i] = beta_trc
        res['beta_se_trc'][i] = beta_se_trc
        res['tstat_trc'][i] = tstat_trc
        res['samples_trc'][i] = samples_trc
        res['dof_trc'][i] = dof_trc

        if res_asc is not None:
            res_asc = [i.cpu().numpy().squeeze() for i in res_asc]
            [tstat_asc, beta_asc, beta_se_asc] = res_asc
            res['beta_asc'][i] = beta_asc
            res['beta_se_asc'][i] = beta_se_asc
            res['tstat_asc'][i] = tstat_asc
            res['samples_asc'][i] = samples_asc
            res['dof_asc'][i] = dof_asc
            # meta-analysis
            d = 1/beta_se_trc**2 + 1/beta_se_asc**2
            beta_meta = (beta_asc/beta_se_asc**2 + beta_trc/beta_se_trc**2) / d
            beta_se_meta = 1 / np.sqrt(d)
            tstat_meta = beta_meta / beta_se_meta
            res['beta_meta'][i] = beta_meta
            res['beta_se_meta'][i] = beta_se_meta
            res['tstat_meta'][i] = tstat_meta

        rep_df = pd.DataFrame(res)
        rep_df['pval_trc'] = 2*stats.t.cdf(-rep_df['tstat_trc'].abs(), rep_df['dof_trc'])
        rep_df['pval_asc'] = 2*stats.t.cdf(-rep_df['tstat_trc'].abs(), rep_df['dof_asc'])
        rep_df['pval_meta'] = 2*stats.norm.cdf(-rep_df['tstat_meta'].abs())
    return rep_df

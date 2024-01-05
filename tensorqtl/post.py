import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import subprocess
import sys
import os
import glob
from datetime import datetime

sys.path.insert(1, os.path.dirname(__file__))
from core import SimpleLogger, Residualizer, center_normalize, impute_mean, get_allele_stats
import mixqtl
import qtl.genotype as gt


has_rpy2 = False
try:
    subprocess.check_call('which R', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call("R -e 'library(qvalue)'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import rpy2
    import rfunc
    has_rpy2 = True
except:
    print("Warning: 'rfunc' cannot be imported. R with the 'qvalue' library, and the 'rpy2' Python package are needed.")


def calculate_qvalues(res_df, fdr=0.05, qvalue_lambda=None, logger=None):
    """Annotate permutation results with q-values, p-value threshold"""
    if logger is None:
        logger = SimpleLogger()

    logger.write('Computing q-values')
    logger.write(f'  * Number of phenotypes tested: {res_df.shape[0]}')

    if not res_df['pval_beta'].isnull().all():
        pval_col = 'pval_beta'
        r = stats.pearsonr(res_df['pval_perm'], res_df['pval_beta'])[0]
        logger.write(f'  * Correlation between Beta-approximated and empirical p-values: {r:.4f}')
    else:
        pval_col = 'pval_perm'
        logger.write(f'  * WARNING: no beta-approximated p-values found, using permutation p-values instead.')

    # calculate q-values
    if qvalue_lambda is not None:
        logger.write(f'  * Calculating q-values with lambda = {qvalue_lambda:.3f}')
    qval, pi0 = rfunc.qvalue(res_df[pval_col], lambda_qvalue=qvalue_lambda)

    res_df['qval'] = qval
    logger.write(f'  * Proportion of significant phenotypes (1-pi0): {1-pi0:.2f}')
    logger.write(f"  * QTL phenotypes @ FDR {fdr:.2f}: {(res_df['qval'] <= fdr).sum()}")

    # determine global min(p) significance threshold and calculate nominal p-value threshold for each gene
    if pval_col == 'pval_beta':
        lb = res_df.loc[res_df['qval'] <= fdr, 'pval_beta'].sort_values()
        ub = res_df.loc[res_df['qval'] > fdr, 'pval_beta'].sort_values()

        if len(lb) > 0:  # significant phenotypes
            lb = lb.iloc[-1]
            if len(ub) > 0:
                ub = ub.iloc[0]
                pthreshold = (lb+ub)/2
            else:
                pthreshold = lb
            logger.write(f'  * min p-value threshold @ FDR {fdr}: {pthreshold:.6g}')
            res_df['pval_nominal_threshold'] = stats.beta.ppf(pthreshold, res_df['beta_shape1'], res_df['beta_shape2'])


def calculate_afc(assoc_df, counts_df, genotype_df, variant_df=None, covariates_df=None,
                  select_covariates=True, group='gene_id',
                  imputation='offset', count_threshold=0, verbose=True):
    """
    Calculate allelic fold-change (aFC) for variant-gene pairs

    Inputs
      assoc_df: dataframe containing variant-gene associations, must have 'gene_id'
                and 'variant_id' columns. If multiple variants/gene are detected, effects
                are estimated jointly.
      genotype_df: genotype dosages
      counts_df: read counts scaled with DESeq size factors. Zeros are imputed using
                 log(counts + 1) (imputation='offset'; default) or with half-minimum
                 (imputation='half_min').
      covariates_df: covariates (genotype PCs, PEER factors, etc.)

    aFC [1] is computed using the total read count (trc) model from mixQTL [2].

      [1] Mohammadi et al., 2017 (genome.cshlp.org/content/27/11/1872)
      [2] Liang et al., 2021 (10.1038/s41467-021-21592-8)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variant_df is not None:
        gi = gt.GenotypeIndexer(genotype_df, variant_df)
    else:
        assert isinstance(genotype_df, gt.GenotypeIndexer)
        gi = genotype_df
    genotype_ix = np.array([gi.genotype_df.columns.tolist().index(i) for i in counts_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    if covariates_df is not None:
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    else:
        covariates_t = None

    afc_df = []
    n = len(assoc_df[group].unique())
    for k, (phenotype_id, gdf) in enumerate(assoc_df.groupby(group, sort=False), 1):
        if verbose and k % 10 == 0 or k == n:
            print(f"\rCalculating aFC for {group.replace('_id','')} {k}/{n}", end='' if k != n else None, flush=True)

        counts_t = torch.tensor(counts_df.loc[phenotype_id].values,
                                dtype=torch.float32).to(device)
        genotypes_t = torch.tensor(gi.get_genotypes(gdf['variant_id'].tolist()), dtype=torch.float32).to(device)
        genotypes_t = genotypes_t[:,genotype_ix_t]
        impute_mean(genotypes_t)
        try:
            b, b_se = mixqtl.trc(genotypes_t, counts_t, covariates_t=covariates_t,
                                 select_covariates=select_covariates, count_threshold=count_threshold,
                                 imputation=imputation, mode='multi', return_af=False)
            gdf['afc'] = b.cpu().numpy() * np.log2(np.e)
            gdf['afc_se'] = b_se.cpu().numpy() * np.log2(np.e)
            afc_df.append(gdf)
        except:
            print(f'WARNING: aFC calculation failed for {phenotype_id}')
    afc_df = pd.concat(afc_df)

    return afc_df


def calculate_replication(res_df, genotype_df, phenotype_df, covariates_df=None, paired_covariate_df=None,
                          interaction_s=None, compute_pi1=False, lambda_qvalue=None):
    """res_df: DataFrame with 'variant_id' column and phenotype IDs as index"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if paired_covariate_df is not None:
        assert paired_covariate_df.index.equals(covariates_df.index)
        assert paired_covariate_df.columns.isin(phenotype_df.index).all()

    genotypes_t = torch.tensor(genotype_df.loc[res_df['variant_id']].values, dtype=torch.float).to(device)
    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    genotypes_t = genotypes_t[:,genotype_ix_t]
    impute_mean(genotypes_t)
    af_t, ma_samples_t, ma_count_t = get_allele_stats(genotypes_t)

    phenotypes_t = torch.tensor(phenotype_df.loc[res_df.index].values, dtype=torch.float32).to(device)


    if covariates_df is not None:
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        # dof -= covariates_df.shape[1]
    else:
        residualizer = None

    if interaction_s is None:
        if paired_covariate_df is None:
            if residualizer is not None:
                genotype_res_t = residualizer.transform(genotypes_t)  # variants x samples
                phenotype_res_t = residualizer.transform(phenotypes_t)  # phenotypes x samples
                dof = residualizer.dof
                dof_t = dof
            else:
                genotype_res_t = genotypes_t
                phenotype_res_t = phenotypes_t
                dof =  phenotypes_t.shape[1] - 2
                dof_t = dof
        else:
            genotype_res_t = torch.zeros_like(genotypes_t).to(device)
            phenotype_res_t = torch.zeros_like(phenotypes_t).to(device)
            dof = []
            for k,phenotype_id in enumerate(res_df.index):
                if phenotype_id in paired_covariate_df:
                    iresidualizer = Residualizer(torch.tensor(np.c_[covariates_df, paired_covariate_df[phenotype_id]],
                                                              dtype=torch.float32).to(device))
                else:
                    iresidualizer = residualizer
                genotype_res_t[[k]] = iresidualizer.transform(genotypes_t[[k]])
                phenotype_res_t[[k]] = iresidualizer.transform(phenotypes_t[[k]])
                dof.append(iresidualizer.dof)
            dof = np.array(dof)
            dof_t = torch.Tensor(dof).to(device)

        gstd = genotype_res_t.var(1)
        pstd = phenotype_res_t.var(1)
        std_ratio_t = torch.sqrt(pstd / gstd)

        # center and normalize
        genotype_res_t = center_normalize(genotype_res_t, dim=1)
        phenotype_res_t = center_normalize(phenotype_res_t, dim=1)

        r_nominal_t = (genotype_res_t * phenotype_res_t).sum(1)
        r2_nominal_t = r_nominal_t.double().pow(2)

        tstat_t = torch.sqrt((dof_t * r2_nominal_t) / (1 - r2_nominal_t))
        slope_t = r_nominal_t * std_ratio_t
        slope_se_t = (slope_t.abs().double() / tstat_t).float()
        pval = 2*stats.t.cdf(-np.abs(tstat_t.cpu()), dof)

        rep_df = pd.DataFrame(np.c_[res_df.index, res_df['variant_id'], ma_samples_t.cpu(), ma_count_t.cpu(), af_t.cpu(), pval, slope_t.cpu(), slope_se_t.cpu()],
                              columns=['phenotype_id', 'variant_id', 'ma_samples', 'ma_count', 'af', 'pval_nominal', 'slope', 'slope_se']).infer_objects()

    else:
        if paired_covariate_df is not None:
            raise NotImplementedError("Paired covariates are not yet supported for interactions")

        interaction_t = torch.tensor(interaction_s.values.reshape(1,-1), dtype=torch.float32).to(device)
        ng, ns = genotypes_t.shape
        nps = phenotypes_t.shape[0]

        # centered inputs
        g0_t = genotypes_t - genotypes_t.mean(1, keepdim=True)
        gi_t = genotypes_t * interaction_t
        gi0_t = gi_t - gi_t.mean(1, keepdim=True)
        i0_t = interaction_t - interaction_t.mean()
        p0_t = phenotypes_t - phenotypes_t.mean(1, keepdim=True)

        # residualize rows
        g0_t = residualizer.transform(g0_t, center=False)
        gi0_t = residualizer.transform(gi0_t, center=False)
        p0_t = residualizer.transform(p0_t, center=False)  # np x ns
        i0_t = residualizer.transform(i0_t, center=False)
        i0_t = i0_t.repeat(ng, 1)

        # regression (in float; loss of precision may occur in edge cases)
        X_t = torch.stack([g0_t, i0_t, gi0_t], 2)  # ng x ns x 3
        Xinv = torch.matmul(torch.transpose(X_t, 1, 2), X_t).inverse() # ng x 3 x 3
        b_t = (torch.matmul(Xinv, torch.transpose(X_t, 1, 2)) * p0_t.unsqueeze(1)).sum(2) # ng x 3
        r_t = (X_t * b_t.unsqueeze(1)).sum(2) - p0_t
        dof = residualizer.dof - 2
        rss_t = (r_t*r_t).sum(1)  # ng x np
        b_se_t = torch.sqrt( Xinv[:, torch.eye(3, dtype=torch.uint8).bool()] * rss_t.unsqueeze(-1) / dof )
        tstat_t = (b_t.double() / b_se_t.double()).float()
        pval = 2*stats.t.cdf(-np.abs(tstat_t.cpu()), dof)
        b = b_t.cpu()
        b_se = b_se_t.cpu()

        rep_df = pd.DataFrame(np.c_[res_df.index, res_df['variant_id'], ma_samples_t.cpu(), ma_count_t.cpu(), af_t.cpu(),
                                    pval[:,0], b[:,0], b_se[:,0], pval[:,1], b[:,1], b_se[:,1], pval[:,2], b[:,2], b_se[:,2]],
                              columns=['phenotype_id', 'variant_id', 'ma_samples', 'ma_count', 'af',
                                       'pval_g', 'b_g', 'b_g_se', 'pval_i', 'b_i', 'b_i_se', 'pval_gi', 'b_gi', 'b_gi_se']).infer_objects()
        pval = pval[:,2]

    if compute_pi1:
        try:
            pi1 = 1 - rfunc.pi0est(pval, lambda_qvalue=lambda_qvalue)[0]
        except:
            pi1 = np.NaN
        return pi1, rep_df
    else:
        return rep_df


def annotate_genes(gene_df, annotation_gtf, lookup_df=None):
    """
    Add gene and variant annotations (e.g., gene_name, rs_id, etc.) to gene-level output

    gene_df:        output from map_cis()
    annotation_gtf: gene annotation in GTF format
    lookup_df:      DataFrame with variant annotations, indexed by 'variant_id'
    """
    gene_dict = {}
    print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] Adding gene and variant annotations', flush=True)
    print('  * parsing GTF', flush=True)
    with open(annotation_gtf) as gtf:
        for row in gtf:
            row = row.strip().split('\t')
            if row[0][0] == '#' or row[2] != 'gene': continue
            # get gene_id and gene_name from attributes
            attr = dict([i.split() for i in row[8].replace('"','').split(';') if i!=''])
            # gene_name, gene_chr, gene_start, gene_end, strand
            gene_dict[attr['gene_id']] = [attr['gene_name'], row[0], row[3], row[4], row[6]]

    print('  * annotating genes', flush=True)
    if 'group_id' in gene_df:
        gene_info = pd.DataFrame(data=[gene_dict[i] for i in gene_df['group_id']],
                                 columns=['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand'],
                                 index=gene_df.index)
    else:
        gene_info = pd.DataFrame(data=[gene_dict[i] for i in gene_df.index],
                                 columns=['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand'],
                                 index=gene_df.index)
    gene_df = pd.concat([gene_info, gene_df], axis=1)
    assert np.all(gene_df.index == gene_info.index)

    col_order = ['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand',
        'num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df', 'variant_id']
    if 'tss_distance' in gene_df:
        col_order += ['tss_distance']
    else:
        col_order += ['start_distance', 'end_distance']
    if lookup_df is not None:
        print('  * adding variant annotations from lookup table', flush=True)
        gene_df = gene_df.join(lookup_df, on='variant_id')  # add variant information
        col_order += list(lookup_df.columns)
    col_order += ['ma_samples', 'ma_count', 'af', 'pval_nominal',
                  'slope', 'slope_se', 'pval_perm', 'pval_beta']
    if 'group_id' in gene_df:
        col_order += ['group_id', 'group_size']
    col_order += ['qval', 'pval_nominal_threshold']
    gene_df = gene_df[col_order]
    print('done.', flush=True)
    return gene_df


def get_significant_pairs(res_df, nominal_files, group_s=None, fdr=0.05):
    """Significant variant-phenotype pairs based on nominal p-value threshold for each phenotype"""
    print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] tensorQTL: parsing all significant variant-phenotype pairs', flush=True)
    assert 'qval' in res_df

    # significant phenotypes (apply FDR threshold)
    if group_s is not None:
        df = res_df.loc[res_df['qval'] <= fdr, ['pval_nominal_threshold', 'pval_nominal', 'pval_beta', 'group_id']].copy()
        df.set_index('group_id', inplace=True)
    else:
        df = res_df.loc[res_df['qval'] <= fdr, ['pval_nominal_threshold', 'pval_nominal', 'pval_beta']].copy()
    df.rename(columns={'pval_nominal': 'min_pval_nominal'}, inplace=True)
    signif_phenotype_ids = set(df.index)
    threshold_dict = df['pval_nominal_threshold'].to_dict()

    if isinstance(nominal_files, str):
        # chr -> file
        nominal_files = {os.path.basename(i).split('.')[-2]:i for i in glob.glob(nominal_files+'*.parquet')}
    else:
        assert isinstance(nominal_files, dict)

    chroms = sorted(nominal_files.keys(), key=lambda x: int(x.replace('chr', '').replace('X', '23')))
    signif_df = []
    for k,c in enumerate(chroms, 1):
        print(f'  * processing chr. {k}/{len(chroms)}', end='\r', flush=True)
        nominal_df = pd.read_parquet(nominal_files[c])
        # drop pairs that never pass threshold
        nominal_df = nominal_df[nominal_df['pval_nominal'] <= df['pval_nominal_threshold'].max()]
        if group_s is not None:
            nominal_df.insert(1, 'group_id', nominal_df['phenotype_id'].map(group_s))
            nominal_df = nominal_df[nominal_df['group_id'].isin(signif_phenotype_ids)]
            m = nominal_df['pval_nominal'] < nominal_df['group_id'].apply(lambda x: threshold_dict[x])
        else:
            nominal_df = nominal_df[nominal_df['phenotype_id'].isin(signif_phenotype_ids)]
            m = nominal_df['pval_nominal'] < nominal_df['phenotype_id'].apply(lambda x: threshold_dict[x])
        signif_df.append(nominal_df[m])
    print()
    signif_df = pd.concat(signif_df, axis=0)
    if group_s is not None:
        signif_df = signif_df.merge(df, left_on='group_id', right_index=True)
    else:
        signif_df = signif_df.merge(df, left_on='phenotype_id', right_index=True)
    print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] done', flush=True)
    return signif_df.reset_index(drop=True)

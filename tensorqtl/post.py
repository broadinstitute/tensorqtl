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
from core import SimpleLogger, Residualizer, center_normalize, impute_mean


has_rpy2 = False
e = subprocess.call('which R', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    import rpy2
    import rfunc
    if e==0:
        has_rpy2 = True
except:
    pass
if not has_rpy2:
    print("Warning: 'rfunc' cannot be imported. R and the 'rpy2' Python package are needed.")


def calculate_qvalues(res_df, fdr=0.05, qvalue_lambda=None, logger=None):
    """Annotate permutation results with q-values, p-value threshold"""
    if logger is None:
        logger = SimpleLogger()

    logger.write('Computing q-values')
    logger.write('  * Number of phenotypes tested: {}'.format(res_df.shape[0]))
    logger.write('  * Correlation between Beta-approximated and empirical p-values: : {:.4f}'.format(
        stats.pearsonr(res_df['pval_perm'], res_df['pval_beta'])[0]))

    # calculate q-values
    if qvalue_lambda is None:
        qval, pi0 = rfunc.qvalue(res_df['pval_beta'])
    else:
        logger.write('  * Calculating q-values with lambda = {:.3f}'.format(qvalue_lambda))
        qval, pi0 = rfunc.qvalue(res_df['pval_beta'], qvalue_lambda)
    res_df['qval'] = qval
    logger.write('  * Proportion of significant phenotypes (1-pi0): {:.2f}'.format(1 - pi0))
    logger.write('  * QTL phenotypes @ FDR {:.2f}: {}'.format(fdr, np.sum(res_df['qval']<=fdr)))

    # determine global min(p) significance threshold and calculate nominal p-value threshold for each gene
    ub = res_df.loc[res_df['qval']>fdr, 'pval_beta'].sort_values()[0]
    lb = res_df.loc[res_df['qval']<=fdr, 'pval_beta'].sort_values()[-1]
    pthreshold = (lb+ub)/2
    logger.write('  * min p-value threshold @ FDR {}: {:.6g}'.format(fdr, pthreshold))
    res_df['pval_nominal_threshold'] = stats.beta.ppf(pthreshold, res_df['beta_shape1'], res_df['beta_shape2'])


def calculate_replication(res_df, genotype_df, phenotype_df, covariates_df, interaction_s=None,
                          lambda_qvalue=None):
    """res_df: DataFrame with 'variant_id' column and phenotype IDs as index"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genotypes_t = torch.tensor(genotype_df.loc[res_df['variant_id']].values, dtype=torch.float).to(device)
    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)
    genotypes_t = genotypes_t[:,genotype_ix_t]
    impute_mean(genotypes_t)

    phenotypes_t = torch.tensor(phenotype_df.loc[res_df.index].values, dtype=torch.float32).to(device)

    residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))

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

    if interaction_s is None:
        genotype_res_t = residualizer.transform(genotypes_t)  # variants x samples
        phenotype_res_t = residualizer.transform(phenotypes_t)  # phenotypes x samples

        gstd = genotype_res_t.var(1)
        pstd = phenotype_res_t.var(1)
        std_ratio_t = torch.sqrt(pstd / gstd)

        # center and normalize
        genotype_res_t = center_normalize(genotype_res_t, dim=1)
        phenotype_res_t = center_normalize(phenotype_res_t, dim=1)

        r_nominal_t = (genotype_res_t * phenotype_res_t).sum(1)
        r2_nominal_t = r_nominal_t.double().pow(2)

        dof = residualizer.dof
        tstat_t = torch.sqrt((dof * r2_nominal_t) / (1 - r2_nominal_t))
        slope_t = r_nominal_t * std_ratio_t
        slope_se_t = (slope_t.abs().double() / tstat_t).float()
        pval = 2*stats.t.cdf(-np.abs(tstat_t.cpu()), dof)

        rep_df = pd.DataFrame(np.c_[res_df.index, res_df['variant_id'], ma_samples_t.cpu(), ma_count_t.cpu(), maf_t.cpu(), pval, slope_t.cpu(), slope_se_t.cpu()],
                              columns=['phenotype_id', 'variant_id', 'ma_samples', 'ma_count', 'maf', 'pval_nominal', 'slope', 'slope_se']).infer_objects()

    else:
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

        rep_df = pd.DataFrame(np.c_[res_df.index, res_df['variant_id'], ma_samples_t.cpu(), ma_count_t.cpu(), maf_t.cpu(),
                                    pval[:,0], b[:,0], b_se[:,0], pval[:,1], b[:,1], b_se[:,1], pval[:,2], b[:,2], b_se[:,2]],
                              columns=['phenotype_id', 'variant_id', 'ma_samples', 'ma_count', 'maf',
                                       'pval_g', 'b_g', 'b_g_se', 'pval_i', 'b_i', 'b_i_se', 'pval_gi', 'b_gi', 'b_gi_se']).infer_objects()
        pval = pval[:,2]

    try:
        pi1 = 1 - rfunc.pi0est(pval, lambda_qvalue=lambda_qvalue)[0]
    except:
        pi1 = np.NaN
    return pi1, rep_df


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
            if row[0][0]=='#' or row[2]!='gene': continue
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
    assert np.all(gene_df.index==gene_info.index)

    col_order = ['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand',
        'num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df', 'variant_id', 'tss_distance']
    if lookup_df is not None:
        print('  * adding variant annotations from lookup table', flush=True)
        gene_df = gene_df.join(lookup_df, on='variant_id')  # add variant information
        col_order += list(lookup_df.columns)
    col_order += ['ma_samples', 'ma_count', 'maf', 'ref_factor',
        'pval_nominal', 'slope', 'slope_se', 'pval_perm', 'pval_beta']
    if 'group_id' in gene_df:
        col_order += ['group_id', 'group_size']
    col_order += ['qval', 'pval_nominal_threshold']
    gene_df = gene_df[col_order]
    print('done.', flush=True)
    return gene_df


def get_significant_pairs(res_df, nominal_prefix, fdr=0.05):
    """Significant variant-phenotype pairs based on nominal p-value threshold for each phenotype"""
    print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] tensorQTL: filtering significant variant-phenotype pairs', flush=True)
    assert 'qval' in res_df

    # significant phenotypes (apply FDR threshold)
    df = res_df.loc[res_df['qval']<=fdr, ['pval_nominal_threshold', 'pval_nominal', 'pval_beta']].copy()
    df.rename(columns={'pval_nominal': 'min_pval_nominal'}, inplace=True)
    signif_phenotype_ids = set(df.index)
    threshold_dict = df['pval_nominal_threshold'].to_dict()

    nominal_files = {os.path.basename(i).split('.')[-2]:i for i in glob.glob(nominal_prefix+'*.parquet')}
    chroms = sorted(nominal_files.keys(), key=lambda x: int(x.replace('chr','').replace('X','100')))
    signif_df = []
    for k,c in enumerate(chroms, 1):
        print('  * parsing significant variant-phenotype pairs for chr. {}/{}'.format(k, len(chroms)), end='\r', flush=True)
        nominal_df = pd.read_parquet(nominal_files[c])
        nominal_df = nominal_df[nominal_df['phenotype_id'].isin(signif_phenotype_ids)]

        m = nominal_df['pval_nominal']<nominal_df['phenotype_id'].apply(lambda x: threshold_dict[x])
        signif_df.append(nominal_df[m])
    print()
    signif_df = pd.concat(signif_df, axis=0)
    signif_df = signif_df.merge(df, left_on='phenotype_id', right_index=True)
    print('['+datetime.now().strftime("%b %d %H:%M:%S")+'] done', flush=True)
    return signif_df.reset_index(drop=True)

import numpy as np
import scipy.stats as stats
import subprocess

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


def calculate_qvalues(res_df, fdr=0.05, qvalue_lambda=None):
    """Annotate permutation results with q-values, p-value threshold"""

    print('Computing q-values')
    print('  * Number of phenotypes tested: {}'.format(res_df.shape[0]))
    print('  * Correlation between Beta-approximated and empirical p-values: : {:.4f}'.format(
        stats.pearsonr(res_df['pval_perm'], res_df['pval_beta'])[0]))

    # calculate q-values
    if qvalue_lambda is None:
        qval, pi0 = rfunc.qvalue(res_df['pval_beta'])
    else:
        print('  * Calculating q-values with lambda = {:.3f}'.format(qvalue_lambda))
        qval, pi0 = rfunc.qvalue(res_df['pval_beta'], qvalue_lambda)
    res_df['qval'] = qval
    print('  * Proportion of significant phenotypes (1-pi0): {:.2f}'.format(1 - pi0))
    print('  * QTL phenotypes @ FDR {:.2f}: {}'.format(fdr, np.sum(res_df['qval']<=fdr)))

    # determine global min(p) significance threshold and calculate nominal p-value threshold for each gene
    ub = res_df.loc[res_df['qval']>fdr, 'pval_beta'].sort_values()[0]
    lb = res_df.loc[res_df['qval']<=fdr, 'pval_beta'].sort_values()[-1]
    pthreshold = (lb+ub)/2
    print('  * min p-value threshold @ FDR {}: {:.6g}'.format(fdr, pthreshold))
    res_df['pval_nominal_threshold'] = stats.beta.ppf(pthreshold, res_df['beta_shape1'], res_df['beta_shape2'])


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

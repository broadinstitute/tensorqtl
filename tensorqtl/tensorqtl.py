#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import argparse

sys.path.insert(1, os.path.dirname(__file__))
from core import *
from post import *
import genotypeio, cis, trans, susie


def main():
    parser = argparse.ArgumentParser(description='tensorQTL: GPU-based QTL mapper')
    parser.add_argument('genotype_path', help='Genotypes in PLINK format')
    parser.add_argument('phenotype_bed', help='Phenotypes in BED format')
    parser.add_argument('prefix', help='Prefix for output file names')
    parser.add_argument('--mode', default='cis', choices=['cis', 'cis_nominal', 'cis_independent', 'cis_susie', 'trans'], help='Mapping mode. Default: cis')
    parser.add_argument('--covariates', default=None, help='Covariates file, tab-delimited, covariates x samples')
    parser.add_argument('--permutations', type=int, default=10000, help='Number of permutations. Default: 10000')
    parser.add_argument('--interaction', default=None, type=str, help='Interaction term(s)')
    parser.add_argument('--cis_output', default=None, type=str, help="Output from 'cis' mode with q-values. Required for independent cis-QTL mapping.")
    parser.add_argument('--phenotype_groups', default=None, type=str, help='Phenotype groups. Header-less TSV with two columns: phenotype_id, group_id')
    parser.add_argument('--window', default=1000000, type=np.int32, help='Cis-window size, in bases. Default: 1000000.')
    parser.add_argument('--pval_threshold', default=None, type=np.float64, help='Output only significant phenotype-variant pairs with a p-value below threshold. Default: 1e-5 for trans-QTL')
    parser.add_argument('--maf_threshold', default=0, type=np.float64, help='Include only genotypes with minor allele frequency >= maf_threshold. Default: 0')
    parser.add_argument('--maf_threshold_interaction', default=0.05, type=np.float64, help='MAF threshold for interactions, applied to lower and upper half of samples')
    parser.add_argument('--return_dense', action='store_true', help='Return dense output for trans-QTL.')
    parser.add_argument('--return_r2', action='store_true', help='Return r2 (only for sparse trans-QTL output)')
    parser.add_argument('--best_only', action='store_true', help='Only write lead association for each phenotype (interaction mode only)')
    parser.add_argument('--output_text', action='store_true', help='Write output in txt.gz format instead of parquet (trans-QTL mode only)')
    parser.add_argument('--batch_size', type=int, default=20000, help='Batch size. Reduce this if encountering OOM errors.')
    parser.add_argument('--load_split', action='store_true', help='Load genotypes into memory separately for each chromosome.')
    parser.add_argument('--warn_monomorphic', action='store_true', help='Warn if monomorphic variants are found.')
    parser.add_argument('--fdr', default=0.05, type=np.float64, help='FDR for cis-QTLs')
    parser.add_argument('--qvalue_lambda', default=None, type=np.float64, help='lambda parameter for pi0est in qvalue.')
    parser.add_argument('--seed', default=None, type=int, help='Seed for permutations.')
    parser.add_argument('-o', '--output_dir', default='.', help='Output directory')
    args = parser.parse_args()

    # check inputs
    if args.mode == 'cis_independent' and (args.cis_output is None or not os.path.exists(args.cis_output)):
        raise ValueError("Output from 'cis' mode must be provided.")
    if args.interaction is not None and args.mode not in ['cis_nominal', 'trans']:
        raise ValueError("Interactions are only supported in 'cis_nominal' or 'trans' mode.")

    logger = SimpleLogger(os.path.join(args.output_dir, f'{args.prefix}.tensorQTL.{args.mode}.log'))
    logger.write(f'[{datetime.now().strftime("%b %d %H:%M:%S")}] Running TensorQTL: {args.mode.split("_")[0]}-QTL mapping')
    if torch.cuda.is_available():
        logger.write(f'  * using GPU ({torch.cuda.get_device_name(torch.cuda.current_device())})')
    else:
        logger.write('  * WARNING: using CPU!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        logger.write(f'  * using seed {args.seed}')

    # load inputs
    logger.write(f'  * reading phenotypes ({args.phenotype_bed})')
    phenotype_df, phenotype_pos_df = read_phenotype_bed(args.phenotype_bed)

    tss_dict = phenotype_pos_df.T.to_dict()
    if args.covariates is not None:
        logger.write(f'  * reading covariates ({args.covariates})')
        covariates_df = pd.read_csv(args.covariates, sep='\t', index_col=0).T
        assert phenotype_df.columns.equals(covariates_df.index)
    if args.interaction is not None:
        logger.write(f'  * reading interaction term(s) ({args.interaction})')
        # allow headerless input for single interactions
        with open(args.interaction) as f:
            f.readline()
            s = f.readline().strip()
        if len(s.split('\t')) == 2:  # index + value
            interaction_df = pd.read_csv(args.interaction, sep='\t', index_col=0, header=None)
        else:
            interaction_df = pd.read_csv(args.interaction, sep='\t', index_col=0)
        # select samples
        assert covariates_df.index.isin(interaction_df.index).all()
        interaction_df = interaction_df.loc[covariates_df.index].astype(np.float32)
    else:
        interaction_df = None

    if args.maf_threshold is None:
        if args.mode == 'trans':
            maf_threshold = 0.05
        else:
            maf_threshold = 0
    else:
        maf_threshold = args.maf_threshold

    if args.phenotype_groups is not None:
        group_s = pd.read_csv(args.phenotype_groups, sep='\t', index_col=0, header=None).squeeze('columns')
        # verify sort order
        group_dict = group_s.to_dict()
        previous_group = ''
        parsed_groups = 0
        for i in phenotype_df.index:
            if group_dict[i] != previous_group:
                parsed_groups += 1
                previous_group = group_dict[i]
        if not parsed_groups == len(group_s.unique()):
            raise ValueError('Groups defined in input do not match phenotype file (check sort order).')
    else:
        group_s = None

    # load genotypes
    pr = genotypeio.PlinkReader(args.genotype_path, select_samples=phenotype_df.columns, dtype=np.int8)
    variant_df = pr.bim.set_index('snp')[['chrom', 'pos']]
    if not args.load_split or args.mode != 'cis_nominal':  # load all genotypes into memory
        genotype_df = pd.DataFrame(pr.load_genotypes(), index=pr.bim['snp'], columns=pr.fam['iid'])

    if args.mode.startswith('cis'):
        if args.mode == 'cis':
            res_df = cis.map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
                                 group_s=group_s, nperm=args.permutations, window=args.window,
                                 maf_threshold=maf_threshold, warn_monomorphic=args.warn_monomorphic,
                                 logger=logger, seed=args.seed, verbose=True)
            logger.write('  * writing output')
            if has_rpy2:
                calculate_qvalues(res_df, fdr=args.fdr, qvalue_lambda=args.qvalue_lambda, logger=logger)
            out_file = os.path.join(args.output_dir, f'{args.prefix}.cis_qtl.txt.gz')
            res_df.to_csv(out_file, sep='\t', float_format='%.6g')
        elif args.mode == 'cis_nominal':
            if not args.load_split:
                cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, args.prefix, covariates_df=covariates_df,
                                interaction_df=interaction_df, maf_threshold_interaction=args.maf_threshold_interaction,
                                group_s=None, window=args.window, maf_threshold=maf_threshold, run_eigenmt=True,
                                output_dir=args.output_dir, write_top=True, write_stats=not args.best_only, logger=logger, verbose=True)
                # compute significant pairs
                if args.cis_output is not None:
                    cis_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
                    nominal_prefix = os.path.join(args.output_dir, f'{args.prefix}.cis_qtl_pairs')
                    signif_df = get_significant_pairs(cis_df, nominal_prefix, group_s=group_s, fdr=args.fdr)
                    signif_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.cis_qtl.signif_pairs.parquet'))

            else:  # load genotypes for each chromosome separately
                top_df = []
                for chrom in pr.chrs:
                    g, pos_s = pr.get_region(chrom)
                    genotype_df = pd.DataFrame(g, index=pos_s.index, columns=pr.fam['iid'])[phenotype_df.columns]
                    variant_df = pr.bim.set_index('snp')[['chrom', 'pos']]
                    chr_df = cis.map_nominal(genotype_df, variant_df[variant_df['chrom'] == chrom],
                                             phenotype_df[phenotype_pos_df['chr'] == chrom], phenotype_pos_df[phenotype_pos_df['chr'] == chrom],
                                             args.prefix, covariates_df=covariates_df,
                                             interaction_df=interaction_df, maf_threshold_interaction=args.maf_threshold_interaction,
                                             group_s=None, window=args.window, maf_threshold=maf_threshold, run_eigenmt=True,
                                             output_dir=args.output_dir, write_top=True, write_stats=not args.best_only, logger=logger, verbose=True)
                    top_df.append(chr_df)
                if interaction_df is not None:
                    top_df = pd.concat(top_df)
                    top_df.to_csv(os.path.join(args.output_dir, f'{args.prefix}.cis_qtl_top_assoc.txt.gz'),
                                  sep='\t', float_format='%.6g')

        elif args.mode == 'cis_independent':
            summary_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
            summary_df.rename(columns={'minor_allele_samples':'ma_samples', 'minor_allele_count':'ma_count'}, inplace=True)
            res_df = cis.map_independent(genotype_df, variant_df, summary_df, phenotype_df, phenotype_pos_df, covariates_df,
                                         group_s=group_s, fdr=args.fdr, nperm=args.permutations, window=args.window,
                                         maf_threshold=maf_threshold, logger=logger, seed=args.seed, verbose=True)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir, f'{args.prefix}.cis_independent_qtl.txt.gz')
            res_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g')

        elif args.mode == 'cis_susie':
            signif_df = pd.read_parquet(args.cis_output)
            ix = phenotype_df.index[phenotype_df.index.isin(signif_df['phenotype_id'].unique())]
            summary_df = susie.map(genotype_df, variant_df,
                                   phenotype_df.loc[ix], phenotype_pos_df.loc[ix],
                                   covariates_df, max_iter=500, summary_only=True)
            summary_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.SuSiE_summary.parquet'))

    elif args.mode == 'trans':
        return_sparse = not args.return_dense
        pval_threshold = args.pval_threshold
        if pval_threshold is None and return_sparse:
            pval_threshold = 1e-5
            logger.write(f'  * p-value threshold: {pval_threshold:.2g}')

        if interaction_df is not None:
            if interaction_df.shape[1] > 1:
                raise NotImplementedError('trans-QTL mapping currently only supports a single interaction.')
            else:
                interaction_df = interaction_df.squeeze('columns')

        pairs_df = trans.map_trans(genotype_df, phenotype_df, covariates_df, interaction_s=interaction_df,
                                  return_sparse=return_sparse, pval_threshold=pval_threshold,
                                  maf_threshold=maf_threshold, batch_size=args.batch_size,
                                  return_r2=args.return_r2, logger=logger)

        logger.write('  * filtering out cis-QTLs (within +/-5Mb)')
        pairs_df = trans.filter_cis(pairs_df, tss_dict, variant_df, window=5000000)

        logger.write('  * writing output')
        if not args.output_text:
            pairs_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pairs.parquet'))
        else:
            out_file = os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pairs.txt.gz')
            pairs_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g')

    logger.write(f'[{datetime.now().strftime("%b %d %H:%M:%S")}] Finished mapping')


if __name__ == '__main__':
    main()

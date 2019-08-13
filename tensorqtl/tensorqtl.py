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
import genotypeio
import cis
import trans


def main():
    parser = argparse.ArgumentParser(description='tensorQTL: GPU-based QTL mapper')
    parser.add_argument('genotype_path', help='Genotypes/dosages in PLINK or tfrecord format')
    parser.add_argument('phenotype_bed', help='Phenotypes in BED format')
    parser.add_argument('prefix', help='Prefix for output file names')
    parser.add_argument('--mode', default='cis', choices=['cis', 'cis_nominal', 'cis_independent', 'trans'], help='Mapping mode. Default: cis')
    parser.add_argument('--covariates', default=None, help='Covariates file, tab-delimited, covariates x samples')
    parser.add_argument('--permutations', default=10000, help='Number of permutations. Default: 10000')
    parser.add_argument('--interaction', default=None, type=str, help='Interaction term')
    parser.add_argument('--cis_output', default=None, type=str, help="Output from 'cis' mode with q-values. Required for independent cis-QTL mapping.")
    parser.add_argument('--phenotype_groups', default=None, type=str, help='Phenotype groups. Header-less TSV with two columns: phenotype_id, group_id')
    parser.add_argument('--window', default=1000000, type=np.int32, help='Cis-window size, in bases. Default: 1000000.')
    parser.add_argument('--pval_threshold', default=None, type=np.float64, help='Output only significant phenotype-variant pairs with a p-value below threshold. Default: 1e-5 for trans-QTL')
    parser.add_argument('--maf_threshold', default=None, type=np.float64, help='Include only genotypes with minor allele frequency >=maf_threshold. Default: 0')
    parser.add_argument('--maf_threshold_interaction', default=0.05, type=np.float64, help='MAF threshold for interactions, applied to lower and upper half of samples')
    parser.add_argument('--return_dense', action='store_true', help='Return dense output for trans-QTL.')
    parser.add_argument('--return_r2', action='store_true', help='Return r2 (only for sparse trans-QTL output)')
    parser.add_argument('--best_only', action='store_true', help='Produce output only for the top association/phenotype')
    parser.add_argument('--output_text', action='store_true', help='Write output in txt.gz format instead of parquet (trans-QTL mode only)')
    parser.add_argument('--batch_size', type=int, default=50000, help='Batch size. Reduce this if encountering OOM errors.')
    parser.add_argument('--fdr', default=0.05, type=np.float64, help='FDR for cis-QTLs')
    parser.add_argument('--qvalue_lambda', default=None, type=np.float64, help='lambda parameter for pi0est in qvalue.')
    parser.add_argument('--seed', default=None, type=int, help='Seed for permutations.')
    parser.add_argument('-o', '--output_dir', default='.', help='Output directory')
    args = parser.parse_args()

    # check inputs
    if args.mode=='cis_independent' and (args.cis_output is None or not os.path.exists(args.cis_output)):
        raise ValueError("Output from 'cis' mode must be provided.")

    logger = SimpleLogger(os.path.join(args.output_dir, args.prefix+'.tensorQTL.{}.log'.format(args.mode)))
    logger.write('[{}] Running TensorQTL: {}-QTL mapping'.format(datetime.now().strftime("%b %d %H:%M:%S"), args.mode.split('_')[0]))
    if args.seed is not None:
        logger.write('  * using seed {}'.format(args.seed))

    # load inputs
    logger.write('  * reading phenotypes ({})'.format(args.phenotype_bed))
    phenotype_df, phenotype_pos_df = read_phenotype_bed(args.phenotype_bed)

    tss_dict = phenotype_pos_df.T.to_dict()
    if args.covariates is not None:
        logger.write('  * reading covariates ({})'.format(args.covariates))
        covariates_df = pd.read_csv(args.covariates, sep='\t', index_col=0).T
        # assert np.all(phenotype_df.columns==covariates_df.index)
        phenotype_df = phenotype_df[covariates_df.index]   ######################### in case inputs are not sorted the same way
    if args.interaction is not None:
        logger.write('  * reading interaction term ({})'.format(args.interaction))
        interaction_s = pd.read_csv(args.interaction, sep='\t', index_col=0, header=None, squeeze=True)
        assert covariates_df.index.isin(interaction_s.index).all()
        interaction_s = interaction_s.loc[covariates_df.index].astype(np.float32)
    else:
        interaction_s = None

    if args.maf_threshold is None:
        if args.mode=='trans':
            maf_threshold = 0.05
        else:
            maf_threshold = 0
    else:
        maf_threshold = args.maf_threshold

    if args.phenotype_groups is not None:
        group_s = pd.read_csv(args.phenotype_groups, sep='\t', index_col=0, header=None, squeeze=True)
        # verify sort order
        group_dict = group_s.to_dict()
        previous_group = ''
        parsed_groups = 0
        for i in phenotype_df.index:
            if group_dict[i]!=previous_group:
                parsed_groups += 1
                previous_group = group_dict[i]
        if not parsed_groups == len(group_s.unique()):
            raise ValueError('Groups defined in input do not match phenotype file (check sort order).')
    else:
        group_s = None

    if args.mode.startswith('cis'):
        pr = genotypeio.PlinkReader(args.genotype_path, select_samples=phenotype_df.columns)
        if args.mode=='cis':
            res_df = cis.map_cis(pr, phenotype_df, phenotype_pos_df, covariates_df,
                             group_s=group_s, nperm=args.permutations, window=args.window, logger=logger, seed=args.seed)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir, args.prefix+'.cis_qtl.txt.gz')
            if has_rpy2 and group_s is None:
                calculate_qvalues(res_df, fdr=args.fdr, qvalue_lambda=args.qvalue_lambda)
            res_df.to_csv(out_file, sep='\t', float_format='%.6g')
        elif args.mode=='cis_nominal':
            if interaction_s is None:
                cis.map_nominal(pr, phenotype_df, phenotype_pos_df, covariates_df, args.prefix,
                                window=args.window, output_dir=args.output_dir, group_s=group_s, logger=logger)
            else:
                cid.map_nominal_interaction(pr, phenotype_df, phenotype_pos_df, covariates_df, interaction_s,
                                args.prefix, maf_threshold_interaction=args.maf_threshold_interaction,
                                window=args.window, best_only=args.best_only, output_dir=args.output_dir, logger=logger)
        elif args.mode=='cis_independent':
            summary_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
            summary_df.rename(columns={'minor_allele_samples':'ma_samples', 'minor_allele_count':'ma_count'}, inplace=True)
            res_df = cis.map_independent(pr, summary_df, phenotype_df, phenotype_pos_df, covariates_df,
                                         fdr=args.fdr, nperm=args.permutations, window=args.window, logger=logger, seed=args.seed)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir, args.prefix+'.cis_independent_qtl.txt.gz')
            res_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g')
    elif args.mode=='trans':
        return_sparse = not args.return_dense
        pval_threshold = args.pval_threshold
        if pval_threshold is None and return_sparse:
            pval_threshold = 1e-5
            logger.write('  * p-value threshold: {:.2g}'.format(pval_threshold))

        genotype_df = genotypeio.load_genotypes(args.genotype_path, select_samples=phenotype_df.columns, dtype=np.int8)
        pval_df = trans.map_trans(genotype_df, phenotype_df, covariates_df, interaction_s=interaction_s,
                                  return_sparse=return_sparse, pval_threshold=pval_threshold,
                                  maf_threshold=maf_threshold, batch_size=args.batch_size,
                                  return_r2=args.return_r2, logger=logger)

        logger.write('  * filtering out cis-QTLs (within +/-5Mb)')
        pval_df = trans.filter_cis(pval_df, tss_dict, window=5000000)

        logger.write('  * writing output')
        if not args.output_text:
            pval_df.to_parquet(os.path.join(args.output_dir, args.prefix+'.trans_qtl_pairs.parquet'))
        else:
            out_file = os.path.join(args.output_dir, args.prefix+'.trans_qtl_pairs.txt.gz')
            pval_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g')

    logger.write('[{}] Finished mapping'.format(datetime.now().strftime("%b %d %H:%M:%S")))


if __name__ == '__main__':
    main()

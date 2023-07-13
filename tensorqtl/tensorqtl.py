#!/usr/bin/env python3
from __future__ import print_function
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import re
import pickle
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
    parser.add_argument('--paired_covariate', default=None, help='Single phenotype-specific covariate. Tab-delimited file, phenotypes x samples')
    parser.add_argument('--permutations', type=int, default=10000, help='Number of permutations. Default: 10000')
    parser.add_argument('--interaction', default=None, type=str, help='Interaction term(s)')
    parser.add_argument('--cis_output', default=None, type=str, help="Output from 'cis' mode with q-values. Required for independent cis-QTL mapping.")
    parser.add_argument('--phenotype_groups', default=None, type=str, help='Phenotype groups. Header-less TSV with two columns: phenotype_id, group_id')
    parser.add_argument('--window', default=1000000, type=np.int32, help='Cis-window size, in bases. Default: 1000000.')
    parser.add_argument('--pval_threshold', default=None, type=np.float64, help='Output only significant phenotype-variant pairs with a p-value below threshold. Default: 1e-5 for trans-QTL')
    parser.add_argument('--maf_threshold', default=0, type=np.float64, help='Include only genotypes with minor allele frequency >= maf_threshold. Default: 0')
    parser.add_argument('--maf_threshold_interaction', default=0.05, type=np.float64, help='MAF threshold for interactions, applied to lower and upper half of samples')
    parser.add_argument('--dosages', action='store_true', help='Load dosages instead of genotypes (only applies to PLINK2 bgen input).')
    parser.add_argument('--return_dense', action='store_true', help='Return dense output for trans-QTL.')
    parser.add_argument('--return_r2', action='store_true', help='Return r2 (only for sparse trans-QTL output)')
    parser.add_argument('--best_only', action='store_true', help='Only write lead association for each phenotype (interaction mode only)')
    parser.add_argument('--output_text', action='store_true', help='Write output in txt.gz format instead of parquet (trans-QTL mode only)')
    parser.add_argument('--batch_size', type=int, default=20000, help='GPU batch size (trans-QTLs only). Reduce this if encountering OOM errors.')
    parser.add_argument('--chunk_size', default=None, help="For cis-QTL mapping, load genotypes into CPU memory in chunks of chunk_size variants, or by chromosome if chunk_size is 'chr'.")
    parser.add_argument('--disable_beta_approx', action='store_true', help='Disable Beta-distribution approximation of empirical p-values (not recommended).')
    parser.add_argument('--warn_monomorphic', action='store_true', help='Warn if monomorphic variants are found.')
    parser.add_argument('--max_effects', default=10, help='Maximum number of non-zero effects in the SuSiE regression model.')
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
    if phenotype_pos_df.columns[1] == 'pos':
        logger.write(f"  * cis-window detected as position ± {args.window:,}")
    else:
        logger.write(f"  * cis-window detected as [start - {args.window:,}, end + {args.window:,}]")

    if args.covariates is not None:
        logger.write(f'  * reading covariates ({args.covariates})')
        covariates_df = pd.read_csv(args.covariates, sep='\t', index_col=0).T
        assert phenotype_df.columns.equals(covariates_df.index)
    else:
        covariates_df = None

    if args.paired_covariate is not None:
        assert covariates_df is not None, f"Covariates matrix must be provided when using paired covariate"
        paired_covariate_df = pd.read_csv(args.paired_covariate, sep='\t', index_col=0)  # phenotypes x samples
        assert paired_covariate_df.index.isin(phenotype_df.index).all(), f"Paired covariate phenotypes must be present in phenotype matrix."
        assert paired_covariate_df.columns.equals(phenotype_df.columns), f"Paired covariate samples must match samples in phenotype matrix."
    else:
        paired_covariate_df = None

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
    if args.chunk_size is None or not args.mode.startswith('cis'):  # load all genotypes into memory
        logger.write(f'  * loading genotype dosages' if args.dosages else f'  * loading genotypes')
        genotype_df, variant_df = genotypeio.load_genotypes(args.genotype_path, select_samples=phenotype_df.columns, dosages=args.dosages)
        if variant_df is None:
            assert not args.mode.startswith('cis'), f"Genotype data without variant positions is only supported for mode='trans'."
    else:
        if not all([os.path.exists(f"{args.genotype_path}.{ext}") for ext in ['pgen', 'psam', 'pvar']]):
            raise ValueError("Processing in chunks requires PLINK 2 pgen/psam/pvar files.")
        import pgen
        pgr = pgen.PgenReader(args.genotype_path, select_samples=phenotype_df.columns)

    if args.mode.startswith('cis'):
        if args.mode == 'cis':
            if args.chunk_size is None:
                res_df = cis.map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df=covariates_df,
                                     group_s=group_s, paired_covariate_df=paired_covariate_df, nperm=args.permutations,
                                     window=args.window, beta_approx=not args.disable_beta_approx, maf_threshold=maf_threshold,
                                     warn_monomorphic=args.warn_monomorphic, logger=logger, seed=args.seed, verbose=True)
            else:
                res_df = []
                for gt_df, var_df, p_df, p_pos_df, _ in genotypeio.generate_paired_chunks(pgr, phenotype_df, phenotype_pos_df, args.chunk_size,
                                                                                       dosages=args.dosages, verbose=True):
                    res_df.append(cis.map_cis(gt_df, var_df, p_df, p_pos_df, covariates_df=covariates_df,
                                              group_s=group_s, paired_covariate_df=paired_covariate_df, nperm=args.permutations,
                                              window=args.window, beta_approx=not args.disable_beta_approx, maf_threshold=maf_threshold,
                                              warn_monomorphic=args.warn_monomorphic, logger=logger, seed=args.seed, verbose=True))
                res_df = pd.concat(res_df)
            logger.write('  * writing output')
            if has_rpy2:
                calculate_qvalues(res_df, fdr=args.fdr, qvalue_lambda=args.qvalue_lambda, logger=logger)
            out_file = os.path.join(args.output_dir, f'{args.prefix}.cis_qtl.txt.gz')
            res_df.to_csv(out_file, sep='\t', float_format='%.6g')

        elif args.mode == 'cis_nominal':
            if args.chunk_size is None:
                cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, args.prefix, covariates_df=covariates_df,
                                paired_covariate_df=paired_covariate_df, interaction_df=interaction_df,
                                maf_threshold_interaction=args.maf_threshold_interaction,
                                group_s=None, window=args.window, maf_threshold=maf_threshold, run_eigenmt=True,
                                output_dir=args.output_dir, write_top=True, write_stats=not args.best_only, logger=logger, verbose=True)
                # compute significant pairs
                if args.cis_output is not None:
                    cis_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
                    nominal_prefix = os.path.join(args.output_dir, f'{args.prefix}.cis_qtl_pairs')
                    signif_df = get_significant_pairs(cis_df, nominal_prefix, group_s=group_s, fdr=args.fdr)
                    signif_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.cis_qtl.signif_pairs.parquet'))

            else:
                chunks = []
                for gt_df, var_df, p_df, p_pos_df, ci in genotypeio.generate_paired_chunks(pgr, phenotype_df, phenotype_pos_df, args.chunk_size,
                                                                                           dosages=args.dosages, verbose=True):
                    prefix = f"{args.prefix}.chunk{ci+1}"
                    chunks.append(prefix)
                    cis.map_nominal(gt_df, var_df, p_df, p_pos_df, prefix, covariates_df=covariates_df,
                                    paired_covariate_df=paired_covariate_df, interaction_df=interaction_df,
                                    maf_threshold_interaction=args.maf_threshold_interaction,
                                    group_s=None, window=args.window, maf_threshold=maf_threshold, run_eigenmt=True,
                                    output_dir=args.output_dir, write_top=True, write_stats=not args.best_only, logger=logger, verbose=True)
                chunk_files = glob.glob(os.path.join(args.output_dir, f"{args.prefix}.chunk*.cis_qtl_pairs.*.parquet"))
                if args.chunk_size == 'chr':  # remove redundant chunk ID from file names
                    for f in chunk_files:
                        x = re.findall(f"{args.prefix}\.(chunk\d+)", os.path.basename(f))
                        assert len(x) == 1
                        os.rename(f, f.replace(f"{x[0]}.", ""))
                else:  # concatenate outputs by chromosome
                    chunk_df = pd.DataFrame({
                        'file': chunk_files,
                        'chunk': [int(re.findall(f"{args.prefix}\.chunk(\d+)", os.path.basename(i))[0]) for i in chunk_files],
                        'chr': [re.findall("\.cis_qtl_pairs\.(.*)\.parquet", os.path.basename(i))[0] for i in chunk_files],
                    }).sort_values('chunk')
                    for chrom, chr_df in chunk_df.groupby('chr', sort=False):
                        print(f"\rConcatenating outputs for {chrom}", end='' if chrom != chunk_df['chr'].iloc[-1] else None)
                        pd.concat([pd.read_parquet(f) for f in chr_df['file']]).reset_index(drop=True).to_parquet(
                            os.path.join(args.output_dir, f"{args.prefix}.cis_qtl_pairs.{chrom}.parquet"))
                        for f in chr_df['file']:
                            os.remove(f)
                # concatenate interaction results
                if interaction_df is not None:
                    chunk_files = [os.path.join(args.output_dir, f"{c}.cis_qtl_top_assoc.txt.gz") for c in chunks]
                    pd.concat([pd.read_csv(f, sep='\t', index_col=0, dtype=str) for f in chunk_files]).to_csv(
                        os.path.join(args.output_dir, f"{args.prefix}.cis_qtl_top_assoc.txt.gz"), sep='\t')
                    for f in chunk_files:
                        os.remove(f)

        elif args.mode == 'cis_independent':
            summary_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
            summary_df.rename(columns={'minor_allele_samples':'ma_samples', 'minor_allele_count':'ma_count'}, inplace=True)
            if args.chunk_size is None:
                res_df = cis.map_independent(genotype_df, variant_df, summary_df, phenotype_df, phenotype_pos_df, covariates_df,
                                             group_s=group_s, fdr=args.fdr, nperm=args.permutations, window=args.window,
                                             maf_threshold=maf_threshold, logger=logger, seed=args.seed, verbose=True)
            else:
                res_df = []
                for gt_df, var_df, p_df, p_pos_df, _ in genotypeio.generate_paired_chunks(pgr, phenotype_df, phenotype_pos_df, args.chunk_size,
                                                                                          dosages=args.dosages, verbose=True):
                    res_df.append(cis.map_independent(gt_df, var_df, summary_df, p_df, p_pos_df, covariates_df,
                                                      group_s=group_s, fdr=args.fdr, nperm=args.permutations, window=args.window,
                                                      maf_threshold=maf_threshold, logger=logger, seed=args.seed, verbose=True))
                res_df = pd.concat(res_df).reset_index(drop=True)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir, f'{args.prefix}.cis_independent_qtl.txt.gz')
            res_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g')

        elif args.mode == 'cis_susie':
            if args.cis_output.endswith('.parquet'):
                signif_df = pd.read_parquet(args.cis_output)
            else:
                signif_df = pd.read_csv(args.cis_output, sep='\t')
            if 'qval' in signif_df:  # otherwise input is from get_significant_pairs
                signif_df = signif_df[signif_df['qval'] <= args.fdr]
            phenotype_ids = phenotype_df.index[phenotype_df.index.isin(signif_df['phenotype_id'].unique())]
            phenotype_df = phenotype_df.loc[phenotype_ids]
            phenotype_pos_df = phenotype_pos_df.loc[phenotype_ids]
            if args.chunk_size is None:
                summary_df, res = susie.map(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                                            covariates_df, paired_covariate_df=paired_covariate_df, L=args.max_effects,
                                            maf_threshold=maf_threshold, max_iter=500, window=args.window, summary_only=False)
            else:
                summary_df = []
                res = {}
                for gt_df, var_df, p_df, p_pos_df, _ in genotypeio.generate_paired_chunks(pgr, phenotype_df, phenotype_pos_df, args.chunk_size,
                                                                                          dosages=args.dosages, verbose=True):
                    chunk_summary_df, chunk_res = susie.map(gt_df, var_df, p_df, p_pos_df,
                                                            covariates_df, paired_covariate_df=paired_covariate_df, L=args.max_effects,
                                                            maf_threshold=maf_threshold, max_iter=500, window=args.window, summary_only=False)
                    summary_df.append(chunk_summary_df)
                    res |= chunk_res
                summary_df = pd.concat(summary_df).reset_index(drop=True)

            summary_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.SuSiE_summary.parquet'))
            with open(os.path.join(args.output_dir, f'{args.prefix}.SuSiE.pickle'), 'wb') as f:
                pickle.dump(res, f)

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

        pairs_df = trans.map_trans(genotype_df, phenotype_df, covariates_df=covariates_df, interaction_s=interaction_df,
                                  return_sparse=return_sparse, pval_threshold=pval_threshold,
                                  maf_threshold=maf_threshold, batch_size=args.batch_size,
                                  return_r2=args.return_r2, logger=logger)

        if variant_df is not None:
            logger.write('  * filtering out cis-QTLs (within +/-5Mb)')
            pairs_df = trans.filter_cis(pairs_df, phenotype_pos_df, variant_df, window=5000000)

        logger.write('  * writing output')
        if not args.output_text:
            pairs_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pairs.parquet'))
        else:
            out_file = os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pairs.txt.gz')
            pairs_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g')

    logger.write(f'[{datetime.now().strftime("%b %d %H:%M:%S")}] Finished mapping')


if __name__ == '__main__':
    main()

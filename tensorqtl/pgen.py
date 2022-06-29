import pandas as pd
import numpy as np
import pgenlib as pg


def load_dosages(plink_prefix_path, as_dataframe=True):
    """
    Wrapper for pgenlib.PgenReader. Loads dosages as np.array or pd.DataFrame.

    To generate the pgen/psam/pvar files from a VCF, run
    plink2 --vcf ${vcf_file} 'dosage=DS' --out ${plink_prefix_path}

    Inputs:
      plink_prefix_path:  prefix to .pgen/.psam/.pvar files
      as_dataframe: return pd.DataFrame with variant (index) and sample (columns) IDs
    """
    if as_dataframe:
        pvar_df = pd.read_csv(f"{plink_prefix_path}.pvar", sep='\t', comment='#',
                              names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
        psam_df = pd.read_csv(f"{plink_prefix_path}.psam", sep='\t', index_col=0)

    with pg.PgenReader(f"{plink_prefix_path}.pgen".encode()) as r:
        num_samples = r.get_raw_sample_ct()
        num_variants = r.get_variant_ct()
        dosages = np.zeros([num_variants, num_samples], dtype=np.float32)
        r.read_dosages_range(0, num_variants, dosages)

    if as_dataframe:
        dosages = pd.DataFrame(dosages, index=pvar_df['id'], columns=psam_df.index)

    return dosages

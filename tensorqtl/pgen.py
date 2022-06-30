import numpy as np
import pandas as pd
import pgenlib as pg


def read_pvar(pvar_path):
    """Read pvar file as pd.DataFrame"""
    return pd.read_csv(pvar_path, sep='\t', comment='#',
                       names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])


def read_psam(psam_path):
    """Read psam file as pd.DataFrame"""
    return pd.read_csv(psam_path, sep='\t', index_col=0)


def read_dosages(pgen_path, variant_idx, sample_subset=None, dtype=np.float32):
    """
    Get dosages for a variant.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    variant_idx : int
        Variant index
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.
    dtype : np.float{32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotype dosages for the selected variant and samples.
    """
    if sample_subset is not None:
        sample_subset = np.array(sample_subset, dtype=np.uint32)
    with pg.PgenReader(pgen_path.encode(), sample_subset=sample_subset) as r:
        if sample_subset is None:
            num_samples = r.get_raw_sample_ct()
        else:
            num_samples = len(sample_subset)
        dosages = np.zeros(num_samples, dtype=dtype)
        r.read_dosages(np.array(variant_idx, dtype=np.uint32), dosages)
        return dosages


def read_dosages_list(pgen_path, variant_idxs, sample_subset=None, dtype=np.float32):
    """
    Get dosages for a list of variants.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    variant_idxs : array_like
        List of variant indexes
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.
    dtype : np.float{32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotype dosages for the selected variants and samples.
    """
    if sample_subset is not None:
        sample_subset = np.array(sample_subset, dtype=np.uint32)
    with pg.PgenReader(pgen_path.encode(), sample_subset=sample_subset) as r:
        if sample_subset is None:
            num_samples = r.get_raw_sample_ct()
        else:
            num_samples = len(sample_subset)
        num_variants = len(variant_idxs)
        dosages = np.zeros([num_variants, num_samples], dtype=dtype)
        r.read_dosages_list(np.array(variant_idxs, dtype=np.uint32), dosages)
        return dosages


def read_dosages_range(pgen_path, start_idx, end_idx, sample_subset=None, dtype=np.float32):
    """
    Get dosages for a range of variants.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    start_idx : int
        Start index of the range to query.
    end_idx : int
        End index of the range to query (inclusive).
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.
    dtype : np.float{32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotype dosages for the selected variants and samples.
    """
    if sample_subset is not None:
        sample_subset = np.array(sample_subset, dtype=np.uint32)
    with pg.PgenReader(pgen_path.encode(), sample_subset=sample_subset) as r:
        if sample_subset is None:
            num_samples = r.get_raw_sample_ct()
        else:
            num_samples = len(sample_subset)
        num_variants = end_idx - start_idx + 1
        dosages = np.zeros([num_variants, num_samples], dtype=dtype)
        r.read_dosages_range(start_idx, end_idx+1, dosages)
        return dosages


def read_dosages_df(plink_prefix_path, select_samples=None):
    """
    Load dosages for all variants and all/selected samples as a dataframe.

    To generate the pgen/psam/pvar files from a VCF, run
    plink2 --vcf ${vcf_file} 'dosage=DS' --output-chr chrM --out ${plink_prefix_path}

    Parameters
    ----------
    plink_prefix_path : str
        Prefix to .pgen/.psam/.pvar files
    select_samples : array_like
        List of sample IDs to select. Default: all samples.

    Returns
    -------
    dosages_df : pd.DataFrame (variants x samples)
        Genotype dosages for the selected samples.
    """
    pvar_df = read_pvar(f"{plink_prefix_path}.pvar")
    psam_df = read_psam(f"{plink_prefix_path}.psam")
    pgen_file = f"{plink_prefix_path}.pgen"
    sample_ids = psam_df.index.tolist()
    num_variants = pvar_df.shape[0]

    if select_samples is not None:
        sample_idxs = [sample_ids.index(i) for i in select_samples]
        sidx = np.argsort(sample_idxs)
        sample_idxs = [sample_idxs[i] for i in sidx]
        sample_ids = [select_samples[i] for i in sidx]
    else:
        sample_idxs = None

    dosages = read_dosages_range(pgen_file, 0, num_variants-1, sample_subset=sample_idxs)
    dosages = pd.DataFrame(dosages, index=pvar_df['id'], columns=sample_ids)
    return dosages

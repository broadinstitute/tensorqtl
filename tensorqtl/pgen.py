# Functions for reading dosages from PLINK pgen files based on the Pgenlib Python API:
# https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt

import numpy as np
import pandas as pd
import pgenlib as pg


def read_pvar(pvar_path):
    """Read pvar file as pd.DataFrame"""
    return pd.read_csv(pvar_path, sep='\t', comment='#',
                       names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
                       dtype={'chrom':str, 'pos':np.int32, 'id':str, 'ref':str, 'alt':str,
                              'qual':str, 'filter':str, 'info':str})


def read_psam(psam_path):
    """Read psam file as pd.DataFrame"""
    return pd.read_csv(psam_path, sep='\t', index_col=0)


def hardcall_phase_present(pgen_path):
    """Returns True iff phased hardcalls may be present"""
    with pg.PgenReader(pgen_path.encode()) as r:
        return r.hardcall_phase_present()


def get_reader(pgen_path, sample_subset=None):
    """"""
    if sample_subset is not None:
        sample_subset = np.array(sample_subset, dtype=np.uint32)
    reader = pg.PgenReader(pgen_path.encode(), sample_subset=sample_subset)
    if sample_subset is None:
        num_samples = reader.get_raw_sample_ct()
    else:
        num_samples = len(sample_subset)
    return reader, num_samples


def read(pgen_path, variant_idx, sample_subset=None, dtype=np.int8):
    """
    Get genotypes for a variant.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    variant_idx : int
        Variant index
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.
    dtype : np.int{8,32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotypes (as {0, 1, 2, -9}) for the selected variant and samples.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    genotypes = np.zeros(num_samples, dtype=dtype)
    with reader as r:
        r.read(np.array(variant_idx, dtype=np.uint32), genotypes)
    return genotypes


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
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    dosages = np.zeros(num_samples, dtype=dtype)
    with reader as r:
        r.read_dosages(np.array(variant_idx, dtype=np.uint32), dosages)
    return dosages


def read_alleles(pgen_path, variant_idx, sample_subset=None):
    """
    Get alleles for a variant.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    variant_idx : int
        Variant index
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.

    Returns
    -------
    alleles: ndarray (2 * sample_ct)
        Alleles for the selected variant and samples.
        Elements 2n and 2n+1 correspond to sample n.
        Both elements are -9 for missing genotypes.
        If the genotype is unphased, the lower index appears first.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    alleles = np.zeros(2*num_samples, dtype=np.int32)
    with reader as r:
        r.read_alleles(np.array(variant_idx, dtype=np.uint32), alleles)
    return alleles


def read_list(pgen_path, variant_idxs, sample_subset=None, dtype=np.int8):
    """
    Get genotypes for a list of variants.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    variant_idxs : array_like
        List of variant indexes
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.
    dtype : np.int{8,32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotypes for the selected variants and samples.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = len(variant_idxs)
    genotypes = np.zeros([num_variants, num_samples], dtype=dtype)
    with reader as r:
        r.read_list(np.array(variant_idxs, dtype=np.uint32), genotypes)
    return genotypes


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
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = len(variant_idxs)
    dosages = np.zeros([num_variants, num_samples], dtype=dtype)
    with reader as r:
        r.read_dosages_list(np.array(variant_idxs, dtype=np.uint32), dosages)
    return dosages


def read_alleles_list(pgen_path, variant_idxs, sample_subset=None):
    """
    Get alleles for a list of variants.

    Parameters
    ----------
    pgen_path : str
        Path of PLINK 2 pgen file
    variant_idxs : array_like
        List of variant indexes
    sample_subset : array_like
        List of sample indexes to select. Must be sorted.

    Returns
    -------
    alleles : ndarray
        Alleles for the selected variants and samples.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = len(variant_idxs)
    alleles = np.zeros([num_variants, 2*num_samples], dtype=np.int32)
    with reader as r:
        r.read_alleles_list(np.array(variant_idxs, dtype=np.uint32), alleles)
    return alleles


def read_range(pgen_path, start_idx, end_idx, sample_subset=None, dtype=np.int8):
    """
    Get genotypes for a range of variants.

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
    dtype : np.int{8,32,64}
        Data type of the returned array.

    Returns
    -------
    dosages : ndarray
        Genotypes for the selected variants and samples.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = end_idx - start_idx + 1
    genotypes = np.zeros([num_variants, num_samples], dtype=dtype)
    with reader as r:
        r.read_range(start_idx, end_idx+1, genotypes)
    return genotypes


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
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = end_idx - start_idx + 1
    dosages = np.zeros([num_variants, num_samples], dtype=dtype)
    with reader as r:
        r.read_dosages_range(start_idx, end_idx+1, dosages)
    return dosages


def read_alleles_range(pgen_path, start_idx, end_idx, sample_subset=None):
    """
    Get alleles for a range of variants.

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

    Returns
    -------
    alleles : ndarray
        Alleles for the selected variants and samples.
    """
    reader, num_samples = get_reader(pgen_path, sample_subset=sample_subset)
    num_variants = end_idx - start_idx + 1
    alleles = np.zeros([num_variants, 2*num_samples], dtype=np.int32)
    with reader as r:
        r.read_alleles_range(start_idx, end_idx+1, alleles)
    return alleles


def _impute_mean(genotypes):
    """Impute missing genotypes to mean"""
    m = genotypes == -9
    ix = np.nonzero(m)[0]
    if len(ix) > 0:
        a = genotypes.sum(1)
        b = m.sum(1)
        mu = (a + 9*b) / (genotypes.shape[1] - b)
        genotypes[m] = mu[ix]


class PgenReader(object):
    """

    To generate the pgen/psam/pvar files from a VCF, run
    plink2 --vcf ${vcf_file} 'dosage=DS' --output-chr chrM --out ${plink_prefix_path}
    """
    def __init__(self, plink_prefix_path, select_samples=None):

        self.pvar_df = read_pvar(f"{plink_prefix_path}.pvar")
        self.psam_df = read_psam(f"{plink_prefix_path}.psam")
        self.pgen_file = f"{plink_prefix_path}.pgen"

        self.num_variants = self.pvar_df.shape[0]
        self.variant_ids = self.pvar_df['id'].tolist()
        self.variant_idx_dict = {i:k for k,i in enumerate(self.variant_ids)}

        self.sample_id_list = self.psam_df.index.tolist()
        self.set_samples(select_samples)

    def set_samples(self, sample_ids=None, sort=True):
        """
        Set samples to load.

        Parameters
        ----------
        sample_ids : array_like
            List of samples to select.
        sort : bool
            Preserve sample order from pgen file.
        """
        if sample_ids is None:
            self.sample_ids = self.sample_id_list
            self.sample_idxs = None
        else:
            sample_idxs = [self.sample_id_list.index(i) for i in sample_ids]
            if sort:
                sidx = np.argsort(sample_idxs)
                sample_idxs = [sample_idxs[i] for i in sidx]
                sample_ids = [sample_ids[i] for i in sidx]
            self.sample_ids = sample_ids
            self.sample_idxs = sample_idxs

    def read(self, variant_id, impute_mean=True, dtype=np.float32):
        """Read genotypes (as 0,1,2,-9; impute missing values to mean)."""
        variant_idx = self.variant_idx_dict[variant_id]
        genotypes = read(self.pgen_file, variant_idx, sample_subset=self.sample_idxs,
                         dtype=np.int8).astype(dtype)
        if impute_mean:
            m = genotypes == -9
            if any(m):
                genotypes[m] = genotypes[~m].mean()
        return pd.Series(genotypes, index=self.sample_ids, name=variant_id)

    def read_dosages(self, variant_id, dtype=np.float32):
        variant_idx = self.variant_idx_dict[variant_id]
        dosages = read_dosages(self.pgen_file, variant_idx, sample_subset=self.sample_idxs, dtype=dtype)
        return pd.Series(dosages, index=self.sample_ids, name=variant_id)

    def read_list(self, variant_ids, impute_mean=True, dtype=np.float32):
        variant_idxs = [self.variant_idx_dict[i] for i in variant_ids]
        genotypes = read_list(self.pgen_file, variant_idxs, sample_subset=self.sample_idxs,
                              dtype=np.int8).astype(dtype)
        if impute_mean:
            _impute_mean(genotypes)
        return pd.DataFrame(genotypes, index=variant_ids, columns=self.sample_ids)

    def read_dosages_list(self, variant_ids, dtype=np.float32):
        variant_idxs = [self.variant_idx_dict[i] for i in variant_ids]
        dosages = read_dosages_list(self.pgen_file, variant_idxs, sample_subset=self.sample_idxs, dtype=dtype)
        return pd.DataFrame(dosages, index=variant_ids, columns=self.sample_ids)

    def read_alleles_list(self, variant_ids):
        variant_idxs = [self.variant_idx_dict[i] for i in variant_ids]
        alleles = read_alleles_list(self.pgen_file, variant_idxs, sample_subset=self.sample_idxs)
        df1 = pd.DataFrame(alleles[:,::2], index=variant_ids, columns=self.sample_ids)
        df2 = pd.DataFrame(alleles[:,1::2], index=variant_ids, columns=self.sample_ids)
        return df1, df2

    def load_genotypes_df(self):
        """Load all genotypes as np.int8, without imputing missing values."""
        genotypes = read_range(self.pgen_file, 0, self.num_variants-1, sample_subset=self.sample_idxs)
        return pd.DataFrame(genotypes, index=self.pvar_df['id'], columns=self.sample_ids)

    def load_dosages_df(self):
        """Load all dosages."""
        dosages = read_dosages_range(self.pgen_file, 0, self.num_variants-1, sample_subset=self.sample_idxs)
        return pd.DataFrame(dosages, index=self.pvar_df['id'], columns=self.sample_ids)

    def get_pairwise_ld(self, id1, id2):
        """Compute pairwise LD (R2) between (lists of) variants"""
        if isinstance(id1, str) and isinstance(id2, str):
            g1 = self.read(id1, dtype=np.float64)
            g2 = self.read(id2, dtype=np.float64)
            g1 -= g1.mean()
            g2 -= g2.mean()
            return (g1 * g2).sum()**2 / ( (g1**2).sum() * (g2**2).sum() )
        elif isinstance(id1, str):
            g1 = self.read(id1, dtype=np.float64)
            g2 = self.read_list(id2, dtype=np.float64)
            g1 -= g1.mean()
            g2 -= g2.values.mean(1, keepdims=True)
            return (g1 * g2).sum(1)**2 / ( (g1**2).sum() * (g2**2).sum(1) )
        elif isinstance(id2, str):
            g1 = self.read_list(id1, dtype=np.float64)
            g2 = self.read(id2, dtype=np.float64)
            g1 -= g1.values.mean(1, keepdims=True)
            g2 -= g2.mean()
            return (g1 * g2).sum(1)**2 / ( (g1**2).sum(1) * (g2**2).sum() )
        else:
            assert len(id1) == len(id2)
            g1 = self.read_list(id1, dtype=np.float64).values
            g2 = self.read_list(id2, dtype=np.float64).values
            g1 -= g1.mean(1, keepdims=True)
            g2 -= g2.mean(1, keepdims=True)
            return (g1 * g2).sum(1) ** 2 / ( (g1**2).sum(1) * (g2**2).sum(1) )


def load_dosages_df(plink_prefix_path, select_samples=None):
    """
    Load dosages for all variants and all/selected samples as a dataframe.

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
    p = Pgen(plink_prefix_path, select_samples=select_samples)
    return p.load_dosages_df()

import pandas as pd
import tempfile
import numpy as np
import subprocess
import gzip
import sys
import threading
import queue
import bisect
from pandas_plink import read_plink

try:
    import pgen
except ImportError as e:
    pgen = None


gt_to_dosage_dict = {'0/0':0, '0/1':1, '1/1':2, './.':np.NaN,
                     '0|0':0, '0|1':1, '1|0':1, '1|1':2, '.|.':np.NaN}


def _check_dependency(name):
    e = subprocess.call('which '+name, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if e!=0:
        raise RuntimeError('External dependency \''+name+'\' not installed')


def print_progress(k, n, entity):
    s = f'\r    processing {entity} {k}/{n}'
    if k == n:
        s += '\n'
    sys.stdout.write(s)
    sys.stdout.flush()


class BackgroundGenerator(threading.Thread):
    # Adapted from https://github.com/justheuristic/prefetch_generator
    def __init__(self, generator, max_prefetch=10):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class background:
    def __init__(self, max_prefetch=10):
        self.max_prefetch = max_prefetch
    def __call__(self,gen):
        def bg_generator(*args,**kwargs):
            return BackgroundGenerator(gen(*args,**kwargs), max_prefetch=self.max_prefetch)
        return bg_generator


#------------------------------------------------------------------------------
#  Functions for writing VCFs
#------------------------------------------------------------------------------
def _get_vcf_opener(vcfpath):
    if vcfpath.endswith('.vcf.gz'):
        return gzip.open(vcfpath, 'rt')
    else:
        return open(vcfpath)


def get_sample_ids(vcfpath):
    """Get sample IDs from VCF"""
    with _get_vcf_opener(vcfpath) as vcf:
        for header in vcf:
            if header[:2]=='##': continue
            break
    return header.strip().split('\t')[9:]


def parse_genotypes(x, field='GT'):
    """Convert list of genotypes (str) to np.float32"""
    if field=='GT':
        g = np.float32([gt_to_dosage_dict[i] for i in x])
    elif field=='DS':
        g = np.float32(x)
    return g


def _get_field_ix(line, field):
    """Get position of field ('GT' or 'DS') in FORMAT"""
    fmt = line[8].split(':')
    if field not in fmt:
        raise ValueError(f'FORMAT field does not contain {field}')
    return fmt.index(field)

#------------------------------------------------------------------------------
#  Functions for loading regions/variants from VCFs
#------------------------------------------------------------------------------
def _impute_mean(g, missing=-1, verbose=False):
    """Impute rows to mean (in place)"""
    if not g.dtype in [np.float32, np.float64]:
        raise ValueError('Input dtype must be np.float32 or np.float64')
    n = 0
    for i in np.where((g == missing).any(1))[0]:
        ix = g[i] == missing
        g[i][ix] = np.mean(g[i][~ix])
        n += 1
    if verbose and n > 0:
        print(f'    imputed at least 1 sample in {n}/{g.shape[0]} sites')


class PlinkReader(object):
    def __init__(self, plink_prefix_path, select_samples=None, include_variants=None,
                 exclude_variants=None, exclude_chrs=None, verbose=True, dtype=np.int8):
        """
        Class for reading genotypes from PLINK bed files

        plink_prefix_path: prefix to PLINK bed,bim,fam files
        select_samples: specify a subset of samples

        Notes:
          Use this command to convert a VCF to PLINK format:
            plink2 --make-bed \
                --output-chr chrM \
                --vcf ${plink_prefix_path}.vcf.gz \
                --out ${plink_prefix_path}

            If using plink v1, the --keep-allele-order flag must be included.

          Uses read_plink from pandas_plink.
        """

        self.bim, self.fam, self.bed = read_plink(plink_prefix_path, verbose=verbose)
        self.bed = 2 - self.bed  # flip allele order: PLINK uses REF as effect allele
        if dtype == np.int8:
            self.bed[np.isnan(self.bed)] = -1  # convert missing (NaN) to -1 for int8
        self.bed = self.bed.astype(dtype, copy=False)
        self.sample_ids = self.fam['iid'].tolist()
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.fam = self.fam.loc[ix]
            self.bed = self.bed[:,ix]
            self.sample_ids = self.fam['iid'].tolist()
        if include_variants is not None:
            m = self.bim['snp'].isin(include_variants).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        if exclude_variants is not None:
            m = ~self.bim['snp'].isin(exclude_variants).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        if exclude_chrs is not None:
            m = ~self.bim['chrom'].isin(exclude_chrs).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        self.n_samples = self.fam.shape[0]
        self.chrs = list(self.bim['chrom'].unique())
        self.variant_pos = {i:g['pos'] for i,g in self.bim.set_index('snp')[['chrom', 'pos']].groupby('chrom')}
        self.variant_pos_dict = self.bim.set_index('snp')['pos'].to_dict()

    def get_region_index(self, region_str, return_pos=False):
        s = region_str.split(':')
        chrom = s[0]
        c = self.bim[self.bim['chrom'] == chrom]
        if len(s) > 1:
            start, end = s[1].split('-')
            start = int(start)
            end = int(end)
            c = c[(c['pos'] >= start) & (c['pos'] <= end)]
        if return_pos:
            return c['i'].values, c.set_index('snp')['pos']
        else:
            return c['i'].values

    def get_region(self, region_str, sample_ids=None, impute=False, verbose=False, dtype=np.int8):
        """Get genotypes for a region defined by 'chr:start-end' or 'chr'"""
        ix, pos_s = self.get_region_index(region_str, return_pos=True)
        g = self.bed[ix, :].compute().astype(dtype)
        if sample_ids is not None:
            ix = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, ix]
        if impute:
            _impute_mean(g, verbose=verbose)
        return g, pos_s

    def get_genotypes(self, variant_ids, sample_ids=None, impute=False, verbose=False, dtype=np.int8):
        """Load genotypes for selected variant IDs"""
        c = self.bim[self.bim['snp'].isin(variant_ids)]
        g = self.bed[c.i.values, :].compute().astype(dtype)
        if sample_ids is not None:
            ix = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, ix]
        if impute:
            _impute_mean(g, verbose=verbose)
        return g, c.set_index('snp')['pos']

    def get_genotype(self, variant_id, sample_ids=None, impute=False, verbose=False, dtype=np.int8):
        """Load genotypes for a single variant ID as pd.Series"""
        g,_ = self.get_genotypes([variant_id], sample_ids=sample_ids, impute=impute, verbose=verbose, dtype=dtype)
        if sample_ids is None:
            return pd.Series(g[0], index=self.fam['iid'], name=variant_id)
        else:
            return pd.Series(g[0], index=sample_ids, name=variant_id)

    def load_genotypes(self):
        """Load all genotypes into memory, as pd.DataFrame"""
        return pd.DataFrame(self.bed.compute(), index=self.bim['snp'], columns=self.fam['iid'])


class Plink2Reader(object):
    def __init__(self, plink_prefix_path, verbose=True, dtype=np.float32):
        """
        Class for reading genotypes from PLINK 2 pgen files

        plink_prefix_path: prefix to PLINK pgen,psam,pvar files
        select_samples: specify a subset of samples

        Notes:
          Use this command to convert a VCF to PLINK format:
            plink2 \
                --output-chr chrM \
                --vcf ${plink_prefix_path}.vcf.gz \
                --out ${plink_prefix_path}
          To use dosages, run:
            plink2 \
                --output-chr chrM \
                --vcf ${plink_prefix_path}.vcf.gz 'dosage=DS' \
                --out ${plink_prefix_path}

          Requires pgenlib: https://github.com/chrchang/plink-ng/tree/master/2.0/Python
        """

        self.psam = pgen.read_psam(f"{plink_prefix_path}.psam")
        self.pvar = pgen.read_pvar(f"{plink_prefix_path}.pvar")
        self.pgen_file = f"{plink_prefix_path}.pgen"
        self.sample_ids = self.psam.index.tolist()
        self.n_samples = self.psam.shape[0]
        self.chrs = list(self.pvar['chrom'].unique())
        self.variant_pos = {i:g['pos'] for i,g in self.pvar.set_index('id')[['chrom', 'pos']].groupby('chrom')}
        self.variant_pos_dict = self.pvar.set_index('id')['pos'].to_dict()

    def get_sample_indexes(self, sample_ids):
        """Get sorted indexes corresponding to sample IDs"""
        sample_idxs = [self.sample_ids.index(i) for i in sample_ids]
        # sort sample IDs by pgen order
        sidx = np.argsort(sample_idxs)
        sample_idxs = [sample_idxs[i] for i in sidx]
        sample_ids_sorted = [sample_ids[i] for i in sidx]
        return sample_idxs, sample_ids_sorted

    def get_region_index(self, region_str, return_pos=False):
        s = region_str.split(':')
        chrom = s[0]
        c = self.pvar[self.pvar['chrom'] == chrom]
        if len(s) > 1:
            start, end = s[1].split('-')
            c = c[(c['pos'] >= int(start)) & (c['pos'] <= int(end))]
        if return_pos:
            return c.index.values, c.set_index('id')['pos']
        else:
            return c.index.values

    def get_region(self, region_str, sample_ids=None, impute=False, verbose=False, dtype=np.float32):
        """Get genotypes for a region defined by 'chr:start-end' or 'chr'"""
        ix, pos_s = self.get_region_index(region_str, return_pos=True)
        g = pgen.read_dosages_range(self.pgen_file, ix[0], ix[-1], dtype=dtype)
        if sample_ids is not None:
            ix = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, ix]
        if impute:
            _impute_mean(g, missing=-9, verbose=verbose)
        return g, pos_s

    def get_genotypes(self, variant_ids, sample_ids=None, impute=False, verbose=False, dtype=np.float32):
        """Load genotypes for selected variant IDs"""
        c = self.pvar[self.pvar['id'].isin(variant_ids)]
        g = pgen.read_dosages_list(self.pgen_file, c.index, dtype=dtype)
        if sample_ids is not None:
            ix = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, ix]
        if impute:
            _impute_mean(g, missing=-9, verbose=verbose)
        return g, c.set_index('id')['pos']

    def get_genotype(self, variant_id, sample_ids=None, impute=False, verbose=False, dtype=np.float32):
        """Load genotypes for a single variant ID as pd.Series"""
        g,_ = self.get_genotypes([variant_id], sample_ids=sample_ids, impute=impute, verbose=verbose, dtype=dtype)
        if sample_ids is None:
            return pd.Series(g[0], index=self.sample_ids, name=variant_id)
        else:
            return pd.Series(g[0], index=sample_ids, name=variant_id)

    def load_genotypes(self, sample_ids=None):
        """Load all genotypes into memory, as pd.DataFrame"""
        if sample_ids is not None:
            sample_idxs, sample_ids = self.get_sample_indexes(sample_ids)
        else:
            sample_idxs = None
            sample_ids = self.sample_ids

        return pd.DataFrame(pgen.read_dosages_range(self.pgen_file, 0, self.pvar.shape[0]-1, sample_subset=sample_idxs),
                            index=self.pvar['id'], columns=sample_ids)


def load_genotypes(plink_prefix_path, select_samples=None, dtype=np.int8):
    """Load all genotypes into a dataframe"""

    print('Loading genotypes ... ', end='', flush=True)

    # identify format
    if all([os.path.exists(f"{plink_prefix_path}.{ext}" for ext in ['bed', 'bim', 'fam'])]):
        pr = PlinkReader(plink_prefix_path, select_samples=select_samples, dtype=dtype)
        df = pr.load_genotypes()
    elif all([os.path.exists(f"{plink_prefix_path}.{ext}" for ext in ['pgen', 'psam', 'pvar'])]):
        if pgen is None:
            raise ImportError('The pgenlib package must be installed to use PLINK 2 pgen/psam/pvar files.')
        pr = Plink2Reader(plink_prefix_path)
        df = pr.load_genotypes(sample_ids=select_samples)
    else:
        raise ValueError(f"No compatible PLINK files found for {plink_prefix_path}")

    print('done.', flush=True)
    return df


def get_vcf_region(region_str, vcfpath, field='GT', sample_ids=None, select_samples=None, impute_missing=True):
    """Load VCF region (str: 'chr:start-end') as DataFrame (requires tabix)"""
    s = subprocess.check_output(f'tabix {vcfpath} {region_str}', shell=True)
    s = s.decode().strip().split('\n')
    s = [i.split('\t') for i in s]

    if sample_ids is None:
        sample_ids = get_sample_ids(vcfpath)
    variant_ids = [i[2] for i in s]
    pos_s = pd.Series([int(i[1]) for i in s], index=variant_ids)

    ix = _get_field_ix(s[0], field)
    g = np.array([parse_genotypes([i.split(':')[ix] for i in line[9:]], field=field) for line in s])
    df = pd.DataFrame(g, index=variant_ids, columns=sample_ids)

    if select_samples is not None:
        df = df[select_samples]

    if impute_missing:
        n = 0
        for v in df.values:
            ix = np.isnan(v)
            if np.any(ix):
                v[ix] = np.mean(v[~ix])
                n += 1
        if n > 0:
            print(f'    imputed at least 1 sample in {n} sites')

    return df, pos_s


def get_vcf_variants(variant_ids, vcfpath, field='GT', sample_ids=None):
    """Load a set of variants in VCF as DataFrame (requires tabix)"""
    variant_id_set = set(variant_ids)
    with tempfile.NamedTemporaryFile() as regions_file:
        df = pd.DataFrame([i.split('_')[:2] for i in variant_id_set], columns=['chr', 'pos'])
        df['pos'] = df['pos'].astype(int)
        df = df.sort_values(['chr', 'pos'])
        df.to_csv(regions_file.name, sep='\t', index=False, header=False)
        s = subprocess.check_output(f'tabix {vcfpath} --regions {regions_file.name}', shell=True)
    s = s.decode().strip().split('\n')
    s = [i.split('\t') for i in s]

    if sample_ids is None:
        sample_ids = get_sample_ids(vcfpath)

    ix = _get_field_ix(s[0], field)
    g = np.array([parse_genotypes([i.split(':')[ix] for i in line[9:]], field=field) for line in s])
    g = np.array([i for i in g if -1 not in i])  # filter missing here instead of ValueError?

    returned_variant_ids = [i[2] for i in s]
    ix = [k for k,i in enumerate(returned_variant_ids) if i in variant_id_set]
    g = np.array([g[i] for i in ix])
    returned_variant_ids = [returned_variant_ids[i] for i in ix]
    return pd.DataFrame(g.astype(np.float32), index=returned_variant_ids, columns=sample_ids)

#------------------------------------------------------------------------------
#  Generator classes for batch processing of genotypes/phenotypes
#------------------------------------------------------------------------------
class GenotypeGeneratorTrans(object):
    def __init__(self, genotype_df, batch_size=50000, chr_s=None):
        """
        Generator for iterating over all variants (trans-scan)

        Inputs:
          genotype_df: Dataframe with genotypes (variants x samples)
          batch_size: Batch size for GPU processing

        Generates: genotype array (2D), variant ID array
        """
        self.genotype_df = genotype_df
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.genotype_df.shape[0] / batch_size))
        self.batch_indexes = [[i*batch_size, (i+1)*batch_size] for i in range(self.num_batches)]
        self.batch_indexes[-1][1] = self.genotype_df.shape[0]
        if chr_s is not None:
            chroms, chr_ix = np.unique(chr_s, return_index=True)
            s = np.argsort(chr_ix)
            self.chroms = chroms[s]
            chr_ix = list(chr_ix[s]) + [chr_s.shape[0]]
            size_s = pd.Series(np.diff(chr_ix), index=self.chroms)
            self.chr_batch_indexes = {}
            for k,c in enumerate(self.chroms):
                num_batches = int(np.ceil(size_s[c] / batch_size))
                batch_indexes = [[chr_ix[k]+i*batch_size, chr_ix[k]+(i+1)*batch_size] for i in range(num_batches)]
                batch_indexes[-1][1] = chr_ix[k+1]
                self.chr_batch_indexes[c] = batch_indexes

    def __len__(self):
        return self.num_batches

    @background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False, enum_start=1):
        """Generate batches from genotype data"""
        if chrom is None:
            batch_indexes = self.batch_indexes
            num_batches = self.num_batches
        else:
            batch_indexes = self.chr_batch_indexes[chrom]
            num_batches = np.sum([len(i) for i in self.chr_batch_indexes.values()])

        for k,i in enumerate(batch_indexes, enum_start):  # loop through batches
            if verbose:
                print_progress(k, num_batches, 'batch')
            g = self.genotype_df.values[i[0]:i[1]]
            ix = self.genotype_df.index[i[0]:i[1]]  # variant IDs
            yield g, ix


class InputGeneratorCis(object):
    """
    Input generator for cis-mapping

    Inputs:
      genotype_df:      genotype DataFrame (genotypes x samples)
      variant_df:       DataFrame mapping variant_id (index) to chrom, pos
      phenotype_df:     phenotype DataFrame (phenotypes x samples)
      phenotype_pos_df: DataFrame defining position of each phenotype, with columns 'chr' and 'tss'
      window:           cis-window (selects variants within +- cis-window from TSS)

    Generates: phenotype array, genotype array (2D), cis-window indices, phenotype ID
    """
    def __init__(self, genotype_df, variant_df, phenotype_df, phenotype_pos_df, group_s=None, window=1000000):
        assert (genotype_df.index == variant_df.index).all()
        assert (phenotype_df.index == phenotype_df.index.unique()).all()
        self.genotype_df = genotype_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(variant_df.shape[0])
        self.n_samples = phenotype_df.shape[1]
        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df
        # check for constant phenotypes and drop
        m = np.all(phenotype_df.values == phenotype_df.values[:,[0]], 1)
        if m.any():
            print(f'    ** dropping {np.sum(m)} constant phenotypes')
            self.phenotype_df = self.phenotype_df.loc[~m]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[~m]
        self.group_s = None
        self.window = window

        self.phenotype_tss = self.phenotype_pos_df['tss'].to_dict()
        self.phenotype_chr = self.phenotype_pos_df['chr'].to_dict()
        variant_chrs = variant_df['chrom'].unique()
        phenotype_chrs = phenotype_pos_df['chr'].unique()
        self.chrs = [i for i in phenotype_chrs if i in variant_chrs]
        self.chr_variant_dfs = {c:g[['pos', 'index']] for c,g in self.variant_df.groupby('chrom')}

        # check phenotypes & calculate genotype ranges
        # get genotype indexes corresponding to cis-window of each phenotype
        valid_ix = []
        self.cis_ranges = {}
        for k,phenotype_id in enumerate(self.phenotype_df.index,1):
            if np.mod(k, 1000) == 0 or k == phenotype_df.shape[0]:
                print(f'\r  * checking phenotypes: {k}/{self.phenotype_df.shape[0]}',
                      end='' if k != phenotype_df.shape[0] else None)

            tss = self.phenotype_tss[phenotype_id]
            chrom = self.phenotype_chr[phenotype_id]

            m = len(self.chr_variant_dfs[chrom]['pos'].values)
            lb = bisect.bisect_left(self.chr_variant_dfs[chrom]['pos'].values, tss - self.window)
            ub = bisect.bisect_right(self.chr_variant_dfs[chrom]['pos'].values, tss + self.window)
            if lb != ub:
                r = self.chr_variant_dfs[chrom]['index'].values[[lb, ub - 1]]
            else:
                r = []

            if len(r) > 0:
                valid_ix.append(phenotype_id)
                self.cis_ranges[phenotype_id] = r

        if len(valid_ix) != self.phenotype_df.shape[0]:
            print('    ** dropping {} phenotypes without variants in cis-window'.format(
                  self.phenotype_df.shape[0] - len(valid_ix)))
            self.phenotype_df = self.phenotype_df.loc[valid_ix]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[valid_ix]
            self.phenotype_tss = self.phenotype_pos_df['tss'].to_dict()
            self.phenotype_chr = self.phenotype_pos_df['chr'].to_dict()
        self.n_phenotypes = self.phenotype_df.shape[0]
        if group_s is not None:
            self.group_s = group_s.loc[self.phenotype_df.index].copy()
            self.n_groups = self.group_s.unique().shape[0]


    @background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False, batch_size=1000):
        """
        Generate batches from genotype data

        Returns: phenotype array, genotype matrix, genotype index, phenotype ID(s), [group ID]
        """
        if chrom is None:
            phenotype_ids = self.phenotype_df.index
            chr_offset = 0
        else:
            phenotype_ids = self.phenotype_pos_df[self.phenotype_pos_df['chr'] == chrom].index
            if self.group_s is None:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'], return_index=True))}
            else:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'][self.group_s.drop_duplicates().index], return_index=True))}
            chr_offset = offset_dict[chrom]

        index_dict = {j:i for i,j in enumerate(self.phenotype_df.index)}

        if self.group_s is None:
            for k,phenotype_id in enumerate(phenotype_ids, chr_offset+1):
                if verbose:
                    print_progress(k, self.n_phenotypes, 'phenotype')
                p = self.phenotype_df.values[index_dict[phenotype_id]]
                # p = self.phenotype_df.values[k]
                r = self.cis_ranges[phenotype_id]
		
                yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), phenotype_id
        else:
            gdf = self.group_s[phenotype_ids].groupby(self.group_s, sort=False)
            for k,(group_id,g) in enumerate(gdf, chr_offset+1):
                if verbose:
                    print_progress(k, self.n_groups, 'phenotype group')
                # check that ranges are the same for all phenotypes within group
                assert np.all([self.cis_ranges[g.index[0]][0] == self.cis_ranges[i][0] and self.cis_ranges[g.index[0]][1] == self.cis_ranges[i][1] for i in g.index[1:]])
                group_phenotype_ids = g.index.tolist()
                # p = self.phenotype_df.loc[group_phenotype_ids].values
                p = self.phenotype_df.values[[index_dict[i] for i in group_phenotype_ids]]
                r = self.cis_ranges[g.index[0]]
                yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), group_phenotype_ids, group_id

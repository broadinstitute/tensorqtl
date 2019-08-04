import pandas as pd
import tempfile
import numpy as np
import subprocess
import gzip
import sys
import threading
import queue
from pandas_plink import read_plink


gt_to_dosage_dict = {'0/0':0, '0/1':1, '1/1':2, './.':np.NaN,
                     '0|0':0, '0|1':1, '1|0':1, '1|1':2, '.|.':np.NaN}


def _check_dependency(name):
    e = subprocess.call('which '+name, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if e!=0:
        raise RuntimeError('External dependency \''+name+'\' not installed')


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
        raise ValueError('FORMAT field does not contain {}'.format(field))
    return fmt.index(field)

#------------------------------------------------------------------------------
#  Functions for loading regions/variants from VCFs
#------------------------------------------------------------------------------
def _impute_mean(g, verbose=False):
    """Impute rows to mean (in place)"""
    n = 0
    for i in np.where(np.isnan(g).any(1))[0]:
        ix = np.isnan(g[i])
        g[i][ix] = np.mean(g[i][~ix])
        n += 1
    if verbose and n>0:
        print('    imputed at least 1 sample in {}/{} sites'.format(n, g.shape[0]))

class PlinkReader(object):
    def __init__(self, plink_prefix_path, select_samples=None, exclude_variants=None, verbose=True, dtype=np.float32):
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
        if dtype==np.int8:
            self.bed[np.isnan(self.bed)] = -1  # convert missing (NaN) to -1 for int8
        self.bed = self.bed.astype(dtype, copy=False)
        self.sample_ids = self.fam['iid'].tolist()
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.fam = self.fam.loc[ix]
            self.bed = self.bed[:,ix]
            self.sample_ids = self.fam['iid'].tolist()
        if exclude_variants is not None:
            m = ~self.bim['snp'].isin(exclude_variants).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        self.n_samples = self.fam.shape[0]
        self.variant_pos = {i:g['pos'] for i,g in self.bim.set_index('snp')[['chrom', 'pos']].groupby('chrom')}
        self.variant_pos_dict = self.bim.set_index('snp')['pos'].to_dict()

    def get_region_index(self, region_str, return_pos=False):
        s = region_str.split(':')
        chrom = s[0]
        c = self.bim[self.bim['chrom']==chrom]
        if len(s)>1:
            start, end = s[1].split('-')
            start = int(start)
            end = int(end)
            c = c[(c['pos']>=start) & (c['pos']<=end)]
        if return_pos:
            return c['i'].values, c.set_index('snp')['pos']
        else:
            return c['i'].values

    def get_region(self, region_str, impute=True, verbose=False):
        """Get genotypes for a region defined by 'chr:start-end' or 'chr'"""
        ix, pos_s = self.get_region_index(region_str, return_pos=True)
        g = self.bed[ix, :].compute()
        if impute:
            _impute_mean(g, verbose=verbose)
        return g, pos_s

    def get_genotypes(self, variant_ids, impute=False, verbose=False):
        """Load genotypes corresponding to variant IDs"""
        c = self.bim[self.bim['snp'].isin(variant_ids)]
        g = self.bed[c.i.values, :].compute()
        if impute:
            _impute_mean(g, verbose=verbose)
        return g, c.set_index('snp')['pos']

    def get_genotype(self, variant_id, impute=False, verbose=False):
        """Load genotype corresponding to variant ID as pd.Series"""
        g,_ = self.get_genotypes([variant_id], impute=impute, verbose=verbose)
        return pd.Series(g[0], index=self.fam['iid'])

    def get_all_genotypes(self, impute=False, verbose=False):
        """Load all genotypes into memory (impute=True should only be used for float formats)"""
        g = self.bed.compute()
        if impute:
            _impute_mean(g, verbose=verbose)
        return g


def load_genotypes(plink_prefix_path, select_samples=None, dtype=np.int8):
    pr = PlinkReader(plink_prefix_path, select_samples=select_samples, dtype=dtype)
    print('Loading genotypes ... ', end='', flush=True)
    df = pd.DataFrame(pr.get_all_genotypes(), index=pr.bim['snp'], columns=pr.fam['iid'])
    print('done.', flush=True)
    return df


def get_vcf_region(region_str, vcfpath, field='GT', sample_ids=None, select_samples=None, impute_missing=True):
    """Load VCF region (str: 'chr:start-end') as DataFrame (requires tabix)"""
    s = subprocess.check_output('tabix {} {}'.format(vcfpath, region_str), shell=True)
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
        if n>0:
            print('    imputed at least 1 sample in {} sites'.format(n))

    return df, pos_s


def get_vcf_variants(variant_ids, vcfpath, field='GT', sample_ids=None):
    """Load a set of variants in VCF as DataFrame (requires tabix)"""
    variant_id_set = set(variant_ids)
    with tempfile.NamedTemporaryFile() as regions_file:
        df = pd.DataFrame([i.split('_')[:2] for i in variant_id_set], columns=['chr', 'pos'])
        df['pos'] = df['pos'].astype(int)
        df = df.sort_values(['chr', 'pos'])
        df.to_csv(regions_file.name, sep='\t', index=False, header=False)
        s = subprocess.check_output('tabix {} --regions {}'.format(vcfpath, regions_file.name), shell=True)
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
    def __init__(self, genotype_df, batch_size=50000, chr_s=None, dtype=np.float32):
        """
        Generator for iterating over all variants (trans-scan)
        
        Inputs:
          genotypes:  Numpy array of genotypes (variants x samples)
                      (see PlinkReader.get_all_genotypes())
          batch_size: Batch size for GPU processing
          dtype:      Batch dtype (default: np.float32).
                      By default genotypes are stored as np.int8.
        """
        self.genotype_df = genotype_df
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.genotype_df.shape[0] / batch_size))
        self.batch_indexes = [[i*batch_size, (i+1)*batch_size] for i in range(self.num_batches)]
        self.batch_indexes[-1][1] = self.genotype_df.shape[0]
        self.dtype = dtype
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
    def generate_data(self, chrom=None):
        """Generate batches from genotype data"""
        if chrom is None:
            for k,i in enumerate(self.batch_indexes, 1):  # loop through batches
                g = self.genotype_df.values[i[0]:i[1]].astype(self.dtype)
                ix = self.genotype_df.index[i[0]:i[1]]  # variant IDs
                yield g, ix
        else:
            for k,i in enumerate(self.chr_batch_indexes[chrom], 1):
                g = self.genotype_df.values[i[0]:i[1]].astype(self.dtype)
                ix = self.genotype_df.index[i[0]:i[1]]
                yield g, ix


class InputGeneratorCis(object):
    """
    Input generator for cis-mapping

    Inputs:
      plink_reader:     PlinkReader object
      phenotype_df:     phenotype DataFrame (phenotypes x samples)
      phenotype_pos_df: DataFrame defining position of each phenotype, with columns 'chr' and 'tss'
      window:           cis-window (selects variants within +- cis-window from TSS)

    Generates: phenotype array, genotype array (2D), cis-window indices, phenotype ID
    """
    def __init__(self, plink_reader, phenotype_df, phenotype_pos_df, genotype_df=None, window=1000000):
        assert (phenotype_df.index==phenotype_df.index.unique()).all()
        self.plink_reader = plink_reader
        self.genotype_df = genotype_df
        self.n_samples = phenotype_df.shape[1]
        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df
        self.window = window

        self.n_phenotypes = phenotype_df.shape[0]
        self.phenotype_tss = phenotype_pos_df['tss'].to_dict()
        self.phenotype_chr = phenotype_pos_df['chr'].to_dict()

        num_var_chr = self.plink_reader.bim['chrom'].value_counts().to_dict()
        # indices of variants for each chromosome
        var_index_chr = {c:np.arange(num_var_chr[c]) for c in num_var_chr}

        # check phenotypes & calculate genotype ranges
        valid_ix = []
        self.cis_ranges = {}

        chr_size_s = plink_reader.bim['chrom'].value_counts()[plink_reader.bim['chrom'].unique()]
        if genotype_df is not None:
            chr_offset_s = pd.Series([0]+chr_size_s.iloc[:-1].tolist(), index=chr_size_s.index).cumsum()
        else:
            chr_offset_s = pd.Series(0, index=chr_size_s.index)

        for k,phenotype_id in enumerate(phenotype_df.index,1):
            if np.mod(k,1000)==0:
                print('\r  * checking phenotypes: {}/{}'.format(k, phenotype_df.shape[0]), end='')
            # find indices corresponding to cis-window of each gene
            tss = self.phenotype_tss[phenotype_id]
            chrom = self.phenotype_chr[phenotype_id]
            r = var_index_chr[chrom][
                (plink_reader.variant_pos[chrom].values>=tss-self.window) &
                (plink_reader.variant_pos[chrom].values<=tss+self.window)
            ] + chr_offset_s[chrom]  # offset is 0 if loading by chr
            if len(r)>0:
                valid_ix.append(phenotype_id)
                self.cis_ranges[phenotype_id] = [r[0],r[-1]]
        print('\r  * checking phenotypes: {}/{}'.format(k, phenotype_df.shape[0]))
        if len(valid_ix)!=phenotype_df.shape[0]:
            print('    ** dropping {} phenotypes without variants in cis-window'.format(
                phenotype_df.shape[0]-len(valid_ix)))
            self.phenotype_df = self.phenotype_df.loc[valid_ix]
            self.n_phenotypes = self.phenotype_df.shape[0]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[valid_ix]
            self.phenotype_tss = phenotype_pos_df['tss'].to_dict()
            self.phenotype_chr = phenotype_pos_df['chr'].to_dict()
        self.phenotype_values = self.phenotype_df.values
        self.loaded_chrom = ''

    @background(max_prefetch=6)
    def generate_data(self):
        if self.genotype_df is None:
            for k,phenotype_id in enumerate(self.phenotype_df.index):
                chrom = self.phenotype_chr[phenotype_id]
                if chrom != self.loaded_chrom:
                    # load genotypes into memory
                    print('  * loading genotypes')
                    self.chr_genotypes, self.chr_variant_pos = self.plink_reader.get_region(chrom, verbose=True)
                    self.loaded_chrom = chrom

                # return phenotype & its permutations in same array; fetch genotypes
                p = self.phenotype_values[k]
                r = self.cis_ranges[phenotype_id]
                yield p, self.chr_genotypes[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), phenotype_id
        else:
            for k,phenotype_id in enumerate(self.phenotype_df.index):
                # return phenotype & its permutations in same array; fetch genotypes
                p = self.phenotype_values[k]
                r = self.cis_ranges[phenotype_id]
                yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), phenotype_id

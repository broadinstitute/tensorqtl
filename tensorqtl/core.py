import pandas as pd
import torch
import sys


class SimpleLogger(object):
    def __init__(self, logfile=None, verbose=True):
        self.console = sys.stdout
        self.verbose = verbose
        if logfile is not None:
            self.log = open(logfile, 'w')
        else:
            self.log = None

    def write(self, message):
        if self.verbose:
            self.console.write(message+'\n')
        if self.log is not None:
            self.log.write(message+'\n')
            self.log.flush()

#------------------------------------------------------------------------------
#  Core classes/functions for mapping associations on GPU
#------------------------------------------------------------------------------
class Residualizer(object):
    def __init__(self, C_t):
        # center and orthogonalize
        self.Q_t, _ = torch.qr(C_t - C_t.mean(0))

    def transform(self, M_t, center=True):
        """Residualize rows of M wrt columns of C"""
        if center:
            M0_t = M_t - M_t.mean(1, keepdim=True)
        else:
            M0_t = M_t
        return M_t - torch.mm(torch.mm(M0_t, self.Q_t), torch.transpose(self.Q_t, 0, 1))  # keep original mean


def calculate_maf(genotype_t, alleles=2):
    """Calculate minor allele frequency"""
    af_t = genotype_t.sum(1) / (alleles * genotype_t.shape[1])
    return torch.where(af_t > 0.5, 1 - af_t, af_t)


def center_normalize(M_t, dim=0):
    """Center and normalize M"""
    N_t = M_t - M_t.mean(dim=dim, keepdim=True)
    return N_t / torch.sqrt(torch.pow(N_t, 2).sum(dim=dim, keepdim=True))


def calculate_corr(genotype_t, phenotype_t, residualizer, return_sd=False):
    """Calculate correlation between normalized residual genotypes and phenotypes"""
    # residualize
    genotype_res_t = residualizer.transform(genotype_t)  # variants x samples
    phenotype_res_t = residualizer.transform(phenotype_t)  # phenotypes x samples

    if return_sd:
        gstd = genotype_res_t.var(1)
        pstd = phenotype_res_t.var(1)

    # center and normalize
    genotype_res_t = center_normalize(genotype_res_t, dim=1)
    phenotype_res_t = center_normalize(phenotype_res_t, dim=1)

    # correlation
    if return_sd:
        return torch.mm(genotype_res_t, torch.transpose(phenotype_res_t, 0, 1)), torch.sqrt(pstd / gstd)
    else:
        return torch.mm(genotype_res_t, torch.transpose(phenotype_res_t, 0, 1))


#------------------------------------------------------------------------------
#  i/o functions
#------------------------------------------------------------------------------
def read_phenotype_bed(phenotype_bed):
    """Load phenotype BED file as phenotype and TSS DataFrames"""
    if phenotype_bed.endswith('.bed.gz'):
        phenotype_df = pd.read_csv(phenotype_bed, sep='\t', index_col=3, dtype={'#chr':str, '#Chr':str})
    elif phenotype_bed.endswith('.parquet'):
        phenotype_df = pd.read_parquet(phenotype_bed)
    else:
        raise ValueError('Unsupported file type.')
    phenotype_df = phenotype_df.rename(columns={i:i.lower() for i in phenotype_df.columns[:3]})
    phenotype_pos_df = phenotype_df[['#chr', 'end']].rename(columns={'#chr':'chr', 'end':'tss'})
    phenotype_df = phenotype_df.drop(['#chr', 'start', 'end'], axis=1)
    return phenotype_df, phenotype_pos_df

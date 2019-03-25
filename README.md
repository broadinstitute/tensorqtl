## tensorQTL

tensorQTL is a GPU-based QTL mapper, enabling ~200-300 fold faster *cis*- and *trans*-QTL mapping compared to CPU-based implementations.

If you use tensorQTL in your research, please cite the following paper:
[Taylor-Weiner, Aguet, et al., bioRxiv, 2019](https://www.biorxiv.org/content/10.1101/470138v3).

Empirical beta-approximated p-values are computed as described in [FastQTL](http://fastqtl.sourceforge.net/) ([Ongen et al., 2016](https://academic.oup.com/bioinformatics/article/32/10/1479/1742545)).

### Install

Run the following commands to install tensorQTL:
```
$ git clone git@github.com:broadinstitute/tensorqtl.git
$ cd tensorqtl
# set up virtual environment and install
$ virtualenv venv
$ source venv/bin/activate
(venv)$ pip install -r install/requirements.txt .
```
<!-- `pip install tensorqtl` -->

### Requirements

tensorQTL requires an environment configured with a GPU. Instructions for setting up a virtual machine on Google Cloud Platform are provided [here](install/INSTALL.md).

### Input formats

tensorQTL requires three input files: genotypes, phenotypes, and covariates. Phenotypes must be provided in BED format (phenotypes x samples), and covariates as a text file (covariates x samples). Both are in the format used by [FastQTL](http://fastqtl.sourceforge.net/). Genotypes must currently be in [PLINK](https://www.cog-genomics.org/plink/2.0/) format, and can be converted as follows:
```
plink2 --make-bed \
    --output-chr chrM \
    --vcf ${plink_prefix_path}.vcf.gz \
    --out ${plink_prefix_path}
```

### Examples
For examples illustrating *cis*- and *trans*-QTL mapping, please see [tensorqtl_examples.ipynb](example/tensorqtl_examples.ipynb).

### Running tensorQTL from the command line
This section describes how  to run tensorQTL from the command line. For a full list of options, run
```
python3 -m tensorqtl --help
```

#### *cis*-QTL mapping
Phenotype-level summary statistics with empirical p-values:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --mode cis
```
All variant-phenotype associations:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --mode cis_nominal
```
This will generate a [parquet](https://parquet.apache.org/) file for each chromosome. These files can be read using `pandas`:
```
import pandas as pd
df = pd.read_parquet(file_name)
```
Conditionally independent *cis*-QTL (as described in [GTEx Consortium, 2017](https://www.nature.com/articles/nature24277)):
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --cis_results ${cis_results_file} \
    --mode cis_independent
```

#### *trans*-QTL mapping
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --mode trans
```
For *trans*-QTL mapping, tensorQTL generates sparse output by default (associations with p-value < 1e-5). *cis*-associations are filtered out. The output is in parquet format, with four columns: phenotype_id, variant_id, pval, maf.

### Running tensorQTL as a Python module
TensorQTL can also be run as a module to more efficiently run multiple analyses:
```
import pandas as pd
import tensorqtl
from tensorqtl import genotypeio
```
#### Loading input files
Load phenotypes and covariates:
```
phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(phenotype_bed_file)
covariates_df = pd.read_csv(covariates_file, sep='\t', index_col=0).T  # samples x covariates
```
Genotypes can be loaded as follows, where `plink_prefix_path` is the path to the VCF in PLINK format:
```
# for VCFs with hard GT calls only, specify type as np.int8 to save memory:
pr = genotypeio.PlinkReader(plink_prefix_path, dtype=np.int8)
# load all genotypes into a pd.DataFrame
genotype_df = pd.DataFrame(pr.get_all_genotypes(), index=pr.bim['snp'], columns=pr.fam['iid'])
```
To save memory when using genotypes for a subset of samples, specify the samples as follows (this is not strictly necessary, since tensorQTL will select the relevant samples from `genotype_df` otherwise):
```
pr = genotypeio.PlinkReader(plink_prefix_path, select_samples=phenotype_df.columns, dtype=np.int8)
```
#### *cis*-QTL mapping
```
cis_df = tensorqtl.map_cis(pr, phenotype_df, phenotype_pos_df, covariates_df, genotype_df=genotype_df)
tensorqtl.calculate_qvalues(qtl_df, qvalue_lambda=0.85)
```
#### *trans*-QTL mapping
```
trans_df = tensorqtl.map_trans(genotype_df, phenotype_df, covariates_df,
                               return_sparse=True, batch_size=50000)
```

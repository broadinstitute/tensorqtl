## tensorQTL

tensorQTL is a GPU-based QTL mapper, enabling ~200-300 fold faster *cis*- and *trans*-QTL mapping compared to CPU-based implementations.

For *cis*-QTL mapping, beta-approximated empirical p-values are computed as implemented in [FastQTL](http://fastqtl.sourceforge.net/) ([Ongen et al., 2016](https://academic.oup.com/bioinformatics/article/32/10/1479/1742545)).

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

### Running tensorQTL
tensorQTL can be run from the command line. For options, see `tensorqtl.py --help`.

#### *cis*-QTL mapping
Phenotype-level summary statistics with empirical p-values:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} --covariates ${covariates_file} --mode cis
```
All variant-phenotype associations:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} --covariates ${covariates_file} --mode cis_nominal
```
This will generate a [parquet](https://parquet.apache.org/) file for each chromosome. The output can easily be read using `pandas`:
```
import pandas as pd
df = pd.read_parquet(file_name)
```

#### *trans*-QTL mapping
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} --covariates ${covariates_file} --mode trans
```
For *trans*-QTL mapping, tensorQTL generates sparse output by default (associations with p-value < 1e-5). *cis*-associations are filtered out. The output is written in parquet format, with four columns: phenotype_id, variant_id, pval, maf

### Output files
#### Mode `cis_nominal`
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`variant_id` | Variant ID
`start_distance` | Distance between the variant and phenotype start position (e.g., TSS)
`end_distance` | Distance between the variant and phenotype end position (only present if different from start position)
`af` | In-sample ALT allele frequency of the variant
`ma_samples` | Number of samples carrying at least on minor allele
`ma_count` | Number of minor alleles
`pval_nominal` | Nominal p-value of the association between the phenotype and variant
`slope` | Regression slope
`slope_se` | Standard error of the regression slope

#### Mode `cis_nominal`, with interaction term
When an interaction term is included, the output additionally contains the following columns instead of `pval_nominal`, `slope`, `slope_se`:
Column | Description
--- | ---
`pval_g` | Nominal p-value of the genotype term
`b_g` | Slope of the genotype term
`b_g_se` | Standard error of `b_g`
`pval_i` | Nominal p-value of the interaction variable
`b_i` | Slope of the interaction variable
`b_i_se` | Standard error of `b_i`
`pval_gi` | Nominal p-value of the interaction term
`b_gi` | Slope of the interaction term
`b_gi_se` | Standard error of `b_gi`
`tests_emt` | Effective number of independent variants (M<sub>eff</sub>) estimated by eigenMT
`pval_emt` | Bonferroni-adjusted `pval_gi` (i.e., multiplied by M<sub>eff</sub>)
`pval_adj_bh` | Benjamini-Hochberg adjusted `pval_emt`

#### Mode `cis`
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`num_var` | Number of variants in *cis*-window
`beta_shape1` | Parameter of the fitted Beta distribution
`beta_shape2` | Parameter of the fitted Beta distribution
`true_df` | Degrees of freedom used to compute p-values
`pval_true_df` | Nominal p-value based on `true_df`
`variant_id` | Variant ID
`start_distance` | Distance between the variant and phenotype start position (e.g., TSS)
`end_distance` | Distance between the variant and phenotype end position (only present if different from start position)
`ma_samples` | Number of samples carrying at least on minor allele
`ma_count` | Number of minor alleles
`af` | In-sample ALT allele frequency of the variant
`pval_nominal` | Nominal p-value of the association between the phenotype and variant
`slope` | Regression slope
`slope_se` | Standard error of the regression slope
`pval_perm` | Empirical p-value from permutations
`pval_beta` | Beta-approximated empirical p-value
`qval` | Storey q-value corresponding to `pval_beta`
`pval_nominal_threshold` | Nominal p-value threshold for significant associations with the phenotype

#### Mode `cis_independent`
The columns are the same as for `cis`, excluding `qval` and `pval_nominal_threshold`, and adding:
Column | Description
--- | ---
`rank` | Rank of the variant for the phenotype

#### Mode `trans`
Column | Description
--- | ---
`variant_id` | Variant ID
`phenotype_id` | Phenotype ID
`pval` | Nominal p-value of the association between the phenotype and variant
`b` | Regression slope
`b_se` | Standard error of the regression slope
`af` | In-sample ALT allele frequency of the variant

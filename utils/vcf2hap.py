import pandas as pd
import numpy as np
import gzip
import subprocess
import argparse

gt_dict = {'0':np.int8(0), '1':np.int8(1), '.':np.int8(-1)}


def parse_vcf(vcf):
    """
    assuming 0|0, 0|1, 1|0, 1|1, .|., code per-haplotype dosages as 0, 1, or -1 (missing)
    """

    nvariants = int(subprocess.check_output('bcftools index -n {}'.format(vcf), shell=True).decode())
    sample_ids = subprocess.check_output('bcftools query -l {}'.format(vcf), shell=True).decode().strip().split()
    nsamples = len(sample_ids)
    print('Parsing VCF with {} variants and {} samples'.format(nvariants, nsamples))

    hap1 = np.zeros((nvariants, nsamples), dtype=np.int8)
    hap2 = np.zeros((nvariants, nsamples), dtype=np.int8)
    variant_ids = []

    with gzip.open(vcf, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            break

        # parse GT index from first line
        line = line.strip().split('\t')
        gt_ix = line[8].split(':').index('GT')
        variant_ids.append(line[2])
        d = [i.split(':')[gt_ix].split('|') for i in line[9:]]
        hap1[0] = [gt_dict[i[0]] for i in d]
        hap2[0] = [gt_dict[i[1]] for i in d]

        for k,line in enumerate(f,1):
            line = line.strip().split('\t')
            variant_ids.append(line[2])
            d = [i.split(':')[gt_ix].split('|') for i in line[9:]]
            hap1[k] = [gt_dict[i[0]] for i in d]
            hap2[k] = [gt_dict[i[1]] for i in d]

            if np.mod(k,1000)==0:
                print('\rVariants parsed: {}'.format(k), end='')
        print('Variants parsed: {}'.format(k+1))

    hap1_df = pd.DataFrame(hap1, index=variant_ids, columns=sample_ids)
    hap2_df = pd.DataFrame(hap2, index=variant_ids, columns=sample_ids)
    return hap1_df, hap2_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert VCF to haplotype dosages, as two separate Parquet files.')
    parser.add_argument('vcf', help='Phased VCF file')
    parser.add_argument('-o', '--output_dir', default='.', help='Output directory')
    args = parser.parse_args()

    hap1_df, hap2_df = parse_vcf(args.vcf)

    prefix = os.path.basename(args.vcf).replace('.vcf.gz','')
    hap1_df.to_parquet(os.path.join(args.output_dir, prefix+'.hap1.parquet'))
    hap2_df.to_parquet(os.path.join(args.output_dir, prefix+'.hap2.parquet'))

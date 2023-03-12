# Data Extraction and Processing

## Step 1

Run `01_extract_data_single_drug.py` to get a dataframe of the MICs, VCF (full, gzipped) file paths, and other metadata for each isolate. This script does the following:

1. Get all isolates with MIC data and full VCF files available in the rollingDB database. 
2. Standardize MICs and bounds, and remove non-CRyPTIC isolates for which only one MIC was tested.

## Step 2

Run `02_training_data_vcf_processing.sh` to filter VCF files, annotate them, and extract lineages. It requires as an argument a text file of all VCF (full, gzipped) file paths, which was created in step 1. 

First, the filtered VCF files are saved to `/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF` (if a file is already present, a new one is not processed to save time). 

Next, the script annotates all VCF files in this directory using snpEff. 

Last, the script runs `fast-lineage-caller` on all the VCF files for this particular drug (may not be all the VCF files in `/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF`) and adds the lineages to the same dataframe created in step 1. 

## Step 3

Run `03_clean_data.py` to do the following:

1. Remove isolates with multiple lineages (they may have lots of ambiguous calls, or be a mixed sample).
2. Remove isolates with category 1 mutations and MICs < 1/2 of the CC. 

Isolates that meet these criteria may have genotypic or phenotypic mislabeling, so this step cleans the data further so that they don't artificially increase the error of our models. 

## Step 4

Run `04_extract_validation_data.py`

## Step 5

Run `QC_scripts/check_pass_proportion.sh` on BOTH dataframes of the training and validation datasets for each drug. This script will write a text file, where each line contains a sample ID (ROLLINGDB_ID) and the proportion of variants in the region of interest that have the PASS or Amb flags, which are indicative of good sequencing quality. Any other flag (including a flag that CONTAINS "Amb", but also contains another string like "Del") indicative low coverage or low sequencing quality.

The two text files written from `QC_scripts/check_pass_proportion.sh` for the training and validation datasets are required for the next script `05_combine_datasets.py` to run. When combining the datasets, it will exclude isolates that contain at least 25% low coverage calls in the region of interest. 

## Step 6

Run `05_combine_datasets.py` to exclude isolates with less than 75% PASS or Amb variant calls in the region of interest and write a combined txt file of paths "combined_paths_for_aln.txt" in the output directory. This is the file that should be passed into `06_make_MSA.py`. The script creates a new file `validtion_data_for_model.csv` to reflect the dropped isolates due to too many low coverage calls. In addition, it further cleans up the training dataset by:

1. Removing isolates that have fewer than 75% PASS or Amb calls in the region of interest.
2. Removing isolates that are all of the same lineage and have the same binary resistance phenotype. 
2. Splitting the data into train and test data sets, stratifying on binary phenotype and lineage. 

It then rewrites `data_for_model.csv`. 

## Note on other QC scripts

`check_read_depths.sh`


Run `04_make_MSA.py` to make a multiple sequence alignment of the region of interest. The script inserts SNPs, MNPs, and indels that pass quality control into the H37Rv reference sequence (NC_000962.3) for each isolate in the dataset and outputs . 

The thresholds for meeting quality control are SNP quality > 10, PASS or Amb FILTER with an alternative allele fraction in the range [0.25, 0.75], no imprecise structural variants, no heterogeneous alternative alleles, not a low coverage region…

SNPs and MNPs that did not pass quality control were inserted as ambiguous nucleotides (N), but indels that did not meet quality control were not inserted. Complex variants (reference and alternative alleles are of different lengths, and both are longer than 1 nucleotide) that passed quality control were also not inserted. 


## Running SnpEff

Instructions are in `snpEff_instructions.md`. Follow them to annotate VCF files with mutation effects.

## Training CNNs on Subdistributions

If CNNs are only trained on part of a drug's MIC distribution, how well can they predict MICs that are outside of this distribution?

The script `create_subdistributions.py` creates subdistributions (based on user-selected bounds) and saves them to an output phenotype file. Because the training scripts are based on a phenotypes dataframe and select isolates based on their presence in the dataframe, this is the easiest to achieve this step.
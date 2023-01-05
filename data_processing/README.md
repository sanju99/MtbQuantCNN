# Data Extraction and Processing

## Step 1

Run `01_extract_data_single_drug.py` to get a dataframe of the MICs, VCF (full, gzipped) file paths, and other metadata for each isolate. This script does the following:

1. Get all isolates with MIC data and full VCF files available in the rollingDB database. 
2. Standardize MICs and bounds, and remove non-CRyPTIC isolates for which only one MIC was tested.

## Step 2

Run `02_vcf_processing.sh` to filter VCF files, annotate them, and extract lineages. It requires as an argument a text file of all VCF (full, gzipped) file paths, which was created in step 1. 

First, the filtered VCF files are saved to `/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF` (if a file is already present, a new one is not processed to save time). 

Next, the script annotates all VCF files in this directory using snpEff. 

Last, the script runs `fast-lineage-caller` on all the VCF files for this particular drug (may not be all the VCF files in `/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF`) and adds the lineages to the same dataframe created in step 1. 

## Step 3

Run `03_clean_data.py` to do the following:

1. Remove isolates with multiple lineages (they may have lots of ambiguous calls, or be a mixed sample).
2. Remove isolates with category 1 mutations and MICs < 1/2 of the CC. 
3. Split data into train and test data sets, stratifying on MIC and primary lineage. 

Isolates that meet the top 2 criteria may have genotypic or phenotypic mislabeling, so this step cleans the data further so that they don't artificially increase the error of our models. 

## Step 4

Run `04_make_MSA.py`, which is a SNP concateantor script. It inserts SNPs and high-quality indels into the H37Rv reference sequence for each isolate in the dataset and outputs a multiple sequence alignment. 

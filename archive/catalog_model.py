import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sparse, sys, warnings, yaml, vcf, pickle, shutil, subprocess, argparse
import scipy.optimize

sys.path.append("utils")
from data_utils import *

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"

# who_variants_v1 = pd.read_csv("./data_processing/data_utils/WHO_catalog_V1.csv")
who_variants_V2 = pd.read_csv("./data_processing/data_utils/WHO_catalog_V2.csv", header=[2]).reset_index(drop=True)

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
cc = kwargs['binary_thresh']

output_path = f"{results_dir}/{drug}"
variants_fName = f"{data_dir}/{drug}/group_1_2_candidate_variants.tsv"

# get test set isolates
df_test = pd.read_csv(os.path.join(data_dir, drug, "data_for_model.csv")).query("category=='test_set'").reset_index(drop=True)
df_test[['ROLLINGDB_ID']].to_csv(f"{data_dir}/{drug}/test_set_isolates.txt", sep='\t', header=None, index=False)

    
###################################### STEP 1: PREPARE FILES FOR SNPSIFT EXTRACTION ######################################
    
    
# get group 1-2 variants
drug_group12_variants = who_variants_V2.loc[(who_variants_V2['drug']==abbr_drug_dict[drug]) & (who_variants_V2['FINAL CONFIDENCE GRADING'].str.contains('Assoc w R'))]

R_assoc_variants = drug_group12_variants['variant'].values
R_assoc_genes = drug_group12_variants['gene'].unique()

R_noncoding_pos_df = drug_group12_variants.loc[(drug_group12_variants['genomic position'] != '(see "Genomic_coordinates" sheet)')][['gene', 'variant', 'genomic position']].reset_index(drop=True).rename(columns={'genomic position': 'POS'})
R_noncoding_pos_df['POS'] = R_noncoding_pos_df['POS'].astype(int)
R_noncoding_pos_df['sense'] = R_noncoding_pos_df['gene'].map(dict(zip(h37Rv_genes['Symbol'], h37Rv_genes['Strand'])))

# create columns for REF and ALT for noncoding mutations only. This is to look for them later
for i, row in R_noncoding_pos_df.iterrows():

    if '>' in row['variant']:
        if row['sense'] == '+':
            R_noncoding_pos_df.loc[i, 'REF'] = row['variant'].split('>')[0][-1]
            R_noncoding_pos_df.loc[i, 'ALT'] = row['variant'].split('>')[-1]
        else:
            R_noncoding_pos_df.loc[i, 'REF'] = reverse_complement(row['variant'].split('>')[0][-1])
            R_noncoding_pos_df.loc[i, 'ALT'] = reverse_complement(row['variant'].split('>')[-1])
    else:
        R_noncoding_pos_df.loc[i, ['REF', 'ALT']] = [np.nan, np.nan]

print(f"Getting catalog variant data for {len(df_test)} test set isolates if they have any of {len(R_assoc_variants)} Group 1/2 variants across {len(R_assoc_genes)} genes and {R_noncoding_pos_df['POS'].nunique()} noncoding positions")

pd.Series(R_assoc_genes).to_csv(f"{data_dir}/{drug}/group_1_2_genes.txt", index=False, sep='\t', header=None)
pd.Series(R_noncoding_pos_df['POS'].unique()).to_csv(f"{data_dir}/{drug}/group_1_2_noncoding_pos.txt", index=False, sep='\t', header=None)
    

###################################### STEP 2 EXTRACT CANDIDATE GROUP 1/2 VARIANTS WITH SNPSIFT ######################################


if not os.path.isfile(variants_fName):
    print("Creating file of candidate Group 1/2 variants...")        
    subprocess.run(f"bash /home/sak0914/MtbQuantCNN/data_processing/prepare_model_inputs/05_get_WHO_R_Assoc_mutations.sh {drug}", shell=True)
    
df_variants = pd.read_csv(variants_fName, sep='\t')

for i, row in df_variants.iterrows():

    if pd.isnull(row['PROT']) or row['PROT'] == '.':
        df_variants.loc[i, "MUTATION"] = row['GENE']+ '_' + row['NUC']
    else:
        df_variants.loc[i, "MUTATION"] = row['GENE']+ '_' + row['PROT']

# get protein-coding variants
df_variants['WHO_Group12_Variant'] = df_variants['MUTATION'].isin(R_assoc_variants).values.astype(int)

# get the isolates that have the noncoding mutations with high confidence -- these will be predicted resistant by the catalog
if len(R_noncoding_pos_df) > 0:
    
    isolates_with_group1_2_noncoding_mutations = df_variants[['ISOLATE', 'POS', 'REF', 'ALT']].merge(R_noncoding_pos_df[['variant', 'POS', 'REF', 'ALT']], on=['POS', 'REF', 'ALT'], how='inner').ISOLATE.values

    # get non-coding variants
    df_variants.loc[df_variants['ISOLATE'].isin(isolates_with_group1_2_noncoding_mutations), 'WHO_Group12_Variant'] = 1

# save to the same file -- so now we have the data for the test set isolates that would be predicted resistant by the catalog
df_variants.to_csv(variants_fName, sep='\t', index=False)


###################################### STEP 3: GET PREDICTIONS USING THE CATALOG MUTATIONS ######################################


# get test set isolates
df_variants = pd.read_csv(variants_fName, sep='\t')

# isolates that have a WHO Group 1-2 variant are predicted resistant
df_test.loc[df_test['ROLLINGDB_ID'].isin(df_variants.query("WHO_Group12_Variant==1").ISOLATE), 'y_pred'] = 1
df_test['y_pred'] = df_test['y_pred'].fillna(0).astype(int)
df_test['y_test'] = (df_test[f'{drug}_upper_bound'] > cc).astype(int)

df_test.to_csv(os.path.join(output_path, "catalog_test_predictions.csv"), index=False)
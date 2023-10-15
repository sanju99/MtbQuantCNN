import pandas as pd
import numpy as np
import sys, vcf, glob, os, yaml
from sklearn.model_selection import train_test_split
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
# genes_list = kwargs["genes_list"]
binary_thresh = kwargs["binary_thresh"]

h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"
    
# if not os.path.isdir(f"{out_dir}/VCF_QC_files"):
#     os.makedirs(f"{out_dir}/VCF_QC_files")

if not os.path.isdir(f"{out_dir}/fastas"):
    os.makedirs(f"{out_dir}/fastas")

# validation data for a single drug
df_train = pd.read_csv(os.path.join(out_dir, "data_intermediate_clean.csv"))

# some drugs: i.e. PZA, there are no MICs from the MIC-ML dataset
if os.path.isfile(os.path.join(out_dir, "validation_data_for_model.csv")):
    df_val = pd.read_csv(os.path.join(out_dir, "validation_data_for_model.csv"))
    val_present = True
else:
    df_val = pd.DataFrame(columns=[])
    val_present = False

print(f"Original: {df_train.shape[0]} samples in the training data")

lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv")


# ################## STEP 3: REMOVE ISOLATES WITH A LARGE PROPORTION OF NON-PASS, NON-AMB VARIANTS IN THE REGION OF INTEREST ##################


# def get_samples_PASS_prop(fName):
    
#     with open(fName, "r+") as file:
#         lines = file.readlines()
        
#     PASS_prop = pd.DataFrame(columns=["ROLLINGDB_ID", "PASS_prop"])

#     for i, val in enumerate(lines):

#         sample_id, prop = val.strip("\n").split(" ")

#         # # no variants in the region of interest
#         # if prop == "":
#         #     prop = np.nan

#         PASS_prop.loc[i, :] = [sample_id, float(prop)]
        
#     # returns a dataframe, where the second column is the proportion of variants that are PASS or Amb
#     return PASS_prop


# found_loci = 0

# for gene in genes_list:

#     START, END = h37Rv_genes.query("Symbol==@gene")[["Start", "End"]].values[0]
    
#     if os.path.isfile(f"{out_dir}/VCF_QC_files/{gene}_training_PASS_prop.txt"):
#         print(f"Found {out_dir}/VCF_QC_files/{gene}_training_PASS_prop.txt")
#         found_loci += 1
#     else:
#         print(f"bash data_processing/QC_scripts/check_pass_proportion.sh {os.path.join(out_dir, 'data_intermediate_clean.csv')} {out_dir}/VCF_QC_files/{gene}_training_PASS_prop.txt {vcf_dir} {START} {END}\n")

# # if all are not found, then don't keep running the script because it will cause errors
# if found_loci < len(genes_list):
#     print(f"Only found {found_loci}/{len(genes_list)} training_PASS_prop.txt files")
#     exit()

    

# for gene in genes_list:
    
#     # get the dataframe of proportions of PASS/Amb calls in the alignment region
#     training_PASS_prop_df = get_samples_PASS_prop(f"{out_dir}/VCF_QC_files/{gene}_training_PASS_prop.txt")
    
#     if len(set(df_train.ROLLINGDB_ID) - set(training_PASS_prop_df.ROLLINGDB_ID)) > 0:
#         raise ValueError(f"Incorrect sample numbers in the {gene} training PASS prop file")
        
#     drop_train_samples = list(set(df_train["ROLLINGDB_ID"].values).intersection(training_PASS_prop_df.query("PASS_prop < 0.75")["ROLLINGDB_ID"].values))
    
#     print(f"Removed {len(drop_train_samples)}/{len(df_train)} training isolates with less than 75% PASS or Amb calls in {gene}")
#     df_train = df_train.query("ROLLINGDB_ID not in @drop_train_samples")

    

# if val_present:

#     found_loci = 0

#     for gene in genes_list:

#         START, END = h37Rv_genes.query("Symbol==@gene")[["Start", "End"]].values[0]
        
#         if os.path.isfile(f"{out_dir}/VCF_QC_files/{gene}_validation_PASS_prop.txt"):
#             print(f"Found {out_dir}/VCF_QC_files/{gene}_validation_PASS_prop.txt")
#             found_loci += 1
#         else:
#             print(f"bash data_processing/QC_scripts/check_pass_proportion.sh {os.path.join(out_dir, 'validation_data.csv')} {out_dir}/VCF_QC_files/{gene}_validation_PASS_prop.txt {vcf_dir} {START} {END}\n")
    
# # if all are not found, then don't keep running the script because it will cause errors
# if found_loci < len(genes_list):
#     print(f"Only found {found_loci}/{len(genes_list)} validation_PASS_prop.txt files")
#     exit()


# if val_present:

#     for gene in genes_list:
        
#         validation_PASS_prop_df = get_samples_PASS_prop(f"{out_dir}/VCF_QC_files/{gene}_validation_PASS_prop.txt")
    
#         if len(set(df_val.ROLLINGDB_ID) - set(validation_PASS_prop_df.ROLLINGDB_ID)) > 0:
#             raise ValueError(f"Incorrect sample numbers in the {gene} validation PASS prop file")
    
#         drop_val_samples = validation_PASS_prop_df.loc[validation_PASS_prop_df["ROLLINGDB_ID"].isin(df_val.ROLLINGDB_ID.values)].query("PASS_prop < 0.75")["ROLLINGDB_ID"].values
#         drop_val_samples = list(set(drop_val_samples).intersection(df_val["ROLLINGDB_ID"].values))
        
#         print(f"Removed {len(drop_val_samples)}/{len(df_val)} validation isolates with less than 75% PASS or Amb calls in {gene}")
            
#         df_val = df_val.query("ROLLINGDB_ID not in @drop_val_samples")
#         df_val["ROLLINGDB_ID"] = df_val["ROLLINGDB_ID"].astype(str)
        
#     df_val = df_val.merge(lineages[["ROLLINGDB_ID", "Coll2014", "Lineage"]], on="ROLLINGDB_ID", how="left").drop_duplicates()

#     if len(df_val.loc[pd.isnull(df_val["Lineage"])]) != 0:
#         raise ValueError(f"Fast-lineage-caller has not been run on all the samples")
    
#     prev_len = len(df_val)
#     # df_val = df_val.query("~Coll2014.str.contains(',')")
#     # print(f"Removed {prev_len - len(df_val)} validation isolates with multiple lineages")


#################################### STEP 4: REMOVE ISOLATES WITH THE SAME PRIMARY LINEAGE AND THE SAME BINARY RESISTANCE PHENOTYPE ###################################


def get_primary_lineage(lineage_str):

    # get the first number from numeric lineages. For alpha lineages (i.e. BOV, BOV_AFRI), remove the underscore
    split_lineage = np.unique([val[0] if val[0].isnumeric() else val.replace("_", "") for val in lineage_str.split(',')])

    # if there are multiple primary lineages, then return a sorted list (then joined into a string separated by commas). If there is only one, return the single one as a string
    if len(split_lineage) == 1:
        return split_lineage[0]
    else:
        return ','.join(np.sort(split_lineage))
        
# need to do this because 1) confounding and 2) when stratifying the groups by primary lineage and binary phenotype, there needs to be at least 1 in each group

df_train["Binary"] = (df_train[f"{drug}_upper_bound"] >= binary_thresh).astype(int)
df_train["Lineage"] = [get_primary_lineage(lineage) for lineage in df_train["Coll2014"]]
stratify_vals = df_train["Lineage"] + "-" + df_train["Binary"].astype(str)

# remove lineage-phenotype groups that don't have at least 2 isolates
# this is because the train-test splitting will fail due to the least populated class in y having only 1 member
# basically only a problem for PZA, when there may only be a single L1 isolate left after the previous cleaning steps
stratify_df = pd.Series(stratify_vals).value_counts().reset_index()
stratify_df.columns = ["stratify", "count"]
remove_groups = stratify_df.query("count < 2").stratify.values

# at the end, reset index so that index can be used for train/test splitting
if len(remove_groups) > 0:
    print(f"Removed {len(remove_groups)} isolates in the {remove_groups} lineages with fewer than 2 isolates")
    keep_idx = [idx for idx, group in enumerate(stratify_vals) if group not in remove_groups]
    df_train = df_train.reset_index(drop=True).iloc[keep_idx, :].reset_index(drop=True)
    stratify_vals = [val for val in stratify_vals if val not in remove_groups]
else:
    df_train = df_train.reset_index(drop=True)
        
    
#################################### STEP 5: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY BINARY PHENOTYPE AND PRIMARY LINEAGE ####################################


train_index, test_index = train_test_split(df_train.index.values, test_size=0.2, stratify=stratify_vals)

df_train.loc[train_index, "category"] = "original_train_set" 
df_train.loc[test_index, "category"] = "original_test_set"

# print the means of the two groups as a check
print(df_train.groupby("category")[["Binary", f"{drug}_midpoint"]].mean())
df_train.to_csv(os.path.join(out_dir, "data_for_model.csv"), index=False)

print(f"Final: {df_train.shape[0]} samples in the training data")

if val_present:
    print(f"Final: {df_val.shape[0]} samples in the validation data\n")


#################################### STEP : WRITE TXT FILE WITH THE PATHS OF THE VCF FILES WITH BOTH THE TRAINING AND VALIDATION DATASETS ####################################


# create a new txt file of paths, adding the validation file paths to the original file
with open(os.path.join(out_dir, "combined_paths_for_aln.txt"), "w+") as file:
    
    for sample_id in df_train["ROLLINGDB_ID"].values:
        
        # this file contains all non-REF calls for each sample. It is also annotated with snpEff, hence the file extension
        fName = os.path.join(vcf_dir, f"{sample_id}/pilon/{sample_id}.eff.vcf")
        if not os.path.isfile(fName):
            raise ValueError(f"{fName} not found!")
        file.write(fName + "\n")

    # if val_present:
        
    #     # get the validation data files
    #     for sample_id in df_val["ROLLINGDB_ID"].values:
    
    #         # this file contains all non-REF calls for each sample.
    #         fName = os.path.join(vcf_dir, f"{sample_id}/pilon/{sample_id}.eff.vcf")
    #         if not os.path.isfile(fName):
    #             raise ValueError(f"{fName} not found!")
    #         file.write(fName + "\n")
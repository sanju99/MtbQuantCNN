import pandas as pd
import numpy as np
import sys, vcf, glob, os
from sklearn.model_selection import train_test_split


_, drug, START, END = sys.argv

# coordinates of the region of interest -- need them to check if there are large proportions of low coverage sites
# START and END should be inclusive and 1-indexed (natural numbers)
START = int(START)
END = int(END)

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
vcf_dir = "/n/scratch3/users/s/sak0914/annotated_VCF"

# validation data for a single drug
df_train = pd.read_csv(os.path.join(out_dir, "data_intermediate_clean.csv"))
df_val = pd.read_csv(os.path.join(out_dir, "validation_data_for_model.csv"))
print(f"Original: {df_train.shape[0]} samples in the training data")
print(f"Original: {df_val.shape[0]} samples in the validation data\n")

unclassified_thresh = 25
high_unclassified_prop = []
finished_samples = 0

lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv")


############################### EXCLUDE SAMPLES WITH LARGE PROPORTIONS OF SITES THAT DO NOT MAP TO MTBC ###############################


# # for each sample, get the distribution of classified (MTBC) vs. unclassified (not MTBC) reads
# for sample_id in df_val["ROLLINGDB_ID"].values:
    
#     if os.path.isfile(os.path.join(vcf_dir, f"sample_id.eff.vcf")):

#         # kraken_class = pd.read_csv(os.path.join(validation_vcf_dir, sample_id, "kraken/kraken_classifications"), sep="\t", header=None)
#         kraken_report = pd.read_csv(os.path.join(validation_vcf_dir, sample_id, "kraken/kraken_report"), sep="\t", header=None)
        
#         # this is out of 100
#         if "unclassified" in kraken_report[5].values:
#             unclassified_percent = kraken_report.loc[kraken_report[5]=="unclassified"][0].values[0]
#         else:
#             unclassified_percent = 0
            
#         if unclassified_percent > unclassified_thresh:
#             high_unclassified_prop.append([sample_id, unclassified_percent])
            
#         finished_samples += 1
        
#     else:
#         raise ValueError(f"There is no VCF file for {sample_id}")
        
# assert finished_samples == len(df_val)
# print(f"Removed {len(high_unclassified_prop)}/{len(df_val)} validation samples with more than {unclassified_thresh}% unclassified reads")

# high_unclassified_samples, _ = list(zip(*high_unclassified_prop))

# # remove samples with high proportions of reads that don't align to MTBC
# df_val = df_val.query("ROLLINGDB_ID not in @high_unclassified_samples\n")


################## STEP 3: REMOVE ISOLATES WITH A LARGE PROPORTION OF NON-PASS, NON-AMB VARIANTS IN THE REGION OF INTEREST ##################
        

# THIS NEEDS TO HAVE BEEN RUN PREVIOUSLY to create text files in the same output folder
if not os.path.isfile(f"{out_dir}/training_PASS_prop.txt"):
    print(f"Please create {out_dir}/training_PASS_prop.txt before running this script! \n   Command:bash data_processing/QC_scripts/check_pass_proportion.sh {os.path.join(out_dir, 'data_for_model.csv')} {out_dir}/training_PASS_prop.txt {vcf_dir} {START} {END}")
    raise ValueError()
    
    
if not os.path.isfile(f"{out_dir}/validation_PASS_prop.txt"):
    print(f"Please create {out_dir}/validation_PASS_prop.txt before running this script! \n   Command:bash data_processing/QC_scripts/check_pass_proportion.sh {os.path.join(out_dir, 'validation_data.csv')} {out_dir}/validation_PASS_prop.txt {vcf_dir} {START} {END}")
    raise ValueError()

    
def get_samples_PASS_prop(fName):
    
    with open(fName, "r+") as file:
        lines = file.readlines()
        
    PASS_prop = pd.DataFrame(columns=["ROLLINGDB_ID", "PASS_prop"])

    for i, val in enumerate(lines):

        sample_id, prop = val.strip("\n").split(" ")

        # no variants in the region of interest
        if prop == "":
            prop = np.nan

        PASS_prop.loc[i, :] = [sample_id, float(prop)]
        
    # returns a dataframe, where the second column is the proportion of variants that are PASS or Amb
    return PASS_prop


# get the dataframe of proportions of PASS/Amb calls in the alignment region
training_PASS_prop_df = get_samples_PASS_prop(f"{out_dir}/training_PASS_prop.txt")
validation_PASS_prop_df = get_samples_PASS_prop(f"{out_dir}/validation_PASS_prop.txt")

drop_train_samples = training_PASS_prop_df.query("PASS_prop < 0.75")["ROLLINGDB_ID"].values
drop_val_samples = validation_PASS_prop_df.loc[validation_PASS_prop_df["ROLLINGDB_ID"].isin(df_val.ROLLINGDB_ID.values)].query("PASS_prop < 0.75")["ROLLINGDB_ID"].values

print(f"Removed {len(drop_train_samples)}/{len(training_PASS_prop_df)} training isolates with less than 75% PASS or Amb calls in the alignment region")
print(drop_train_samples)
df_train = df_train.query("ROLLINGDB_ID not in @drop_train_samples")

print(f"Removed {len(drop_val_samples)}/{len(validation_PASS_prop_df)} validation isolates with less than 75% PASS or Amb calls in the alignment region\n")
print(drop_val_samples)
df_val = df_val.query("ROLLINGDB_ID not in @drop_val_samples")
df_val = df_val.merge(lineages[["ROLLINGDB_ID", "Coll2014", "Lineage"]], on="ROLLINGDB_ID")

prev_len = len(df_val)
df_val = df_val.query("~Coll2014.str.contains(',')")
print(f"Print dropped {prev_len - len(df_val)} validation isolates with multiple lineages")
df_val.to_csv(os.path.join(out_dir, "validation_data_for_model.csv"), index=False)


#################################### STEP 4: REMOVE ISOLATES WITH THE SAME PRIMARY LINEAGE AND THE SAME BINARY RESISTANCE PHENOTYPE ###################################


# need to do this because 1) confounding and 2) when stratifying the groups by primary lineage and binary phenotype, there needs to be at least 1 in each group

df_train["Lineage"] = [val[0] if "." in val else val.replace("_", "") for val in df_train["Coll2014"]]
stratify_vals = df_train["Lineage"] + "-" + df_train["Binary"].astype(str)

summary_counts = pd.DataFrame(pd.Series(stratify_vals).value_counts()).rename(columns={0:"Count"}).reset_index()
summary_counts[["Lineage", "Resistance"]] = summary_counts["index"].str.split("-", expand=True)

for lineage in summary_counts["Lineage"].unique():
    if len(summary_counts.query("Lineage == @lineage").Resistance.unique()) < 2:
        print(f"Removed {len(df_train.query('Lineage == @lineage'))} isolates in lineage {lineage}")
        df_train = df_train.query("Lineage not in @lineage")
        
    
#################################### STEP 5: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY BINARY PHENOTYPE AND PRIMARY LINEAGE ####################################


# get new stratify vals after removing some lineages
df_train["Lineage"] = [val[0] if "." in val else val.replace("_", "") for val in df_train["Lineage"]]
stratify_vals = df_train["Lineage"] + "-" + df_train["Binary"].astype(str)

# reset index so that index can be used for train/test splitting
df_train = df_train.reset_index(drop=True)
train_index, test_index = train_test_split(df_train.index, test_size=0.2, stratify=stratify_vals)

df_train.loc[train_index, "category"] = "original_train_set" 
df_train.loc[test_index, "category"] = "original_test_set"

# print the means of the two groups as a check
print(df_train.groupby("category")[["Binary", f"{drug}_midpoint"]].mean())
df_train.to_csv(os.path.join(out_dir, "data_for_model.csv"), index=False)

print(f"Final: {df_train.shape[0]} samples in the training data")
print(f"Final {df_val.shape[0]} samples in the validation data\n")


#################################### STEP : WRITE TXT FILE WITH THE PATHS OF THE VCF FILES WITH BOTH THE TRAINING AND VALIDATION DATASETS ####################################


# create a new txt file of paths, adding the validation file paths to the original file
with open(os.path.join(out_dir, "combined_paths_for_aln.txt"), "w+") as file:
    
    for sample_id in df_train["ROLLINGDB_ID"].values:
        
        # this file contains all non-REF calls for each sample. It is also annotated with snpEff, hence the file extension
        fName = os.path.join(vcf_dir, f"{sample_id}.eff.vcf")
        assert os.path.isfile(fName)
        file.write(fName + "\n")
    
    # get the validation data files
    for sample_id in df_val["ROLLINGDB_ID"].values:

        # this file contains all non-REF calls for each sample.
        fName = os.path.join(vcf_dir, f"{sample_id}.eff.vcf")
        assert os.path.isfile(fName)
        file.write(fName + "\n")
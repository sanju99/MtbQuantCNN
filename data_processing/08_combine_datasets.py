import pandas as pd
import numpy as np
import sys, vcf, glob, os, yaml
from sklearn.model_selection import train_test_split


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
locus_list = kwargs["locus_list"]

# dataframe of start and end coordinates and sense of various genes. START and END are 1-indexed and inclusive (natural numbers)
drug_gene_mapping = pd.read_csv("drug_gene_mapping.csv")

# first check that all loci are there
for locus in locus_list:
    if locus not in drug_gene_mapping["Locus Name"].values:
        raise ValueError(f"{locus} not found in drug_gene_mapping.csv")

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
training_vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"
validation_vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/VCF"

if not os.path.isdir(f"{out_dir}/VCF_QC_files"):
    os.makedirs(f"{out_dir}/VCF_QC_files")

# validation data for a single drug
df_train = pd.read_csv(os.path.join(out_dir, "data_intermediate_clean.csv"))

# some drugs: i.e. PZA, there are no MICs from the MIC-ML dataset
if os.path.isfile(os.path.join(out_dir, "validation_data.csv")):
    df_val = pd.read_csv(os.path.join(out_dir, "validation_data.csv"))
    val_present = True
else:
    df_val = pd.DataFrame(columns=[])
    val_present = False

val_present = False

print(f"Original: {df_train.shape[0]} samples in the training data")
print(f"Original: {df_val.shape[0]} samples in the validation data\n")

unclassified_thresh = 25
high_unclassified_prop = []
finished_samples = 0

lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv")


############################### EXCLUDE SAMPLES WITH LARGE PROPORTIONS OF SITES THAT DO NOT MAP TO MTBC ###############################


# for each sample, get the distribution of classified (MTBC) vs. unclassified (not MTBC) reads
if val_present:
    for sample_id in df_val["ROLLINGDB_ID"].values:
        
        if os.path.isfile(os.path.join(validation_vcf_dir, f"{sample_id}/pilon/{sample_id}.vcf")):
    
            # kraken_class = pd.read_csv(os.path.join(validation_vcf_dir, sample_id, "kraken/kraken_classifications"), sep="\t", header=None)
            kraken_report = pd.read_csv(os.path.join(validation_vcf_dir, sample_id, "kraken/kraken_report"), sep="\t", header=None)
            
            # this is out of 100
            if "unclassified" in kraken_report[5].values:
                unclassified_percent = kraken_report.loc[kraken_report[5]=="unclassified"][0].values[0]
            else:
                unclassified_percent = 0
                
            if unclassified_percent > unclassified_thresh:
                high_unclassified_prop.append([sample_id, unclassified_percent])
                
            finished_samples += 1
            
        else:
            raise ValueError(f"There is no VCF file for {sample_id}")
            
    assert finished_samples == len(df_val)
    print(f"Removed {len(high_unclassified_prop)}/{len(df_val)} validation samples with more than {unclassified_thresh}% unclassified reads")
    
    high_unclassified_samples, _ = list(zip(*high_unclassified_prop))
    
    # remove samples with high proportions of reads that don't align to MTBC
    df_val = df_val.query("ROLLINGDB_ID not in @high_unclassified_samples\n")


################## STEP 3: REMOVE ISOLATES WITH A LARGE PROPORTION OF NON-PASS, NON-AMB VARIANTS IN THE REGION OF INTEREST ##################
        

found_loci = 0

for locus in locus_list:

    START, END = drug_gene_mapping.loc[drug_gene_mapping["Locus Name"]==locus][["Start", "End"]].values[0]
    
    if os.path.isfile(f"{out_dir}/VCF_QC_files/{locus}_training_PASS_prop.txt"):
        print(f"Found {out_dir}/VCF_QC_files/{locus}_training_PASS_prop.txt")
        found_loci += 1
    else:
        print(f"Please create {out_dir}/VCF_QC_files/{locus}_training_PASS_prop.txt before running this script! Command:\nbash data_processing/QC_scripts/check_pass_proportion.sh {os.path.join(out_dir, 'data_intermediate_clean.csv')} {out_dir}/VCF_QC_files/{locus}_training_PASS_prop.txt {training_vcf_dir} {START} {END}\n")

# if all are not found, then don't keep running the script because it will cause errors
if found_loci < len(locus_list):
    print(f"Only found {found_loci}/{len(locus_list)} training_PASS_prop.txt files")
    exit()
    

if val_present:

    found_loci = 0

    for locus in locus_list:

        START, END = drug_gene_mapping.loc[drug_gene_mapping["Locus Name"]==locus][["Start", "End"]].values[0]
        
        if os.path.isfile(f"{out_dir}/VCF_QC_files/{locus}_validation_PASS_prop.txt"):
            found_loci += 1
        else:
            print(f"Please create {out_dir}/VCF_QC_files/{locus}_validation_PASS_prop.txt before running this script! Command:\nbash data_processing/QC_scripts/check_pass_proportion.sh {os.path.join(out_dir, 'validation_data.csv')} {out_dir}/VCF_QC_files/{locus}_validation_PASS_prop.txt {validation_vcf_dir} {START} {END}\n")
    
    # if all are not found, then don't keep running the script because it will cause errors
    if found_loci < len(locus_list):
        print(f"Only found {found_loci}/{len(locus_list)} validation_PASS_prop.txt files")
        exit()

    
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
training_PASS_prop_df = get_samples_PASS_prop(f"{out_dir}/VCF_QC_files/{locus}_training_PASS_prop.txt")
drop_train_samples = list(set(df_train["ROLLINGDB_ID"].values).intersection(training_PASS_prop_df.query("PASS_prop < 0.75")["ROLLINGDB_ID"].values))

print(f"Removed {len(drop_train_samples)}/{len(df_train)} training isolates with less than 75% PASS or Amb calls in the alignment region")
print(drop_train_samples)
df_train = df_train.query("ROLLINGDB_ID not in @drop_train_samples")

if val_present:
    validation_PASS_prop_df = get_samples_PASS_prop(f"{out_dir}/VCF_QC_files/{locus}_validation_PASS_prop.txt")

    drop_val_samples = validation_PASS_prop_df.loc[validation_PASS_prop_df["ROLLINGDB_ID"].isin(df_val.ROLLINGDB_ID.values)].query("PASS_prop < 0.75")["ROLLINGDB_ID"].values
    drop_val_samples = list(set(drop_val_samples).intersection(df_val["ROLLINGDB_ID"].values))
    
    print(f"Removed {len(drop_val_samples)}/{len(df_val)} validation isolates with less than 75% PASS or Amb calls in the alignment region\n")
    print(drop_val_samples)
    df_val = df_val.query("ROLLINGDB_ID not in @drop_val_samples")
    
    df_val = df_val.merge(lineages[["ROLLINGDB_ID", "Coll2014", "Lineage"]], on="ROLLINGDB_ID", how="left")
    # assert len(df_val.loc[pd.isnull(df_val["Lineage"])]) == 0
    print(df_val.loc[pd.isnull(df_val["Lineage"])])

    if len(df_val.loc[pd.isnull(df_val["Lineage"])]) != 0:
        raise ValueError()
    
    prev_len = len(df_val)
    df_val = df_val.query("~Coll2014.str.contains(',')")
    print(f"Dropped {prev_len - len(df_val)} validation isolates with multiple lineages")
    df_val.to_csv(os.path.join(out_dir, "validation_data_for_model.csv"), index=False)


#################################### STEP 4: REMOVE ISOLATES WITH THE SAME PRIMARY LINEAGE AND THE SAME BINARY RESISTANCE PHENOTYPE ###################################


# need to do this because 1) confounding and 2) when stratifying the groups by primary lineage and binary phenotype, there needs to be at least 1 in each group

df_train["Lineage"] = [val[0] if "." in val else val.replace("_", "") for val in df_train["Coll2014"]]

if drug != "PZA":
    stratify_vals = df_train["Lineage"] + "-" + df_train["Binary"].astype(str)
    
    summary_counts = pd.DataFrame(pd.Series(stratify_vals).value_counts()).rename(columns={0:"Count"}).reset_index()
    summary_counts[["Lineage", "Resistance"]] = summary_counts["index"].str.split("-", expand=True)
    
    for lineage in summary_counts["Lineage"].unique():
        if len(summary_counts.query("Lineage == @lineage").Resistance.unique()) < 2:
            print(f"Removed {len(df_train.query('Lineage == @lineage'))} isolates in lineage {lineage} from the train/test set")
            df_train = df_train.query("Lineage not in @lineage")
        
    
#################################### STEP 5: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY BINARY PHENOTYPE AND PRIMARY LINEAGE ####################################


# first ensure that there are at least 2 isolates from each primary lineage and binary phenotype

# get new stratify vals after removing some lineages
df_train["Lineage"] = [val[0] if "." in val else val.replace("_", "") for val in df_train["Lineage"]]
stratify_vals = df_train["Lineage"] + "-" + df_train["Binary"].astype(str)

stratify_df = pd.Series(stratify_vals).value_counts().reset_index()
stratify_df.columns = ["stratify", "count"]
remove_groups = stratify_df.query("count < 2").stratify.values
print(f"Removed {len(remove_groups)} isolates in the {remove_groups} lineages")

keep_idx = [idx for idx, group in enumerate(stratify_vals) if group not in remove_groups]
stratify_vals = [val for val in stratify_vals if val not in remove_groups]

# reset index so that index can be used for train/test splitting
df_train = df_train.reset_index(drop=True).iloc[keep_idx, :].reset_index(drop=True)
train_index, test_index = train_test_split(df_train.index, test_size=0.2, stratify=stratify_vals)

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
        fName = os.path.join(training_vcf_dir, f"{sample_id}/pilon/{sample_id}.vcf")
        assert os.path.isfile(fName)
        file.write(fName + "\n")

    if val_present:
        
        # get the validation data files
        for sample_id in df_val["ROLLINGDB_ID"].values:
    
            # this file contains all non-REF calls for each sample.
            fName = os.path.join(validation_vcf_dir, f"{sample_id}/pilon/{sample_id}.vcf")
            assert os.path.isfile(fName)
            file.write(fName + "\n")
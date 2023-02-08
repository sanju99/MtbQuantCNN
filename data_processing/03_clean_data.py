import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml, vcf
from sklearn.model_selection import train_test_split
import Bio.SeqUtils
import warnings
warnings.filterwarnings("ignore")


# example: python3 -u data_processing/03_clean_data.py MXF 0.5 /n/scratch3/users/s/sak0914/vcf_for_annot
# example: python3 -u data_processing/03_clean_data.py RIF 0.5 /n/scratch3/users/s/sak0914/vcf_for_annot
_, drug, cc, vcf_dir = sys.argv
cc = float(cc)
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")


###### STEP 1: REMOVE ISOLATES WITH MULTIPLE RECORDED LINEAGES -- LOTS OF AMBIGUOUS CALLS DUE TO POLYCLONAL INFECTIONS OR SEQUENCING ERROR ######


# first 2 columns are the Isolate name and the Freschi lineage
lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv", sep="\t", header=None, usecols=[0, 1])
lineages.columns = ["ROLLINGDB_ID", "Lineage"]

# the Freschi lineages have "lineage" appended to the front, so remove that
lineages["Lineage"] = [val.replace("lineage", "") for val in lineages["Lineage"]]

df_phenos = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/data_with_paths.csv")

df_combined = df_phenos.merge(lineages, on="ROLLINGDB_ID", how="left")
assert len(df_combined) == len(df_phenos)

# create binary phenotype column
df_combined["Binary"] = (df_combined[f"{drug}_midpoint"] > cc).astype(int)

df_combined = df_combined.loc[~df_combined['Lineage'].str.contains(',')]
print(f"Removed {len(df_phenos)-len(df_combined)} isolates with multiple lineages")
del df_phenos


###### STEP 2: REMOVE ISOLATES WITH CATEGORY 1 MUTATIONS AND MIC < 1/2 THE CC ######


# Get all category 1 variants (don't use category 2 to avoid dropping too many isolates and to be more stringent about what we drop)
who_high_conf = who_variants.loc[(who_variants["drug"] == drug) & (who_variants.confidence.str.contains("|".join(["1"])))].reset_index(drop=True)

for _, row in who_high_conf.iterrows():
    if "," in row["genome_index"]:
        expanded_pos = row["genome_index"].split(",")
        
        for pos in expanded_pos:
            add_df = pd.DataFrame({"drug": drug, "genome_index": pos, "confidence": row["confidence"], "gene": row["gene"], "variant": row["variant"]}, index=[len(who_high_conf)])
            who_high_conf = pd.concat([who_high_conf, add_df])
          
who_high_conf = who_high_conf.loc[~who_high_conf.genome_index.str.contains(",")]
who_high_conf = who_high_conf.drop_duplicates().reset_index(drop=True)
who_high_conf["genome_index"] = who_high_conf["genome_index"].astype(int)

# aa_code_dict = {'Val':'V', 'Ile':'I', 'Leu':'L', 'Glu':'E', 'Gln':'Q', \
# 'Asp':'D', 'Asn':'N', 'His':'H', 'Trp':'W', 'Phe':'F', 'Tyr':'Y',    \
# 'Arg':'R', 'Lys':'K', 'Ser':'S', 'Thr':'T', 'Met':'M', 'Ala':'A',    \
# 'Gly':'G', 'Pro':'P', 'Cys':'C'}

# code_aa_dict = {val: key for key, val in aa_code_dict.items()}

# convert them to 3-letter amino acid codes, which is what the ANN field has
for i, row in who_high_conf.iterrows():
    
    if len(row["variant"].split("_")) == 2:
        var = row["variant"].split("_")[1]
        expand_code = Bio.SeqUtils.IUPACData.protein_letters_1to3[var[0]] + var[1:-1] + Bio.SeqUtils.IUPACData.protein_letters_1to3[var[-1]]
        who_high_conf.loc[i, "ANN"] = expand_code
    else:
        who_high_conf.loc[i, "ANN"] = row["variant"]
        
  
# read in list of VCF files
vcf_files_list = glob.glob(f"{vcf_dir}/*.eff.vcf")
vcf_files_list = [val for val in vcf_files_list if os.path.basename(val).split(".")[0] in df_combined.ROLLINGDB_ID.values]

assert len(vcf_files_list) == len(df_combined)
highConf_isolates = []

for i, fName in enumerate(vcf_files_list):
    
    vcf_file = vcf.Reader(filename=fName)
    
    for record in vcf_file:

        # if FILTER == PASS, the FILTER field is an empty list, so the length is 0
        if record.POS in who_high_conf.genome_index.values and len(record.FILTER) == 0:

            variant_to_check = who_high_conf.loc[who_high_conf["genome_index"]==record.POS, "ANN"].values[0]

            if variant_to_check in ",".join(record.INFO['ANN']):
                highConf_isolates.append(os.path.basename(fName).split(".")[0])
                break
            
    if i % 100 == 0:
        print(i)
            
            
df_combined = df_combined.loc[~((df_combined["ROLLINGDB_ID"].isin(highConf_isolates)) & (df_combined[f"{drug}_midpoint"] < cc/2))]
print(f"Removed {len(vcf_files_list) - len(df_combined)} isolates with category 1 mutations and MICs < 1/2 CC")


# ###### STEP 3: REMOVE LINEAGE WITH ONLY ONE MEMBER -- CAN CAUSE CONFOUNDING ######


# drop_single_lineages = pd.DataFrame(df_combined.Lineage.value_counts()).query("Lineage==1").index.values
# for lineage in drop_single_lineages:
#     print(f"Removed lineage {lineage}")

# df_combined = df_combined.query("Lineage not in @drop_single_lineages")


###### STEP 4: REMOVE ISOLATES WITH THE SAME PRIMARY LINEAGE AND THE SAME BINARY RESISTANCE PHENOTYPE (I.E. ALL MEMBERS OF A LINEAGE ARE RESISTANT) ######


df_combined["Primary_Lineage"] = [val[0] if "." in val else val.replace("_", "") for val in df_combined["Lineage"]]
stratify_vals = df_combined["Primary_Lineage"] + "-" + df_combined["Binary"].astype(str)

summary_counts = pd.DataFrame(pd.Series(stratify_vals).value_counts()).rename(columns={0:"Count"}).reset_index()
summary_counts[["Lineage", "Resistance"]] = summary_counts["index"].str.split("-", expand=True)

for lineage in summary_counts["Lineage"].unique():
    if len(summary_counts.query("Lineage == @lineage").Resistance.unique()) < 2:
        print(f"Removed lineage {lineage}")
        df_combined = df_combined.query("Primary_Lineage not in @lineage")
        
        
###### STEP 5: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY BINARY PHENOTYPE AND PRIMARY LINEAGE ######

        
# get new stratify vals after removing some lineages
df_combined["Primary_Lineage"] = [val[0] if "." in val else val.replace("_", "") for val in df_combined["Lineage"]]
stratify_vals = df_combined["Primary_Lineage"] + "-" + df_combined["Binary"].astype(str)

# reset index so that index can be used for train/test splitting
df_combined = df_combined.reset_index(drop=True)
train_index, test_index = train_test_split(df_combined.index, test_size=0.2, stratify=stratify_vals)

df_combined.loc[train_index, "category"] = "original_train_set" 
df_combined.loc[test_index, "category"] = "original_test_set"

# print the means of the two groups as a cursory check
print(df_combined.groupby("category")[["Binary", f"{drug}_midpoint"]].mean())

df_combined.to_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/data_for_model.csv", index=False)
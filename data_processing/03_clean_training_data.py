import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml, vcf
import Bio.SeqUtils
import warnings
warnings.filterwarnings("ignore")


# example: python3 -u data_processing/03_clean_training_data.py MXF 0.5 /n/scratch3/users/s/sak0914/annotated_VCF
# example: python3 -u data_processing/03_clean_training_data.py RIF 0.5 /n/scratch3/users/s/sak0914/annotated_VCF
_, drug, cc, vcf_dir = sys.argv
cc = float(cc)
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
model_df = f"{out_dir}/data_intermediate_clean.csv"


################## STEP 1: REMOVE ISOLATES WITH MULTIPLE RECORDED LINEAGES -- LOTS OF AMBIGUOUS CALLS DUE TO POLYCLONAL INFECTIONS OR SEQUENCING ERROR ##################


# lineages file should already be cleaned
lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv")
df_phenos = pd.read_csv(f"{out_dir}/data_with_paths.csv")#.query("DB_OF_ORIGIN=='CRyPTIC'")

# use Coll 2014 scheme
df_combined = df_phenos.merge(lineages[["ROLLINGDB_ID", "Coll2014"]], on="ROLLINGDB_ID", how="left")
assert len(df_combined) == len(df_phenos)

# create binary phenotype column
df_combined["Binary"] = (df_combined[f"{drug}_midpoint"] > cc).astype(int)
print(df_combined.shape)

# remove isolates that have mixed lineages
df_combined = df_combined.loc[~df_combined['Coll2014'].str.contains(',')]
print(f"Removed {len(df_phenos)-len(df_combined)} isolates with multiple lineages")
del df_phenos


#################################### STEP 2: REMOVE ISOLATES WITH CATEGORY 1 MUTATIONS AND MIC < 1/2 THE CC ####################################


# Get all category 1 variants (don't use category 2 because it's the interim category
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

# convert them to 3-letter amino acid codes, which is what the ANN field has
for i, row in who_high_conf.iterrows():
    
    if len(row["variant"].split("_")) == 2:
        var = row["variant"].split("_")[1]
        expand_code = Bio.SeqUtils.IUPACData.protein_letters_1to3[var[0]] + var[1:-1] + Bio.SeqUtils.IUPACData.protein_letters_1to3[var[-1]]
        who_high_conf.loc[i, "ANN"] = expand_code
    else:
        who_high_conf.loc[i, "ANN"] = row["variant"]
        
print(who_high_conf)
  
# # read in list of VCF files
# vcf_files_list = glob.glob(f"{vcf_dir}/*.eff.vcf")
# vcf_files_list = [val for val in vcf_files_list if os.path.basename(val).split(".")[0] in df_combined.ROLLINGDB_ID.values]

# assert len(vcf_files_list) == len(df_combined)
# highConf_isolates = []

# for i, fName in enumerate(vcf_files_list):
    
#     vcf_file = vcf.Reader(filename=fName)
    
#     for record in vcf_file:

#         # if FILTER == PASS, the FILTER field is an empty list, so the length is 0
#         # only exclude variants that have FILTER = PASS. Amb variants will be in the model as missing and can discuss them as mispredictions
#         if record.POS in who_high_conf.genome_index.values and len(record.FILTER) == 0:

#             variants_to_check = who_high_conf.loc[who_high_conf["genome_index"]==record.POS, "ANN"].values

#             for variant in variants_to_check:
            
#                 if variant in ",".join(record.INFO['ANN']):
#                     # print(os.path.basename(fName).split(".")[0], variant)
#                     highConf_isolates.append(os.path.basename(fName).split(".")[0])
#                     break
            
#     if i % 1000 == 0:
#         print(i)
            
            
# df_combined = df_combined.loc[~((df_combined["ROLLINGDB_ID"].isin(highConf_isolates)) & (df_combined[f"{drug}_midpoint"] < cc/2))]
# print(f"Removed {len(vcf_files_list) - len(df_combined)} isolates with category 1 mutations and MICs < 1/2 CC")

# prev_len = len(df_combined)
# df_combined = df_combined.query(f"~({drug}_lower_bound < @cc & {drug}_upper_bound > @cc)")
# print(f"Removed {prev_len - len(df_combined)} isolates with MIC bounds that span the CC of {cc}")

# # save at intermediate steps in case (because the previous step takes almost 1 hour)
# df_combined.to_csv(model_df, index=False)
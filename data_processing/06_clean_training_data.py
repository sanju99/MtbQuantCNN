import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
import Bio.SeqUtils
import warnings
warnings.filterwarnings("ignore")


cc_df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/criticalConcentrations_updated.csv")
# cc_df = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/critical_concentrations_all.csv")

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETH",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMI",
                  "Pretomanid": "PTM",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LEV",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

abbr_drug_dict = {val: key for (key, val) in drug_abbr_dict.items()}

def get_critical_concentration(drug):

    drug_full_name = abbr_drug_dict[drug].upper()

    # get the row associcated with the particular drug
    for val in cc_df.query("antb == @drug_full_name").values[0]:

        # skip the columns of the drug or the abbreviation
        if val != drug_full_name and val != drug:
            
            # get the first non-null critical concentration
            if not pd.isnull(val):
                cc = val
                break

    return float(cc)


_, drug = sys.argv
cc = get_critical_concentration(drug)
vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"

# who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")
who_variants["gene"] = [val.split("_")[0] for val in who_variants.mutation.values]
who_variants["variant"] = ["_".join(val.split("_")[1:]) for val in who_variants.mutation.values]

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"

if os.path.isfile(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/isolate_variants_fixed_annot.csv"):
    highConf_variants_present = True
    isolate_variants = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/isolate_variants_fixed_annot.csv")
else:
    highConf_variants_present = False
    

################## STEP 0: UPDATE THE LINEAGE FILE WITH ADDITIONAL ISOLATES ADDED IN THE PREVIOUS SCRIPT ##################

# this script would only be run if another drug is being added, so 

def clean_fast_lineage_caller_output(in_fName):
    
    lineages = pd.read_csv(in_fName, sep="\t", header=None)
    print(f"Found lineages for {len(lineages)} isolates")
    
    lineages.columns = ["ROLLINGDB_ID", 'Coll2014', 'Freschi2020', 'Lipworth2019', 'Shitikov2017', 'Stucki2016']
    
    # remove file extensions if there are any
    lineages["ROLLINGDB_ID"] = lineages["ROLLINGDB_ID"].str.split(".", expand=True)[0]
    
    # remove the "lineage" prefix in the Coll2014 scheme 
    lineages["Coll2014"] = lineages["Coll2014"].str.replace("lineage", "")
    
    # add a column for the primary lineage based on the Coll 2014 scheme
    lineages["Lineage"] = [val.split(".")[0] if val[0].isnumeric() else val for val in lineages["Coll2014"].values]
    
    out_fName = os.path.join(os.path.dirname(in_fName), os.path.basename(in_fName).replace(".tsv", ".csv"))
    print(f"Saving cleaned lineage file to {out_fName}")
    lineages.to_csv(out_fName, index=False)
    

# lineages file should already be cleaned
lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv")
df_phenos = pd.read_csv(f"{out_dir}/data_with_paths.csv")

# use Coll 2014 scheme
df_combined = df_phenos.merge(lineages[["ROLLINGDB_ID", "Coll2014"]], on="ROLLINGDB_ID", how="left")
assert len(df_combined) == len(df_phenos)

prev_len = len(df_combined)
del df_phenos

# df_combined = df_combined.query("~Coll2014.str.contains(',')").reset_index(drop=True)
# print(f"Removed {prev_len - len(df_combined)} isolates with multiple Coll et al., 2014 lineages called")
# prev_len = len(df_combined)


#################################### STEP 1: REMOVE ISOLATES WITH CATEGORY 1 MUTATIONS AND MIC < 1/2 THE CC ####################################


if highConf_variants_present:
    # Get all category 1 variants (don't use category 2 because it's the interim category
    who_high_conf = who_variants.loc[(who_variants["drug"] == drug) & (who_variants.confidence.str.contains("|".join(["1"])))].reset_index(drop=True)
    
    for _, row in who_high_conf.iterrows():
        if "," in row["genome_index"]:
            expanded_pos = row["genome_index"].split(",")
            
            for pos in expanded_pos:
                add_df = pd.DataFrame({"drug": drug, "genome_index": pos, "confidence": row["confidence"], "gene": row["gene"], "variant": row["variant"]}, index=[len(who_high_conf)])
                who_high_conf = pd.concat([who_high_conf, add_df])
            
    highConf_isolates = isolate_variants.query("mutation in @who_high_conf.mutation.values").Isolate.unique()
    df_combined = df_combined.loc[~((df_combined["ROLLINGDB_ID"].isin(highConf_isolates)) & (df_combined[f"{drug}_upper_bound"] < cc / 2))]
    print(f"Removed {prev_len - len(df_combined)} isolates with any of {len(who_high_conf)} category 1 mutations and MIC upper bound < {cc / 2}")

# kind of a weird case, but in some studies, they basically just measured R vs. S, but they record the MIC as <= CC or > CC. 
# The MIC of CC / 2 is rather uninformative and probably introduces noise because it's a very coarse value, so remove those cases
prev_len = len(df_combined)
df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == 0) & (df_combined[f"{drug}_upper_bound"] == cc))]
df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == cc) & (df_combined[drug] == f">{int(cc)}"))]
print(f"Removed {prev_len - len(df_combined)} isolates with MICs that are known only relative to the critical concentration of {cc}")

# because the CC for RIF was updated, also include the CC of 1
if drug == "RIF":
    prev_len = len(df_combined)
    df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == 0) & (df_combined[f"{drug}_upper_bound"] == 1))]
    df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == 1) & (df_combined[drug] == ">1"))]
    print(f"Removed {prev_len - len(df_combined)} isolates with MICs that are known only relative to the old critical concentration of 1")

# remove the following from datasets because otherwise not sure what to do about sensitivity/specificity. i.e. is such a sample R or S?
prev_len = len(df_combined)
df_combined = df_combined.query(f"~({drug}_lower_bound < @cc & {drug}_upper_bound > @cc)")
print(f"Removed {prev_len - len(df_combined)} isolates with MIC bounds that span the CC of {cc}")

def get_primary_lineage(lineage_str):

    # get the first number from numeric lineages. For alpha lineages (i.e. BOV, BOV_AFRI), remove the underscore
    split_lineage = np.unique([val[0] if val[0].isnumeric() else val.replace("_", "") for val in lineage_str.split(',')])

    # if there are multiple primary lineages, then return a sorted list (then joined into a string separated by commas). If there is only one, return the single one as a string
    if len(split_lineage) == 1:
        return split_lineage[0]
    else:
        return ','.join(np.sort(split_lineage))
        
# need to do this because 1) confounding and 2) when stratifying the groups by primary lineage and binary phenotype, there needs to be at least 1 in each group

# BECAUSE WE ARE USING THE UPPER BOUND, NOT THE MIDPOINT, SHOULD BE EXCLUSIVE FOR DETRMINING BINARY RESISTANCE
# this is because i.e. PZA = (50, 100) means susceptible, even though PZA_upper_bound = 100
df_combined["Binary"] = (df_combined[f"{drug}_upper_bound"] > cc).astype(int)
df_combined["Lineage"] = [get_primary_lineage(lineage) for lineage in df_combined["Coll2014"]]
stratify_vals = df_combined["Lineage"] + "-" + df_combined["Binary"].astype(str)

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
    df_combined = df_combined.reset_index(drop=True).iloc[keep_idx, :].reset_index(drop=True)
    stratify_vals = [val for val in stratify_vals if val not in remove_groups]
else:
    df_combined = df_combined.reset_index(drop=True)
        
    
#################################### STEP 5: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY BINARY PHENOTYPE AND PRIMARY LINEAGE ####################################


train_index, test_index = train_test_split(df_combined.index.values, test_size=0.2, stratify=stratify_vals)

df_combined.loc[train_index, "category"] = "original_train_set" 
df_combined.loc[test_index, "category"] = "original_test_set"

# print the means of the two groups as a check
print(df_combined.groupby("category")[["Binary", f"{drug}_midpoint"]].mean())
df_combined.to_csv(os.path.join(out_dir, "data_for_model.csv"), index=False)

print(f"Final: {df_combined.shape[0]} samples in the training data")


#################################### STEP : WRITE TXT FILE WITH THE PATHS OF THE VCF FILES WITH BOTH THE TRAINING AND VALIDATION DATASETS ####################################


# create a new txt file of paths, adding the validation file paths to the original file
with open(os.path.join(out_dir, "combined_paths_for_aln.txt"), "w+") as file:
    
    for sample_id in df_combined["ROLLINGDB_ID"].values:
        
        # this file contains all non-REF calls for each sample. It is also annotated with snpEff, hence the file extension
        fName = os.path.join(vcf_dir, f"{sample_id}/pilon/{sample_id}.eff.vcf")
        if not os.path.isfile(fName):
            raise ValueError(f"{fName} not found!")
        file.write(fName + "\n")
import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
import Bio.SeqUtils
import warnings
warnings.filterwarnings("ignore")


cc_df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/criticalConcentrations_updated.csv")

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

    return cc


_, drug = sys.argv
cc = get_critical_concentration(drug)

# who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
model_df = f"{out_dir}/data_intermediate_clean.csv"
isolate_variants = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/isolate_variants_fixed_annot.csv")


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
    

################## STEP 1: REMOVE ISOLATES WITH MULTIPLE RECORDED LINEAGES -- LOTS OF AMBIGUOUS CALLS DUE TO POLYCLONAL INFECTIONS OR SEQUENCING ERROR ##################


# lineages file should already be cleaned
lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv")
df_phenos = pd.read_csv(f"{out_dir}/data_with_paths.csv")#.query("DB_OF_ORIGIN=='CRyPTIC'")

# use Coll 2014 scheme
df_combined = df_phenos.merge(lineages[["ROLLINGDB_ID", "Coll2014"]], on="ROLLINGDB_ID", how="left")
assert len(df_combined) == len(df_phenos)

# create binary phenotype column
df_combined["Binary"] = (df_combined[f"{drug}_midpoint"] >= cc).astype(int)
print(df_combined.shape)

# remove isolates that have mixed lineages
df_combined = df_combined.loc[~df_combined['Coll2014'].str.contains(',')]
print(f"Removed {len(df_phenos)-len(df_combined)} isolates with multiple lineages")
prev_len = len(df_combined)
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
        
highConf_isolates = isolate_variants.query("mutation in @who_high_conf.mutation.values").Isolate.unique()
            
df_combined = df_combined.loc[~((df_combined["ROLLINGDB_ID"].isin(highConf_isolates)) & (df_combined[f"{drug}_midpoint"] < cc/2))]
print(f"Removed {prev_len - len(df_combined)} isolates with any of {len(who_high_conf)} category 1 mutations and MICs < 1/2 CC")

prev_len = len(df_combined)
df_combined = df_combined.query(f"~({drug}_lower_bound < @cc & {drug}_upper_bound > @cc)")
print(f"Removed {prev_len - len(df_combined)} isolates with MIC bounds that span the CC of {cc}")

# save at intermediate steps in case (because the previous step takes almost 1 hour)
df_combined.to_csv(model_df, index=False)
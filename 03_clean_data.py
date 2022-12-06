import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


_, drug, drug_who, cc, phenos_path = sys.argv


###### STEP 1: READ IN THE LINEAGES FILE AND COMBINE WITH THE PHENOTYPES FILE ######


# first 2 columns are the Isolate name and the Freschi lineage
lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv", sep="\t", header=None, usecols=[0, 1])
lineages.columns = ["ROLLINGDB_ID", "Lineage"]

# the Freschi lineages have "lineage" appended to the front, so remove that
lineages["Lineage"] = [val.lstrip("lineage") for val in lineages["Lineage"]]

df_phenos = pd.read_csv(phenos_path)

df_combined = df_phenos.merge(lineages, on="ROLLINGDB_ID", how="left")
assert len(df_combined) == len(df_phenos)
del df_phenos


###### STEP 2: REMOVE ISOLATES WITH CATEGORY 1 MUTATIONS AND MIC < 1/2 THE CC ######


# create binary phenotype column
df_combined["Binary"] = (df_combined[f"{drug}_midpoint"] > cc).astype(int)


# Get all category 1 variants (don't use category 2 to avoid dropping too many isolates and to be more stringent about what we drop)
who_high_conf = who_variants.loc[(who_variants["drug"] == drug_who) & (who_variants.confidence.str.contains("|".join(["1"])))].reset_index(drop=True)

for _, row in who_high_conf.iterrows():
    if "," in row["genome_index"]:
        expanded_pos = row["genome_index"].split(",")
        
        for pos in expanded_pos:
            add_df = pd.DataFrame({"drug": drug_who, "genome_index": pos, "confidence": row["confidence"], "gene": row["gene"], "variant": row["variant"]}, index=[len(who_high_conf)])
            who_high_conf = pd.concat([who_high_conf, add_df])
          
who_high_conf = who_high_conf.loc[~who_high_conf.genome_index.str.contains(",")]
who_high_conf = who_high_conf.drop_duplicates().reset_index(drop=True)
who_high_conf["genome_index"] = who_high_conf["genome_index"].astype(int)

aa_code_dict = {'Val':'V', 'Ile':'I', 'Leu':'L', 'Glu':'E', 'Gln':'Q', \
'Asp':'D', 'Asn':'N', 'His':'H', 'Trp':'W', 'Phe':'F', 'Tyr':'Y',    \
'Arg':'R', 'Lys':'K', 'Ser':'S', 'Thr':'T', 'Met':'M', 'Ala':'A',    \
'Gly':'G', 'Pro':'P', 'Cys':'C'}

code_aa_dict = {val: key for key, val in aa_code_dict.items()}

# convert them to 3-letter amino acid codes, which is what the ANN field in moxi_isolate_variants uses
for i, row in who_high_conf.iterrows():
    
    if len(row["variant"].split("_")) == 2:
        var = row["variant"].split("_")[1]
        expand_code = code_aa_dict[var[0]] + var[1:-1] + code_aa_dict[var[-1]]
        who_high_conf.loc[i, "ANN"] = expand_code
    else:
        who_high_conf.loc[i, "ANN"] = row["variant"]
        
        
        
###### STEP 3: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY MIC AND LINEAGE ######

        
# separate data points into bins. Use log-transformed MICs because they are normally distributed. Actual MICs are exponentially distributed
try:
    midpoint_bins = np.digitize(np.log(df_post_qc[drug+"_midpoint"]), bins=np.linspace(np.log(df_post_qc[drug+"_midpoint"].min()), np.log(df_post_qc[drug+"_midpoint"].max()), num=10))
    train_index, test_index = train_test_split(df_post_qc.index, test_size=0.2,
                                           stratify=midpoint_bins)

# the above will fail if there aren't enough isolates. In that case, don't stratify because it probably won't be even anyway
except:
    train_index, test_index = train_test_split(df_post_qc.index, test_size=0.2)

df_post_qc.loc[train_index, "category"] = "original_train_set" 
df_post_qc.loc[test_index, "category"] = "original_test_set"
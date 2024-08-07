import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
import Bio.SeqUtils
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

sys.path.append("utils")
from data_utils import *

h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")
cc_df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/criticalConcentrations_updated.csv")

# all variants in the 2023 WHO catalog regions, using their BED file
isolate_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/isolate_WHO_catalog_variants.csv").dropna(subset='variant').reset_index(drop=True)

# this contains the lineage designations from fast-lineage-caller and the F2 score
df_pass_QC = pd.read_csv("./data_processing/samples_pass_geno_QC.csv")# file of isolates that pass genotypic quality control and have a VCF file. Need to do F2 score filtering though
del df_pass_QC['DB_OF_ORIGIN']  # delete this column because it's already in combined_MIC.csv, but keep everything else


drug_abbr_dict = {'Amikacin': 'AMI',
                  'Bedaquiline': 'BDQ',
                  'Capreomycin': 'CAP',
                  'Clofazimine': 'CFZ',
                  'Delamanid': 'DLM',
                  'Ethambutol': 'EMB',
                  'Ethionamide': 'ETH',
                  'Isoniazid': 'INH',
                  'Kanamycin': 'KAN',
                  'Levofloxacin': 'LEV',
                  'Linezolid': 'LZD',
                  'Moxifloxacin': 'MXF',
                  'Ofloxacin': 'OFX',
                  'Prothionamide': 'PRO',
                  'Pyrazinamide': 'PZA',
                  'Rifampicin': 'RIF',
                  'Rifabutin': 'RFB',
                  'Streptomycin': 'STM'
                 }

abbr_drug_dict = {val: key for (key, val) in drug_abbr_dict.items()}

def get_critical_concentration(drug):

    drug_full_name = abbr_drug_dict[drug].upper()

    # get the row associcated with the particular drug
    for val in cc_df.query("antb == @drug_full_name").values[0]:

        # skip the columns of the drug or the abbreviation
        if val != drug_full_name and val != drug:
            
            # get the first non-null critical concentration. This will prefer 7H10 > 7H11 > LJ > MGIT > UKMYC
            if not pd.isnull(val):
                cc = val
                break

    return float(cc)


def reverse_complement(seq):
    
    comp_dict = {'A': 'T', 
                 'C': 'G', 
                 'G': 'C', 
                 'T': 'A', 
                 'N': 'N', 
                 '-': '-'
                }
    
    # this is to turn it into a list where each element is of length 1
    seq = list("".join(seq))
    
    if len(np.unique(seq)) > 6:
        raise ValueError(f"More than 6 types of characters in the sequence!")

    if "X" in np.unique(seq):
        raise ValueError(f"There are Xs in the sequence!")
        
    seq = [comp_dict[base] for base in seq] 
    
    # reverse the sequence and return as a list
    return "".join(seq[::-1])


_, drug = sys.argv

cc = get_critical_concentration(drug)
full_drug_name = abbr_drug_dict[drug]

out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
df_combined = pd.read_csv(f"{out_dir}/combined_MIC.csv")

prev_len = len(df_combined)
df_combined = df_combined.merge(df_pass_QC, on='ROLLINGDB_ID', how='inner')
print(f"{len(df_combined)}/{prev_len} isolates have high-quality WGS data")

# remove original MIC columns, then rename the NORM columns to the original names
for col in df_combined.columns:
    if drug in col and 'NORM' not in col:
        del df_combined[col]

for col in df_combined.columns:
    if drug in col:
        assert 'NORM' in col
        df_combined.rename(columns={col: col.replace('_NORM', '')}, inplace=True)


#################################### STEP 1: REMOVE ISOLATES WITH AN F2 SCORE ABOVE 0.05 ####################################


F2_thresh = 0.05
prev_len = len(df_combined)
df_combined = df_combined.query("F2 <= @F2_thresh").reset_index(drop=True)
print(f"Removed {prev_len-len(df_combined)} isolates with an F2 score > {F2_thresh}")


#################################### STEP 2: REMOVE ISOLATES WITH CATEGORY 1 MUTATIONS AND MIC < 1/2 THE CC ####################################


# check that the inverse of the filters is not met because some indels don't have all fields filled in
isolates_with_group1_mutations = isolate_variants.query("ROLLINGDB_ID in @df_combined.ROLLINGDB_ID.values & drug_V2==@full_drug_name & confidence_V2=='1) Assoc w R' & ~FILTER.str.contains('|'.join(['Del', 'LowCov'])) & ~(AF <= 0.75) & ~(DP < 5) & ~(BQ < 20) & ~(MQ < 30)").ROLLINGDB_ID.unique()

prev_len = len(df_combined)
df_combined = df_combined.query(f"~(ROLLINGDB_ID in @isolates_with_group1_mutations & {drug}_upper_bound < {cc/2})")
print(f"Removed {prev_len - len(df_combined)} isolates with a Group 1 variant and an MIC upper bound < {cc/2} µg/mL")

# unclear, sometimes there are samples with MIC <= 0, so midpoint = 0 which doesn't make any sense
df_combined = df_combined.query(f"{drug}_midpoint != 0")

# kind of a weird case, but in some studies, they basically just measured R vs. S, but they record the MIC as <= CC or > CC. Uninformative because it's not MIC data, it's binary data
prev_len = len(df_combined)
df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == 0) & (df_combined[f"{drug}_upper_bound"] == cc))]
df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == cc) & (df_combined[f"{drug}_upper_bound"] == np.inf))]
print(f"Removed {prev_len - len(df_combined)} isolates with MICs that are known only relative to the critical concentration of {cc}")

# because the CC for RIF was updated, also include the CC of 1
if drug == "RIF":
    prev_len = len(df_combined)
    df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == 0) & (df_combined[f"{drug}_upper_bound"] == 1))]
    df_combined = df_combined.loc[~((df_combined[f"{drug}_lower_bound"] == 1) & (df_combined[f"{drug}_upper_bound"] == np.inf))]
    print(f"Removed {prev_len - len(df_combined)} isolates with MICs that are known only relative to the old critical concentration of 1")

# remove the following from datasets because otherwise not sure what to do about sensitivity/specificity. i.e. is such a sample R or S?
prev_len = len(df_combined)

# don't remove them from the dataframe so that they remain in the alignment files
df_combined.loc[(df_combined[f"{drug}_lower_bound"] < cc) & (df_combined[f"{drug}_upper_bound"] > cc), "Span_CC"] = 1
df_combined['Span_CC'] = df_combined['Span_CC'].fillna(0).astype(int)

print(f"{sum(df_combined['Span_CC'])} isolates have MIC bounds that span the CC of {cc}")


def remove_isolates_multiple_amb_frameshifts(df_variants):
    '''
    Isolates with multiple frameshifts with the Amb tag (AF ≤ 0.75) may be cases with mixed populations. 
    
    They can be of the same lineage, so they are not filtered out using the F2 metric. 

    Looking through some manually, they look like real frameshifts, but no reads appear to support both frameshifts, so they may not both be occurring in any cell.

    To be safe, remove these isolates so that they don't bias the models.

    We will keep isolates with multiple inframe indels because if they aren't in the same cell, the change in the AA sequence is not as drastic as if frameshifts are misencoded.
    '''
    
    kwargs = yaml.safe_load(open(f"config_files/config_{drug.lower()}.yaml", "r"))
    locus_list = kwargs['tier1_loci'] + kwargs['tier2_loci']
    genes_lst = get_genes_lst(locus_list)

    # count the number of Amb frameshifts are in each isolate, gene pair
    # then keep only the cases with more than 1 such frameshift

    # already filtered df_variants to keep only isolates that are in the phenotypes dataframe
    frameshift_count_by_gene = df_variants.query("GENE in @genes_lst & variant.str.contains('fs') & FILTER.str.contains('Amb') & AF <= 0.75").groupby(['ROLLINGDB_ID', 'GENE'])['POS'].count().reset_index().query("POS > 1")

    print(frameshift_count_by_gene)
    return frameshift_count_by_gene.ROLLINGDB_ID.unique()


#################################### STEP 3: REMOVE ISOLATES WITH MULTIPLE AMBIGUOUS FRAMESHIFT MUTATIONS IN THE GENES OF INTEREST


# keep only isolates in the phenotypes dataframe, then drop duplicates (because there are duplicate variants with one line per drug they are related to)
single_drug_variants = isolate_variants.query("ROLLINGDB_ID in @df_combined.ROLLINGDB_ID.values").drop_duplicates(['ROLLINGDB_ID', 'variant'], keep='first')

isolates_with_multiple_Amb_frameshifts_per_gene = remove_isolates_multiple_amb_frameshifts(single_drug_variants)
print(isolates_with_multiple_Amb_frameshifts_per_gene)

print(f"Removed {len(isolates_with_multiple_Amb_frameshifts_per_gene)} isolates with multiple ambiguous frameshifts in a single gene")

df_combined = df_combined.query("ROLLINGDB_ID not in @isolates_with_multiple_Amb_frameshifts_per_gene")


def get_primary_lineage(lineage_str):

    # get the first number from numeric lineages. Take the unique ones so that i.e. 4.1.2 and 4.3 will result in a single 4
    split_lineage = np.unique([val[0] if val[0].isnumeric() else val for val in lineage_str.split(',')])

    # if there are multiple primary lineages, then return a sorted list (then joined into a string separated by commas). If there is only one, return the single one as a string
    if len(split_lineage) == 1:
        return split_lineage[0]
    else:
        return ','.join(np.sort(split_lineage))
        
# separate for stratification because the isolates with MICs that span the CC shouldn't be part of the train or validation sets. They will manually be placed in the test set
df_MIC_span_CC = df_combined.query("Span_CC==1")
df_combined = df_combined.query("Span_CC==0")

# need to do this because 1) confounding and 2) when stratifying the groups by primary lineage and binary phenotype, there needs to be at least 1 in each group
# BECAUSE WE ARE USING THE UPPER BOUND, NOT THE MIDPOINT, SHOULD BE EXCLUSIVE FOR DETRMINING BINARY RESISTANCE
# this is because i.e. PZA = (50, 100) means susceptible, even though PZA_upper_bound = 100
df_combined["Binary"] = (df_combined[f"{drug}_lower_bound"] >= cc).astype(int)
df_combined["Lineage"] = [get_primary_lineage(lineage) for lineage in df_combined["Coll2014"]]
df_combined['Stratify'] = df_combined["Lineage"] + "-" + df_combined["Binary"].astype(str)

# find lineage-phenotype groups that don't have at least 2 isolates
# this is because the train-test splitting will fail due to the least populated class in y having only 1 member
# BUT, then you have to remove all isolates in that lineage, otherwise you risk biasing the data to have only R or only S isolates for that lineage
stratify_df = pd.Series(df_combined['Stratify'].values).value_counts().reset_index()
stratify_df.columns = ["stratify", "count"]
remove_lineages = [val.split('-')[0] for val in stratify_df.query("count < 2").stratify.values]

# at the end, reset index so that index can be used for train/test splitting
if len(remove_lineages) > 0:
    
    prev_len = len(df_combined)
    df_combined = df_combined.query("Lineage not in @remove_lineages")
    print(f"Removed {prev_len - len(df_combined)} isolates in the {remove_lineages} lineage groups with fewer than 2 isolates in one of the binary resistance groups")

    # remake the Stratify column for the purposes of splitting train + validation from test
    df_combined['Stratify'] = df_combined["Lineage"] + "-" + df_combined["Binary"].astype(str)
    

df_combined = df_combined.reset_index(drop=True)

    
#################################### STEP 5: CREATE TRAIN AND TEST SPLITS, STRATIFYING BY BINARY PHENOTYPE AND PRIMARY LINEAGE ####################################


# split train + validation from test, stratifying by binary phenotype and primary lineage. 80% train, 10% validation, 10% test
train_index, test_index = train_test_split(df_combined.index.values, test_size=0.2, stratify=df_combined['Stratify'].values)

df_combined.loc[train_index, "category"] = "train_set" 
df_combined.loc[test_index, "category"] = "test_set"

# # do it again on the test set only to split test and validation
# df_train = df_combined.query("category=='train_set'").reset_index(drop=True)
# train_index, validation_index = train_test_split(df_train.index.values, test_size=float(1/9), stratify=df_train['Binary'].values)

# df_train.loc[train_index, "category"] = "train_set" 
# df_train.loc[validation_index, "category"] = "validation_set"

# df_combined = pd.concat([df_combined.query("category=='test_set'"), df_train]).reset_index(drop=True)
# assert df_combined['ROLLINGDB_ID'].nunique() == len(df_combined) # check that no IDs were duplicated

# put the isolates with MICs that span the CC back into the dataframe, all in the test set. Don't add Binary because MIC spans the CC 
df_MIC_span_CC['category'] = 'test_set'
df_MIC_span_CC["Lineage"] = [get_primary_lineage(lineage) for lineage in df_MIC_span_CC["Coll2014"]]

df_combined = pd.concat([df_combined, df_MIC_span_CC], axis=0)
print(f"Final: {df_combined.shape[0]} samples in the training data")
# print(df_combined['DB_OF_ORIGIN'].value_counts())

# print the means of the two groups as a check
print(df_combined.groupby("category")[["Binary", f"{drug}_midpoint"]].mean())
df_combined.to_csv(os.path.join(out_dir, "data_for_model.csv"), index=False)


#################################### STEP : WRITE TXT FILE WITH THE PATHS OF THE VCF FILES WITH BOTH THE TRAINING AND VALIDATION DATASETS ####################################


# create a new txt file of paths, adding the validation file paths to the original file
with open(os.path.join(out_dir, "combined_paths_for_aln.txt"), "w+") as file:

    # already reset the index above and previously checked that all files exist
    for fName in df_combined['VCF']:
        new_fName = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF_clean/{os.path.basename(fName).replace('_variants', '')}"
        assert os.path.isfile(new_fName)
        file.write(new_fName + "\n")
import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
import warnings
warnings.filterwarnings("ignore")

### DONE PREVIOUSLY TO GENERATE THE FULL DATA FILE
    
# cryptic = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/cryptic_data_curated_filtered.csv")
# rollingdb = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_combined_data.csv")

# full_df = pd.concat([cryptic, rollingdb])

# # cryptic has a quality column. Rollingdb does not, so fill those with high so that they don't get removed later
# full_df.loc[full_df.DB_OF_ORIGIN != "CRyPTIC", full_df.columns[full_df.columns.str.contains("quality")]] = "HIGH"

# combined dataframe with all ~14,000 isolates with any MIC data is at /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_rollingdb_cryptic.csv

### THIS SCRIPT EXTRACTS DATA FOR A SINGLE DRUG, GETS THE VCF FILE PATHS, AND STANDARDIZES THE MICS (INCLUDING GETTING BOUNDS) ###


def extract_single_drug(data_df, drug, metadata_cols_num=9):
    '''
    Extracts data for isolates with quality score of at least MEDIUM. 
    '''

    data_df = data_df.rename(columns={"LEVO": "LEV",
                                      "LEVO_lower_bound": "LEV_lower_bound",
                                      "LEVO_midpoint": "LEV_midpoint",
                                      "LEVO_upper_bound": "LEV_upper_bound",
                                      "LEVO_quality": "LEV_quality"
                                     })

    drug = drug.upper()

    if drug + "_midpoint" not in data_df.columns:
        raise ValueError(f"Drug {drug} is not a valid drug name!")

    # sometimes the quality column is missing, so check them all. if it's not there, keep all isolates
    if drug + "_quality" in data_df.columns:
        high_quality = data_df.loc[data_df[drug+"_quality"].isin(["MEDIUM", "HIGH"])]
    else:
        high_quality = data_df.copy()

    # first several columns the metadata columns
    keep_df = high_quality[np.concatenate([data_df.columns[:metadata_cols_num], data_df.columns[data_df.columns.str.contains(drug, case=False)]])]

    if drug in ["BDQ", "DLM"]:
        keep_df.loc[:, "MEDIA"] = "m7h11"
    else:
        keep_df.loc[:, "MEDIA"] = "m7h10"

    single_drug_df = keep_df.loc[~pd.isnull(keep_df[drug + "_midpoint"])]
    
    # this is basically just for the Cryptic data, sometimes the drug field is missing. Replace it with midpoint
    # all 3 columns are the same: drug_lower_bound, drug_midpoint, drug_upper_bound, but one of the bounds will be NaN if the MIC is at the extremes
    # midpoint field is never NaN
    single_drug_df.loc[pd.isnull(single_drug_df[drug]), drug] = single_drug_df.loc[pd.isnull(single_drug_df[drug])][f"{drug}_midpoint"].astype(str)
    assert len(single_drug_df.dropna(subset=[drug, f"{drug}_midpoint"])) == len(single_drug_df)
    print(f"Found {len(single_drug_df)} isolates for {drug}")

    return single_drug_df
    
    
    
def standardize_MICs(df, drug):
    '''
    For CRyPTIC data, the recorded MIC becomes the upper bound. The lower bound is one dilution lower (typically 1/2 of the upper bound). The midpoint is the average.
    
    For ROLLINGDB data, most of it is already in bound form, so the upper and lower bounds are extracted, and the average becomes the midpoint. Isolates with only a single MIC (no range) recorded
    are removed. 
    
    MICs at the upper extreme have the same value for the lower and upper bounds and the midpoint. MICs at the lower extreme have lower = 0, upper = MIC, and midpoint = average. 
    '''
    
    df_new = df.copy()
    drug = drug.upper()
    new_midpoints = []
    
    cryptic_unique = df.loc[df.Path.str.contains('cryptic')][drug].unique()
    cryptic_unique = list(np.sort(np.unique([float(num.strip(">").strip("<=")) for num in cryptic_unique])))
    
    # get a gigantic value (impossible to get any MIC at that value for the upper bounds of MICs listed as > N)
    max_val = np.max(df_new[f"{drug}_midpoint"])*1000

    for i, row in df_new.iterrows():
        if "<" in row[drug]:
            val = float(row[drug].strip("<="))
            df_new.loc[i, f"{drug}_lower_bound"] = 0
            df_new.loc[i, f"{drug}_upper_bound"] = val
            df_new.loc[i, f"{drug}_midpoint"] = val / 2
        # for this case, put the value for everything because the error function should be computed normally
        elif ">" in row[drug]:
            val = float(row[drug].strip(">"))
            df_new.loc[i, f"{drug}_lower_bound"] = val
            df_new.loc[i, f"{drug}_upper_bound"] = max_val
            df_new.loc[i, f"{drug}_midpoint"] = val
        elif "-" in row[drug]:
            lower = float(row[drug].split("-")[0])
            upper = float(row[drug].split("-")[1])
            df_new.loc[i, f"{drug}_lower_bound"] = lower
            df_new.loc[i, f"{drug}_upper_bound"] = upper
            df_new.loc[i, f"{drug}_midpoint"] = np.mean([lower, upper])
        else:
            upper = float(row[drug])
            if "cryptic" in row["Path"]:

                if cryptic_unique.index(upper) > 0:
                    lower = cryptic_unique[cryptic_unique.index(upper)-1]
                else:
                    lower = 0

                df_new.loc[i, f"{drug}_lower_bound"] = lower

                if ">" in row[drug]:
                    df_new.loc[i, f"{drug}_upper_bound"] = max_val
                else:
                    df_new.loc[i, f"{drug}_upper_bound"] = upper

                df_new.loc[i, f"{drug}_midpoint"] = np.mean([lower, upper])
            # because we don't know the tested concentrations for the non-cryptic data, remove them if only one concentration was tested
            else:
                df_new.loc[i, f"{drug}_midpoint"] = np.nan

    # check that bounds make sense with the midpoints
    assert len(df_new.query(f"{drug}_lower_bound > {drug}_midpoint")) == 0
    assert len(df_new.query(f"{drug}_upper_bound < {drug}_midpoint")) == 0
                
    print(f"{len(df_new.loc[pd.isnull(df_new[f'{drug}_midpoint'])])} isolates without MIC ranges were removed")
    return df_new.dropna(subset=f"{drug}_midpoint", how="any")
    

    
def get_isolate_paths_and_process(df, output_dir, drug, cryptic_genomic_path, rollingdb_genomic_path):
    '''
    This function gets the paths to all the .vcf files specified in the input dataframe. It outputs the paths to a new text file in a new directory. 
    
    It also separates the data into training and testing subsets.
    '''
    
    print(f"Looking for VCF paths...")
        
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():

        # get the current rollingDB ID from the dataframe
        rollingDB = str(row["ROLLINGDB_ID"])
        ID = str(row["ID"])
        accession = str(row["BIOSAMPLE_ACCESSION"])

        # don't think it's necessary to do the exhaustive itertools search based on the 2 datasets worked with so far
        # but do it anyway to be complete, it doesn't take much longer to run this
        all_names = list(itertools.product([rollingDB, ID, accession], [rollingDB, ID, accession]))

        # sometimes RollingDB ID and Bioaccession number are the same, so take the unique ones to speed up the code
        if row["DB_OF_ORIGIN"].lower() == "cryptic":
            possible_paths = np.unique([os.path.join(cryptic_genomic_path, name[0], "pilon", name[1] + "_full.vcf.gz") for name in all_names])
        else:
            possible_paths = np.unique([os.path.join(rollingdb_genomic_path, name[0], "pilon", name[1] + "_full.vcf.gz") for name in all_names])

        # if the expected file exists, add it to the dataframe and break out of the loop because the path was found
        for path_name in possible_paths:
            if os.path.isfile(path_name):             
                df.loc[i, "Path"] = path_name
                break
                
    # drop isolates with no full VCF file available (these will have NaN for the Path column)
    df_post_qc = df.dropna(subset="Path")
    
    # print the number of isolates for which data is unavailable because it didn't pass QC
    print(f"{len(df_post_qc)} out of {len(df)} isolates passed QC")
    
    # standardize MICs. Now nothing should be NaN in the drug MIC columns
    df_post_qc = standardize_MICs(df_post_qc, drug)
    assert len(df_post_qc.dropna(subset=[drug, f"{drug}_lower_bound", f"{drug}_midpoint", f"{drug}_upper_bound"])) == len(df_post_qc)
    
    # save the full data file and a file of the paths
    df_post_qc.to_csv(os.path.join(output_dir, "data_with_paths.csv"), index=False)
    pd.Series(df_post_qc["Path"].values).to_csv(os.path.join(output_dir, "paths.txt"), sep="\t", header=None, index=False)
    
       
_, drug = sys.argv

output_dir = os.path.join("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs", drug.upper())
cryptic_genomic_path = "/n/data1/hms/dbmi/farhat/rollingDB/cryptic_output"
rollingdb_genomic_path = "/n/data1/hms/dbmi/farhat/rollingDB/genomic_data"
    
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_rollingdb_cryptic.csv")
print(f"Full data shape: {df.shape}")

# get the dataframe of MICs for a single drug
single_drug_df = extract_single_drug(df, drug)
    
# create a text file of the VCF paths, copy the CSV made in the line above, and add the path column to it
get_isolate_paths_and_process(single_drug_df, output_dir, drug, cryptic_genomic_path, rollingdb_genomic_path)
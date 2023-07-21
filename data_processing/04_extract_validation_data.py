import pandas as pd
import numpy as np
import sys, os, glob

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
validation_data = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/MIC_ML_consortium_MIC_table.csv")
validation_data_metadata = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/isolate_metadata.csv")

MIC_ML_data = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/data.csv"

if not os.path.isfile(MIC_ML_data):

    print(f"Combining MIC-ML data into a single dataframe {MIC_ML_data}")
    
    paths_df = pd.DataFrame(columns=["ROLLINGDB_ID", "FASTQ1_FILE", "FASTQ2_FILE"] + list(validation_data.columns[~validation_data.columns.isin(["ID", "BIOSAMPLE_ACCESSION", "ROLLINGDB_ID"])])).set_index("ROLLINGDB_ID")

    for _, row in validation_data.iterrows():

        possible_dirs = [row["ID"], row["BIOSAMPLE_ACCESSION"], row["ROLLINGDB_ID"]]
        found_dir = False

        for dir_name in possible_dirs:
            if not pd.isnull(dir_name):
                sample_dir = f"/n/data1/hms/dbmi/farhat/rollingDB/fastq_db/{dir_name}"

                if os.path.isdir(sample_dir):
                    sample_id = os.path.basename(sample_dir)
                    found_dir = True
                    break

        if found_dir:
            reads_1 = f"{sample_dir}/{dir_name}_R1.fastq.gz"
            reads_2 = f"{sample_dir}/{dir_name}_R2.fastq.gz"

            if not os.path.isfile(reads_1) or not os.path.isfile(reads_2):
                #print(f"{sample_dir} does not contain both reads files!")
                if validation_data_metadata.query("ROLLINGDB_ID == @sample_id")["UNPAIRED/PAIRED"].values[0] == 0:
                    print(f"{sample_dir} contains unpaired data!")
            else:            
                # add the remaining columns to the dataframe as well
                add_data = pd.DataFrame(row).T
                add_data = add_data[add_data.columns[~add_data.columns.isin(["ID", "BIOSAMPLE_ACCESSION", "ROLLINGDB_ID"])]]

                paths_df.loc[dir_name, :] = [reads_1, reads_2] + list(np.squeeze(add_data.values))


    paths_df = paths_df.reset_index()
    print(paths_df.shape)

    assert len(paths_df.ROLLINGDB_ID.unique()) == len(paths_df)

    # standardize names for use with the World Health Organization mutation catalog   
    prefix_conv_dict = {"MOXI": "MXF", "LEVO": "LEV", "LIN": "LZD", "CLO": "CFZ", "ETA": "ETH"}
    
    for i, (old, new) in enumerate(prefix_conv_dict.items()):
        for col in paths_df.columns:
            if old in col and col != "ROLLINGDB_ID":
                print(col)
                paths_df.rename(columns={col: col.replace(old, new)}, inplace=True)

    # save full dataframe
    paths_df.to_csv(MIC_ML_data, index=False)
    
    # save the 3 columns for running the processing pipeline to get VCF files
    paths_df[["ROLLINGDB_ID", "FASTQ1_FILE", "FASTQ2_FILE"]].to_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML_metadata_for_megapipe.tsv", index=False, header=None, sep="\t")
    
else:
    paths_df = pd.read_csv(MIC_ML_data)
    
    
# keep only samples that have MICs measured for the drug of interest
# paths_df = paths_df.loc[~pd.isnull(paths_df[f"{drug}_midpoint"])]

# # only keep the columns relevant for the drug of interest, then save the dataframe
# metadata_cols = 9

# col_list = []

# for col in [f"{drug}_quality", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound"]:
#     if col in paths_df.columns:
#         col_list.append(col)

def process_bounds_MICML_data(df, drug):

    if f"{drug}_midpoint" not in df.columns:
        print(f"There are no validation samples for {drug}. Quitting this script...")
        exit()
        
    df_single_drug = df.loc[~pd.isnull(df[f"{drug}_midpoint"])]
    new_dfs = []
    
    for db in df_single_drug["DB_OF_ORIGIN"].unique():
        
        df_single_db = df_single_drug.query("DB_OF_ORIGIN==@db").reset_index(drop=True)
        
        lower, midpoint, upper = df_single_db[[f"{drug}_lower_bound", f"{drug}_midpoint", f"{drug}_upper_bound"]].values.T
        db_bounds = list(np.sort(np.unique(np.concatenate([lower, midpoint, upper]))))
        print(db, db_bounds)
        
        for i, row in df_single_db.iterrows():
            
            if (row[f"{drug}_lower_bound"] == row[f"{drug}_midpoint"]) or (row[f"{drug}_upper_bound"] == row[f"{drug}_midpoint"]):
                
                # set the midpoint to the second smallest value
                if row[f"{drug}_midpoint"] == 0:
                    midpoint_idx = 1
                else:
                    midpoint_idx = db_bounds.index(row[f"{drug}_midpoint"])
                
                # the recorded value is the upper bound
                new_high = db_bounds[midpoint_idx]
                
                # if the upper bound is the smallest concentration, the lower bound should be 0
                if midpoint_idx == 0:
                    new_low = 0 #db_bounds[midpoint_idx]
                # the lower bound should be 1 concentration below the upper bound
                else:
                    new_low = db_bounds[midpoint_idx-1]
                
                df_single_db.loc[i, [f"{drug}_lower_bound", f"{drug}_upper_bound", f"{drug}_midpoint"]] = [new_low, new_high, np.mean([new_low, new_high])]
                
        new_dfs.append(df_single_db)
    
    df_final = pd.concat(new_dfs, axis=0)
    assert len(df_final) == len(df_single_drug)
    assert len(set(df_final["ROLLINGDB_ID"]).symmetric_difference(df_single_drug["ROLLINGDB_ID"])) == 0
    
    cols = ['ID', 'BIOSAMPLE_ACCESSION', 'ROLLINGDB_ID', 'ISOLATION_LOCATION',
       'DB_OF_ORIGIN', 'STUDY_NAME', 'STUDY_PMID', 'TESTING_LOCATION', 'MEDIA'] + [f"{drug}_quality", f"{drug}_lower_bound", f"{drug}_midpoint", f"{drug}_upper_bound"]
    cols_lst = []
    
    for col in cols:
        if col in df_final.columns:
            cols_lst.append(col)

    return df_final[cols_lst]
    
        
# paths_df = paths_df[list(paths_df.columns[:metadata_cols]) + col_list]       
paths_df = process_bounds_MICML_data(paths_df, drug)
prev_len = len(paths_df)
paths_df = paths_df.query(f"~({drug}_lower_bound < @cc & {drug}_upper_bound > @cc)")
print(f"Removed {prev_len - len(paths_df)} isolates with MIC bounds that span the CC of {cc}")

paths_df.to_csv(os.path.join(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/validation_data.csv"), index=False)

# create text file for later to get the paths of the VCF files in the validation dataset
paths_txt_file = pd.Series([f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/VCF/{isolate}/pilon/{isolate}.vcf" for isolate in paths_df["ROLLINGDB_ID"].values])

for fName in paths_txt_file:
    assert os.path.isfile(fName)

paths_txt_file.to_csv(os.path.join(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/validation_paths.txt"), sep="\t", header=None, index=False)
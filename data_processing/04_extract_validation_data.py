import pandas as pd
import numpy as np
import sys, os, glob


_, drug, cc = sys.argv

# paths_df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML_data.csv")
validation_data = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/MIC_ML_consortium_MIC_table.csv").rename(columns={"MOXI_lower_bound": "MXF_lower_bound",
                                                                                                                                  "MOXI_midpoint": "MXF_midpoint",
                                                                                                                                  "MOXI_upper_bound": "MXF_upper_bound",
                                                                                                                                  "MOXI_quality": "MXF_quality"
                                                                                                                                 })

validation_data_metadata = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/isolate_metadata.csv")

MIC_ML_data = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML_data.csv"

if not os.path.isfile(MIC_ML_data):

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

    # save full dataframe
    paths_df.to_csv(MIC_ML_data, index=False)
    
    # save the 3 columns for running the processing pipeline to get VCF files
    paths_df[["ROLLINGDB_ID", "FASTQ1_FILE", "FASTQ2_FILE"]].to_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML_metadata_for_megapipe.tsv", index=False, header=None, sep="\t")
    
else:
    paths_df = pd.read_csv(MIC_ML_data)
    
    
# keep only samples that have MICs measured for the drug of interest
paths_df = paths_df.loc[~pd.isnull(paths_df[f"{drug}_midpoint"])]

# only keep the columns relevant for the drug of interest, then save the dataframe
metadata_cols = 9
paths_df = paths_df[list(paths_df.columns[:metadata_cols]) + [f"{drug}_quality", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound"]]       

prev_len = len(paths_df)
paths_df = paths_df.query(f"~({drug}_lower_bound < @cc & {drug}_upper_bound > @cc)")
print(f"Removed {prev_len - len(paths_df)} isolates with MIC bounds that span the CC of {cc}")

paths_df.to_csv(os.path.join(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/validation_data.csv"), index=False)
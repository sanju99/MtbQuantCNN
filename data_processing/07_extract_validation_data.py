import pandas as pd
import numpy as np
import sys, os, glob, yaml

cc_df = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/critical_concentrations_all.csv", index_col=[0])
del cc_df["ABBR"]
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

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

abbr_drug_dict = {value: key for key, value in drug_abbr_dict.items()}

# critical concentrations values : MIC-ML naming convention
media_dict = {'mgit960/mgit7h9': 'MGIT960/MGIT7h9',
              'BMD': 'Broth micro-dilution (BMD)',
              'nsw_sensititre': 'NSW sensititre plate',
              'UKMYC': 'UKMYC (Cryptic sensititre)',
              'MYCOTB': 'MYCOTB',
              'm7h10': 'M7H10',
              'm7h11': 'M7H11'
             }

def get_critical_concentration(drug):

    full_drug_name = abbr_drug_dict[drug]
    
    # get the row associcated with the particular drug
    for val in cc_df.loc[full_drug_name.upper()].values:

        # skip the columns of the drug or the abbreviation
        if val not in [full_drug_name, drug]:
            
            # get the first non-null critical concentration
            if not pd.isnull(val):
                cc = val
                break

    return cc


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
binary_thresh = kwargs["binary_thresh"]
out_dir = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}"
vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"

validation_data = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/MIC_ML_consortium_MIC_table.csv")
validation_data_metadata = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/isolate_metadata.csv")

inconsistent_FASTQ = pd.read_csv("/home/sak0914/MtbQuantCNN/data_processing/micml_weird_fastq.txt", sep="\t", header=None)[0].values
MIC_ML_data = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/data_with_paths.csv"

who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")
who_variants["gene"] = [val.split("_")[0] for val in who_variants.mutation.values]
who_variants["variant"] = ["_".join(val.split("_")[1:]) for val in who_variants.mutation.values]

if os.path.isfile(os.path.join(data_dir, drug, "isolate_variants_fixed_annot.csv")):
    highConf = True
    isolate_variants = pd.read_csv(os.path.join(data_dir, drug, "isolate_variants_fixed_annot.csv"))
else:
    highConf = False
    
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
                    print(f"No FASTQs for {sample_id}!")
            else:            
                # add the remaining columns to the dataframe as well
                add_data = pd.DataFrame(row).T
                add_data = add_data[add_data.columns[~add_data.columns.isin(["ID", "BIOSAMPLE_ACCESSION", "ROLLINGDB_ID"])]]

                paths_df.loc[dir_name, :] = [reads_1, reads_2] + list(np.squeeze(add_data.values))


    paths_df = paths_df.reset_index()
    print(f"Found paired FASTQs for {len(paths_df)} samples")
    
    paths_df = paths_df.query("ROLLINGDB_ID not in @inconsistent_FASTQ")
    print(f"Removed paired FASTQs with different line counts for {len(inconsistent_FASTQ)} samples")

    assert len(paths_df.ROLLINGDB_ID.unique()) == len(paths_df)
    print(f"Found paired, consistent FASTQs for {len(paths_df)} samples")

    # standardize names for use with the World Health Organization mutation catalog   
    prefix_conv_dict = {"MOXI": "MXF", "LEVO": "LEV", "LIN": "LZD", "CLO": "CFZ", "ETA": "ETH"}

    # rename prefixes for consistency with other code / analyses
    for i, (old, new) in enumerate(prefix_conv_dict.items()):
        for col in paths_df.columns:
            if old in col and col != "ROLLINGDB_ID":
                paths_df.rename(columns={col: col.replace(old, new)}, inplace=True)

    # save full dataframe
    paths_df.to_csv(MIC_ML_data, index=False)
    
    # save the 3 columns for running the processing pipeline to get VCF files
    paths_df[["ROLLINGDB_ID", "FASTQ1_FILE", "FASTQ2_FILE"]].to_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/metadata_for_megapipe.tsv", index=False, header=None, sep="\t")
    
else:
    paths_df = pd.read_csv(MIC_ML_data)
    
    

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

        # check that each study only used one medium for each drug
        assert df_single_db['MEDIA'].nunique() == 1

        print(f"Study: {db}, Medium: {df_single_db['MEDIA'].unique()[0]}, Breakpoints: {db_bounds}")
        
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



def normalize_validation_data(df, drug):

    df_copy = paths_df.copy().reset_index(drop=True)
    full_drug_name = abbr_drug_dict[drug]
    
    cc_df_single_drug = pd.DataFrame(dict(cc_df.loc[full_drug_name.upper()]), index=[0]).T.reset_index()
    cc_df_single_drug.columns = ["Media", "CC"]
    
    m7h10_cc = cc_df_single_drug.query("Media=='m7h10'")["CC"].values[0]
    
    # use M7H11 if M7H10 is not there. Use the same variable name though
    if pd.isnull(m7h10_cc):
        m7h10_cc = cc_df_single_drug.query("Media=='m7h11'")["CC"].values[0]
    
        # also update the dictionary for mapping purposes
        media_dict['m7h11'] = 'M7H10'
    
    cc_df_single_drug["MIC_ML_Media_Name"] = cc_df_single_drug["Media"].map(media_dict)    
    cc_dict_single_drug = dict(zip(cc_df_single_drug["MIC_ML_Media_Name"], cc_df_single_drug["CC"]))

    # add the critical concentration for ach media to the dataframe
    df_copy['MEDIA_CC'] = df_copy['MEDIA'].map(cc_dict_single_drug)

    if len(df_copy.loc[pd.isnull(df_copy["MEDIA_CC"])]) > 0:
        print(m7h10_cc, cc_dict_single_drug)
        raise ValueError("There are NaNs!")

    df_copy.rename(columns={f"{drug}_lower_bound": f"{drug}_lower_bound_original", 
                            f"{drug}_midpoint": f"{drug}_midpoint_original", 
                            f"{drug}_upper_bound": f"{drug}_upper_bound_original"}
                  , inplace=True)

    # multiply each MIC by the ratio of the M7H10 CC to the current media MIC = M7H10 MIC
    df_copy[f"{drug}_lower_bound"] = df_copy[f"{drug}_lower_bound_original"] / df_copy["MEDIA_CC"] * m7h10_cc
    df_copy[f"{drug}_midpoint"] = df_copy[f"{drug}_midpoint_original"] / df_copy["MEDIA_CC"] * m7h10_cc
    df_copy[f"{drug}_upper_bound"] = df_copy[f"{drug}_upper_bound_original"] / df_copy["MEDIA_CC"] * m7h10_cc
    
    # for i, row in df_copy.iterrows():

    #     lower, mid, upper = row[f"{drug}_lower_bound"], row[f"{drug}_midpoint"], row[f"{drug}_upper_bound"]
        
    #     if lower < cc and upper > cc:

    #         # assign the lower bound to be the critical concentration, leave upper, and update the midpoint
    #         if cc - lower < 0.05:
    #             df_copy.loc[i, [f"{drug}_lower_bound", f"{drug}_midpoint", f"{drug}_upper_bound"]] = [cc, np.mean([cc, upper]), upper]

    #         # assign the upper bound to be the critical concentration, leave lower, and update the midpoint
    #         elif upper - cc < 0.05:
    #             df_copy.loc[i, [f"{drug}_lower_bound", f"{drug}_midpoint", f"{drug}_upper_bound"]] = [lower, np.mean([cc, lower]), cc]
    
    # assert len(df_copy.query(f"{drug}_lower_bound > {drug}_midpoint")) == 0
    # assert len(df_copy.query(f"{drug}_upper_bound < {drug}_midpoint")) == 0
    assert len(df) == len(df_copy)

    assert len(df_copy.query(f"{drug}_midpoint == 0")) == 0
    assert len(df_copy.query(f"{drug}_lower_bound > {drug}_midpoint")) == 0
    assert len(df_copy.query(f"{drug}_upper_bound < {drug}_midpoint")) == 0
    assert len(df_copy.query(f"{drug}_midpoint_original == {drug}_lower_bound_original | {drug}_midpoint_original == {drug}_upper_bound_original")) == 0
    assert len(df_copy.query(f"{drug}_midpoint == {drug}_lower_bound | {drug}_midpoint == {drug}_upper_bound")) == 0

    return df_copy
    

# critical concentration in primary media (usually M7H10)
cc = get_critical_concentration(drug)
        
# paths_df = paths_df[list(paths_df.columns[:metadata_cols]) + col_list]       
paths_df = process_bounds_MICML_data(paths_df, drug)
paths_df = normalize_validation_data(paths_df, drug)
paths_df.to_csv(os.path.join(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/validation_data.csv"), index=False)


############################### CLEANING STEP 1: REMOVE SAMPLES WHERE THE MIC SPANS THE CC ###############################


# because this makes computing binary metrics (sens, spec, etc.) weird
prev_len = len(paths_df)
paths_df = paths_df.query(f"~({drug}_lower_bound < @binary_thresh & {drug}_upper_bound > @binary_thresh)")
print(f"Removed {prev_len - len(paths_df)} isolates with MIC bounds that span the CC of {binary_thresh}")


############################### CLEANING STEP 2: EXCLUDE SAMPLES WITH LARGE PROPORTIONS OF READS THAT DO NOT MAP TO MTBC ###############################


unclassified_thresh = 10
high_unclassified_prop = []

# for each sample, get the distribution of classified (MTBC) vs. unclassified (not MTBC) reads
for sample_id in paths_df["ROLLINGDB_ID"].values:
    
    if os.path.isfile(os.path.join(vcf_dir, f"{sample_id}/pilon/{sample_id}.eff.vcf")):

        # kraken_class = pd.read_csv(os.path.join(vcf_dir, sample_id, "kraken/kraken_classifications"), sep="\t", header=None)
        kraken_report = pd.read_csv(os.path.join(vcf_dir, sample_id, "kraken/kraken_report"), sep="\t", header=None)
        
        # this is out of 100
        if "unclassified" in kraken_report[5].values:
            unclassified_percent = kraken_report.loc[kraken_report[5]=="unclassified"][0].values[0]
        else:
            unclassified_percent = 0
            
        if unclassified_percent > unclassified_thresh:
            high_unclassified_prop.append([sample_id, unclassified_percent])        
    else:
        raise ValueError(f"There is no VCF file for {sample_id}")
        
print(f"Removed {len(high_unclassified_prop)}/{len(paths_df)} validation samples with more than {unclassified_thresh}% unclassified reads")

high_unclassified_samples, _ = list(zip(*high_unclassified_prop))

# remove samples with high proportions of reads that don't align to MTBC
paths_df = paths_df.query("ROLLINGDB_ID not in @high_unclassified_samples\n")

    
if highConf:
    
    # Get all category 1 variants (don't use category 2 because it's the interim category
    who_high_conf = who_variants.loc[(who_variants["drug"] == drug) & (who_variants.confidence.str.contains("|".join(["1"])))].reset_index(drop=True)
    
    for _, row in who_high_conf.iterrows():
        if "," in row["genome_index"]:
            expanded_pos = row["genome_index"].split(",")
            
            for pos in expanded_pos:
                add_df = pd.DataFrame({"drug": drug, "genome_index": pos, "confidence": row["confidence"], "gene": row["gene"], "variant": row["variant"]}, index=[len(who_high_conf)])
                who_high_conf = pd.concat([who_high_conf, add_df])
            
    highConf_isolates = isolate_variants.query("mutation in @who_high_conf.mutation.values").Isolate.unique()
    
    prev_len = len(paths_df)
    
    # lower bound should be equal to or higher than the critical concentration
    paths_df = paths_df.loc[~((paths_df["ROLLINGDB_ID"].isin(highConf_isolates)) & (paths_df[f"{drug}_lower_bound"] < binary_thresh / 2))]
    print(f"Removed {prev_len - len(paths_df)} validation isolates with any of {len(who_high_conf)} category 1 mutations and MIC lower bound < {binary_thresh / 2}")
    
    paths_df.loc[paths_df["ROLLINGDB_ID"].isin(highConf_isolates), "WHO_Cat1_mutation"] = 1
    paths_df["WHO_Cat1_mutation"] = paths_df["WHO_Cat1_mutation"].fillna(0).astype(int)

else:
    paths_df["WHO_Cat1_mutation"] = 0

# create text file for later to get the paths of the VCF files in the validation dataset
paths_txt_file = pd.Series([f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF/{isolate}/pilon/{isolate}.eff.vcf" for isolate in paths_df["ROLLINGDB_ID"].values])
paths_txt_file.to_csv(os.path.join(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/validation_paths.txt"), sep="\t", header=None, index=False)

paths_df.to_csv(os.path.join(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/validation_data_for_model.csv"), index=False)
print(f"{len(paths_df)} validation samples")
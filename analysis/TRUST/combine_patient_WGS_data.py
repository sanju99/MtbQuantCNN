import pandas as pd
import numpy as np
import seaborn as sns
import glob, os, re, warnings, argparse
warnings.filterwarnings("ignore")


def combine_TRUST_patient_samples(df_trust_patient_data, WGS_metadata):

    print(f"{len(df_trust_patient_data.patient_num.unique())} patients in phenotypes dataframe")
    
    # each sample ID is in the format like 260-04, S0299-01, or S0271-01A. So use re to remove all alpha characters and keep only numbers.
    # first number is the patient number, second is the sample number
    # merge the patient number extracted from each sample with the patient numbers in df_trust_patient_data
    for i, row in WGS_metadata.iterrows():
        try:
            patient_num = int(re.findall(r'\d+', row["Original_ID"].split("-")[0])[0])
        # weird names like T-7, T-5, and P-T- 1
        except:
            patient_num = row["Original_ID"]

        try:
            sample_week = int(re.findall(r'\d+', row["Original_ID"].split("-")[1])[0])
        # weird names like T-7, T-5, and P-T- 1
        except:
            sample_week = row['Original_ID']
            
        WGS_metadata.loc[i, ['patient_num', 'sample_week']] = [patient_num, sample_week]
            
    # WGS_metadata["patient_num"] = [int(re.findall(r'\d+', val.split("-")[0])[0]) for val in WGS_metadata["Original_ID"].values]
    # WGS_metadata["sample_week"] = [int(re.findall(r'\d+', val.split("-")[1])[0]) for val in WGS_metadata["Original_ID"].values]

    # keep only samples for the patients in df_trust_patient_data
    WGS_metadata = WGS_metadata.merge(df_trust_patient_data, on="patient_num", how="inner")
    print(f"Found {WGS_metadata.patient_num.nunique()}/{df_trust_patient_data.patient_num.nunique()} patients in the WGS metadata files")
    
    # rename some columns that have spaces in them
    WGS_metadata = WGS_metadata.rename(columns={"Cov Any Mean": "Cov_Any_Mean",
                                                "Cov Unam Perc": "Cov_Unam_Perc",
                                                "Perc. Reads Mapped": "Perc_Reads_Mapped",
                                                "phylogenetic classification (Coll et al., 2014)": "Coll2014_Annotated"
                                       })

    return WGS_metadata




parser = argparse.ArgumentParser()

parser.add_argument("-i", "--patient_input", dest='patient_data_fName', type=str, required=True, help='Full path to a filename for the RedCap data from the TRUST study. This has had some data cleaning done on it')
parser.add_argument("-I", "--WGS_input", dest='WGS_data_fName', type=str, default="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_summary.csv", help='Full path to a filename with all WGS data and QC. Created by add_WGS_metadata_DB.py')
parser.add_argument("-o", "--output", dest='out_fName', type=str, required=True, help='Full path to a file name of the combined WGS and patient data results')
parser.add_argument("-d", "--dir", dest='WGS_report_directory', default="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/WGS_metadata_reports", type=str, help='Directory where the Excel files of WGS metadata reports are')

cmd_line_args = parser.parse_args()
patient_data_fName = cmd_line_args.patient_data_fName
WGS_data_fName = cmd_line_args.WGS_data_fName
out_fName = cmd_line_args.out_fName
WGS_report_directory = cmd_line_args.WGS_report_directory

df_trust_patient_data = pd.read_csv(patient_data_fName, low_memory=False)

# get the patient number to match WGS IDs and pids
df_trust_patient_data["patient_num"] = [int(patient_id.replace("T0", "")) for patient_id in df_trust_patient_data["pid"].values]

# get all Excel files from this directory
trust_report_fNames = glob.glob(f"{WGS_report_directory}/*.xlsx")
print(f"{len(trust_report_fNames)} WGS metadata Excel files")

df_trust_WGS_metadata = []

# sort chronologically because below, we preferentially keep the later one
for fName in np.sort(trust_report_fNames)[::-1]:

    # read in single Excel file
    df = pd.read_excel(fName, sheet_name=None)

    # remove spaces from column name to make querying easier. Also there could be NaN rows if there are additional empty rows in the Excel sheet, so drop them
    df = df['summary'].rename(columns={'original ID': 'Original_ID'}).dropna(axis=0, how='all')

    # append to list for concatenation
    df_trust_WGS_metadata.append(df)

# the Excel files above are running totals, so the most recent file has data that is also in the older files. So drop duplicates, keeping the most recent (last) one
df_trust_WGS_metadata = pd.concat(df_trust_WGS_metadata).drop_duplicates('SampleID', keep='last')#.query("status!='failed'")

# combine patient and sample IDs (pid = patient, Original_ID and SampleID = WGS)
df_trust_combined = combine_TRUST_patient_samples(df_trust_patient_data, df_trust_WGS_metadata.reset_index(drop=True))

# combine with additional WGS QC data. Drop samples that filed QC (no VCF, so they have nothing in the Lineage column)
df_geno = pd.read_csv(WGS_data_fName).dropna(subset='Lineage')

df_trust_combined = df_trust_combined.merge(df_geno[['SampleID', 'F2', 'Coll2014', 'Freschi2020', 'Lineage']], on='SampleID', how='left').reset_index(drop=True)

# Remove columns that are NaN everywhere to clean the dataframe
df_trust_combined = df_trust_combined.dropna(how='all', axis=1)

# put the IDs at the front, then save
print(f"{df_trust_combined.pid.nunique()} patients with sequencing data")
df_trust_combined.set_index(['pid', 'Original_ID', 'SampleID']).to_csv(out_fName)
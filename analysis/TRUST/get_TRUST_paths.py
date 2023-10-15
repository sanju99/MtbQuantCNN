import pandas as pd
import glob, os, sys

trust_report = pd.read_excel("/n/data1/hms/dbmi/farhat/rollingDB/TRUST/2023-01-24_report_Farhat.v09.xlsx", sheet_name=None)
df_trust_report = trust_report["summary"]

trust_paths = []

for i, row in df_trust_report.iterrows():

    sample_id = row['SampleID']

    fName1 = f"/n/data1/hms/dbmi/farhat/rollingDB/TRUST/220617.TRUST.Set1.Illumina.Output/{sample_id}/IlluminaWGS/Pilon_IlluminaPE_AlignedTo_H37rv_minMQ_1_minDP_5_Fix_All_Breaks/{sample_id}.IllPE.H37rv.vcf.gz"
    fName2 = f"/n/data1/hms/dbmi/farhat/rollingDB/TRUST/221216.TRUST.Illumina.Batch2.Output/{sample_id}/IlluminaWGS/Pilon_IlluminaPE_AlignedTo_H37rv_minMQ_1_minDP_5_Fix_All_Breaks/{sample_id}.IllPE.H37rv.vcf.gz"
    
    if row['comment'] == '-':
        if os.path.isfile(fName1):
            trust_paths.append(fName1)
        elif os.path.isfile(fName2):
            trust_paths.append(fName2)
        else:
            print(sample_id)

fNames_to_subset_annotate = []

for fName in trust_paths:

    sample_id = os.path.basename(fName).split(".")[0]

    if not os.path.isfile(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/VCF/{sample_id}.vcf"):
        fNames_to_subset_annotate.append(fName)

print(f"Need to subset and annotate {len(fNames_to_subset_annotate)} TRUST samples")

pd.Series(fNames_to_subset_annotate).to_csv("/home/sak0914/MtbQuantCNN/analysis/TRUST/data_paths.txt", sep="\t", index=False, header=None)
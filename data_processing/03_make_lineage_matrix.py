import numpy as np
import pandas as pd
import vcf, glob, os, sys

_, scheme = sys.argv


# for i in `ls |grep ".vcf"`; do
#     fast-lineage-caller $i --noheader --pass >> /home/sak0914/MtbQuantCNN/analysis/TRUST_lineages.tsv
# done

# TRUST_lineages = pd.read_csv("TRUST_lineages.tsv", sep="\t", header=None)
# TRUST_lineages.columns = ["Isolate", "Coll2014", "Freschi2020", "Lipworth2019", "Shitikov2017", "Stucki2016"]
# TRUST_lineages["Coll2014"] = TRUST_lineages["Coll2014"].str.replace("lineage", "")

# first read in the updated lineages.tsv file from 02_training_data_vcf_processing.sh and convert to .csv file
lineages_df_save = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv", sep="\t", header=None)
lineages_df_save.columns = ['ROLLINGDB_ID', 'Coll2014', 'Freschi2020', 'Lipworth2019', 'Shitikov2017', 'Stucki2016']

# remove the lineage prefix from the Coll 2014 scheme, then add a column for the primary lineage
lineages_df_save["Coll2014"] = [val.replace("lineage", "") for val in lineages_df_save["Coll2014"].values]
lineages_df_save["Lineage"] = [val[0] if val[0].isnumeric() else val for val in lineages_df_save["Coll2014"].values]

# many of the ROLLINGDB ids have .eff in them because the VCF files are .eff.vcf, so remove that here
lineages_df_save["ROLLINGDB_ID"] = [val.split(".")[0] for val in lineages_df_save["ROLLINGDB_ID"].values]

lineages_df_save.to_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.csv", index=False)
del lineages_df_save

if scheme.upper() == "FRESCHI":
    lineage_SNPs = pd.read_csv("/home/sak0914/who-analysis/data/freschi2020_SNP_scheme.tsv", sep="\t")
    suffix = "Freschi2020"
elif scheme.upper() == "COLL":
    lineage_SNPs = pd.read_csv("/home/sak0914/who-analysis/data/coll2014_SNP_scheme.tsv", sep="\t")
    suffix = "Coll2014"


vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"
mic_ml_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/VCF"

# skip isolates for which the SNP vectors were already constructed, otherwise this is very inefficient
old_lineages_mat = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_{suffix}.csv", index_col=[0])
old_lineages_mat.columns = old_lineages_mat.columns.astype(int)
isolates_already_finished = old_lineages_mat.index.values
vcf_files = []

for isolate in os.listdir(vcf_dir):
    if isolate not in isolates_already_finished:
        assert os.path.isfile(os.path.join(vcf_dir, isolate, "pilon", f"{isolate}.vcf"))
        vcf_files.append(os.path.join(vcf_dir, isolate, "pilon", f"{isolate}.vcf"))

for isolate in os.listdir(mic_ml_dir):
    if isolate not in isolates_already_finished:
        assert os.path.isfile(os.path.join(mic_ml_dir, isolate, "pilon", f"{isolate}.vcf"))
        vcf_files.append(os.path.join(mic_ml_dir, isolate, "pilon", f"{isolate}.vcf"))
    
print(f"Getting {scheme} SNPs for {len(vcf_files)} files")

ref, alt = list(zip(*lineage_SNPs["allele_change"].str.split("/")))

lineage_SNPs["REF"] = ref
lineage_SNPs["ALT"] = alt

lineages_df = pd.DataFrame(columns=["ROLLINGDB_ID", "POS"])
samples_lst = []

for i, fName in enumerate(vcf_files):
    
    sample_id = os.path.basename(fName).split(".")[0]
    samples_lst.append(sample_id)
    assert os.path.isfile(fName)

    vcf_file = vcf.Reader(filename=fName)

    for record in vcf_file:

        # if FILTER == PASS, the FILTER field is an empty list, so the length is 0
        # also required AF >= 0.75 to exclude mixed samples
        if record.POS in lineage_SNPs["position"].values and len(record.FILTER) == 0:
            
            # for structural variants, there is no AF key in the INFO field
            if "AF" in record.INFO.keys():
                
                if record.INFO["AF"][0] >= 0.75:

                    ref, alt = lineage_SNPs.loc[lineage_SNPs["position"]==record.POS, ["REF", "ALT"]].values[0]

            if record.REF == ref:
                if type(record.ALT) == list:
                    if record.ALT[0] == alt:
                        # print(record)
                        # pos_lst.append(record.POS)
                        lineages_df = pd.concat([lineages_df, pd.DataFrame({"ROLLINGDB_ID": sample_id,
                                                                            "POS": record.POS,
                                                                           }, index=[0])], 
                                                axis=0)
                else:
                    if record.ALT == alt:
                        # print(record)
                        # pos_lst.append(record.POS)
                        lineages_df = pd.concat([lineages_df, pd.DataFrame({"ROLLINGDB_ID": sample_id,
                                                                            "POS": record.POS,
                                                                           }, index=[0])], 
                                                axis=0)
                
    if i % 1000 == 0:
        print(i)
        
        
assert len(vcf_files) == len(samples_lst)
# lineages_df.to_csv(f"/home/sak0914/MtbQuantCNN/lineages_df_{suffix}.csv", index=False)

lineages_df["Count"] = 1
lineages_mat = lineages_df.pivot(index="ROLLINGDB_ID", columns="POS", values="Count").fillna(0).astype(int)
print(lineages_mat.shape)

# add rows for samples that do not have any SNPs
missing_samples = list(set(samples_lst) - set(lineages_mat.index.values))
zero_sample_df = pd.DataFrame(0, index=missing_samples, columns=lineages_mat.columns)
lineages_mat = pd.concat([lineages_mat, zero_sample_df], axis=0)
print(lineages_mat.shape)

# add columns for SNPs that were not found at high confidence in any of the samples
missing_pos = list(set(lineage_SNPs["position"].values) - set(lineages_mat.columns))
zero_pos_df = pd.DataFrame(0, index=lineages_mat.index.values, columns=missing_pos)
lineages_mat = pd.concat([lineages_mat, zero_pos_df], axis=1)
print(lineages_mat.shape)
assert len(np.unique(lineages_mat.values)) == 2

if len(set(old_lineages_mat.columns).symmetric_difference(lineages_mat.columns)) > 0:
    raise ValueError(f"{len(set(old_lineages_mat.columns).symmetric_difference(lineages_mat.columns))} columns do not match between the old and new lineage matrices")

lineages_mat = pd.concat([old_lineages_mat, lineages_mat], axis=0)

# check that there are no NaNs anywhere
assert lineages_mat.isnull().values.sum() == 0
lineages_mat.to_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_{suffix}.csv")
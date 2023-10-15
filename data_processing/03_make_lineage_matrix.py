import numpy as np
import pandas as pd
import vcf, glob, os, sys

_, scheme = sys.argv

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

lineage_SNPs[["REF", "ALT"]] = lineage_SNPs["allele_change"].str.split("/", expand=True)

vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"
trust_vcf_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/VCF"

lineage_mat_fName = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_{suffix}.csv"

# skip isolates for which the SNP vectors were already constructed, otherwise this is very inefficient
if os.path.isfile(lineage_mat_fName):
    old_lineages_mat = pd.read_csv(lineage_mat_fName, index_col=[0])
    old_lineages_mat.columns = old_lineages_mat.columns.astype(int)
    isolates_already_finished = old_lineages_mat.index.values
else:
    isolates_already_finished = []
    
vcf_files = []

for isolate in os.listdir(vcf_dir):
    if isolate not in isolates_already_finished:
        assert os.path.isfile(os.path.join(vcf_dir, isolate, "pilon", f"{isolate}.eff.vcf"))
        vcf_files.append(os.path.join(vcf_dir, isolate, "pilon", f"{isolate}.eff.vcf"))

for fName in glob.glob(f"{trust_vcf_dir}/*.eff.vcf"):
    
    isolate = os.path.basename(fName).split(".")[0]
    if isolate not in isolates_already_finished:
        vcf_files.append(fName)
    
print(f"Getting {scheme} SNPs for {len(vcf_files)} files")

# def get_AF_for_lineage_SNP(record):

#     # not an imprecise variant and passes the QC filters
#     if "AF" in record.INFO.keys():
#         if record.INFO["BQ"] >= 20 and record.INFO["MQ"] >= 1 and record.INFO["DP"] >= 5:
#             return record.INFO['AF'][0]

#     # anything else, we don't know if it exists with high certainty, so consider it REF. Another type of encoding doesn't make a lot of sense
#     # imprecise variant, where there may be an indel or imprecise variant overlapping the lineage SNP position
#     # return -1 because the values are meant to be AF: 0 = REF, > 0.25 = ALT
#     return 0

def get_lineage_SNP_presence_absence(record):

    # not an imprecise variant and passes the QC filters
    if record.INFO["BQ"] >= 20 and record.INFO["MQ"] >= 1:
        return 1

    # anything else, we don't know if it exists with high certainty, so consider it REF. Another type of encoding doesn't make a lot of sense
    # imprecise variant, where there may be an indel or imprecise variant overlapping the lineage SNP position
    return 0

def get_single_sample_lineage_df(vcf_fName):

    sample_id = os.path.basename(vcf_fName).split(".")[0]
    # samples_lst.append(sample_id)
    assert os.path.isfile(vcf_fName)

    # dictionary to keep track of non-REF positions and their AFs
    lineage_SNPs_dict = {}

    vcf_file = vcf.Reader(filename=vcf_fName)

    for record in vcf_file:

        # position must have the correct REF (trivial) and ALT for the lineage SNP
        if record.POS in lineage_SNPs["position"].values:

            ref, alt = lineage_SNPs.loc[lineage_SNPs["position"]==record.POS, ["REF", "ALT"]].values[0]
            assert record.REF == ref
            
            if len(record.ALT) == 1 and record.ALT[0] == alt:
                lineage_SNPs_dict[record.POS] = get_lineage_SNP_presence_absence(record)
            else:
                print(sample_id, alt, record)

    single_sample_SNP_df = pd.DataFrame(lineage_SNPs_dict, index=[0]).T.reset_index()
    single_sample_SNP_df.columns = ["POS", "AF"]
    single_sample_SNP_df = pd.concat([single_sample_SNP_df, lineage_SNPs.query("position not in @single_sample_SNP_df.POS")[["position"]].rename(columns={"position": "POS"})]).fillna(0)
    single_sample_SNP_df['Isolate'] = sample_id

    return single_sample_SNP_df
        

lineages_df = pd.DataFrame(columns=["Isolate", "POS", "AF"])

for i, fName in enumerate(vcf_files):

    single_sample_lineage_SNP_df = get_single_sample_lineage_df(fName)
    
    assert len(single_sample_lineage_SNP_df['POS'].unique()) == len(single_sample_lineage_SNP_df)
    assert len(set(single_sample_lineage_SNP_df['POS']).symmetric_difference(lineage_SNPs["position"])) == 0

    lineages_df = pd.concat([lineages_df, single_sample_lineage_SNP_df], axis=0)

    if i % 500 == 0:
        print(i)

lineages_mat = lineages_df.pivot(index="Isolate", columns="POS", values="AF")#.fillna(0).astype(int)

# check that there are no NaNs anywhere
assert lineages_mat.isnull().values.sum() == 0

print(f"Lineage matrix shape: {lineages_mat.shape}")

# # add rows for samples that do not have any SNPs
# missing_samples = list(set(samples_lst) - set(lineages_mat.index.values))
# zero_sample_df = pd.DataFrame(0, index=missing_samples, columns=lineages_mat.columns)
# lineages_mat = pd.concat([lineages_mat, zero_sample_df], axis=0)
# print(lineages_mat.shape)

# # add columns for SNPs that were not found at high confidence in any of the samples
# missing_pos = list(set(lineage_SNPs["position"].values) - set(lineages_mat.columns))
# zero_pos_df = pd.DataFrame(0, index=lineages_mat.index.values, columns=missing_pos)
# lineages_mat = pd.concat([lineages_mat, zero_pos_df], axis=1)
# print(lineages_mat.shape)
# assert len(np.unique(lineages_mat.values)) == 2

if os.path.isfile(lineage_mat_fName):
    
    if len(set(old_lineages_mat.columns).symmetric_difference(lineages_mat.columns)) > 0:
        raise ValueError(f"{len(set(old_lineages_mat.columns).symmetric_difference(lineages_mat.columns))} columns do not match between the old and new lineage matrices")

    lineages_mat = pd.concat([old_lineages_mat, lineages_mat], axis=0)

# # check that there are no NaNs anywhere
# assert lineages_mat.isnull().values.sum() == 0
lineages_mat.to_csv(lineage_mat_fName)
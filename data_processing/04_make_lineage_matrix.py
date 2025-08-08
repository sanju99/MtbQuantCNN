import numpy as np
import pandas as pd
import vcf, glob, os, sys

_, scheme = sys.argv

# all isolates passing genotypic quality control and with MICs. Keep all the TRUST isolates. Don't add the mixed non-TRUST samples though
df = pd.read_csv("./samples_pass_geno_QC.csv").query("F2 <= 0.1 | DB_OF_ORIGIN == 'TRUST'").reset_index(drop=True)

print(df.DB_OF_ORIGIN.value_counts())

if scheme.upper() == "FRESCHI":
    lineage_SNPs = pd.read_csv("./data_utils/freschi2020_SNP_scheme.tsv", sep="\t")
    suffix = "Freschi2020"
elif scheme.upper() == "COLL":
    lineage_SNPs = pd.read_csv("./data_utils/coll2014_SNP_scheme.tsv", sep="\t")
    suffix = "Coll2014"
else:
    raise ValueError("Please pass in either coll or freschi as the argument")

lineage_SNPs[["REF", "ALT"]] = lineage_SNPs["allele_change"].str.split("/", expand=True)

lineage_mat_fName = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_{suffix}.csv"
samples_all = list(df['ROLLINGDB_ID'].values)

# skip isolates for which the SNP vectors were already constructed, otherwise this is very inefficient
if os.path.isfile(lineage_mat_fName):
    old_lineages_mat = pd.read_csv(lineage_mat_fName, index_col=[0])
    old_lineages_mat.columns = old_lineages_mat.columns.astype(int)
    isolates_already_finished = old_lineages_mat.index.values
else:
    isolates_already_finished = []

samples_lst = list(set(samples_all) - set(isolates_already_finished))

vcf_files = []

for sample in samples_lst:

    if os.path.isfile(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF/{sample}.vcf"):
        vcf_files.append(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF/{sample}.vcf")

    elif os.path.isfile(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/VCF/{sample}.vcf"):
        vcf_files.append(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/VCF/{sample}.vcf")

    else:
        raise ValueError(f"No VCF file found for {sample}")
    
del samples_all

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

    # code below already checks that it's a SNP, this is to check if it passes the QC filters
    if len(record.FILTER) == 0 and record.INFO["BQ"] >= 20 and record.INFO["MQ"] >= 30 and record.QUAL >= 10 and record.INFO['DP'] >= 5 and float(record.INFO['AF'][0]) > 0.75:
        return 1

    # anything else, we don't know if it exists with high certainty, so consider it REF. Another type of encoding doesn't make a lot of sense, like 0.5 for uncertain variants
    # these should be easy to sequence areas though
    # imprecise variant, where there may be an indel or imprecise variant overlapping the lineage SNP position
    return 0

def get_single_sample_lineage_df(vcf_fName):

    sample_id = os.path.basename(vcf_fName).replace('.vcf', '').replace('_variants', '')
    
    if not os.path.isfile(vcf_fName):
        raise ValueError(vcf_fName)

    # dictionary to keep track of non-REF positions and their AFs
    lineage_SNPs_dict = {}

    vcf_file = vcf.Reader(filename=vcf_fName)

    for record in vcf_file:

        ref_allele = str(record.REF)
        alt_allele = "".join(np.array(record.ALT).astype(str))

        # position must have the correct REF (trivial) and ALT for the lineage SNP
        if record.POS in lineage_SNPs["position"].values:

            ref, alt = lineage_SNPs.loc[lineage_SNPs["position"]==record.POS, ["REF", "ALT"]].values[0]
            assert record.REF == ref
            
            if len(alt_allele) == 1 and alt_allele == alt:
                lineage_SNPs_dict[record.POS] = get_lineage_SNP_presence_absence(record)
            else:
                # different allele, not the lineage-defining one at this site
                print(sample_id, alt, record)

    single_sample_SNP_df = pd.DataFrame(lineage_SNPs_dict, index=[0]).T.reset_index()
    single_sample_SNP_df.columns = ["POS", "AF"]
    single_sample_SNP_df = pd.concat([single_sample_SNP_df, lineage_SNPs.query("position not in @single_sample_SNP_df.POS")[["position"]].rename(columns={"position": "POS"})]).fillna(0)
    single_sample_SNP_df['Isolate'] = sample_id

    return single_sample_SNP_df
        

lineages_df = [] #pd.DataFrame(columns=["Isolate", "POS", "AF"])

for i, fName in enumerate(vcf_files):

    single_sample_lineage_SNP_df = get_single_sample_lineage_df(fName)
    
    assert len(single_sample_lineage_SNP_df['POS'].unique()) == len(single_sample_lineage_SNP_df)
    assert len(set(single_sample_lineage_SNP_df['POS']).symmetric_difference(lineage_SNPs["position"])) == 0

    # lineages_df = pd.concat([lineages_df, single_sample_lineage_SNP_df], axis=0)
    lineages_df.append(single_sample_lineage_SNP_df)

    if i % 1000 == 0:
        print(i)

lineages_mat = pd.concat(lineages_df).pivot(index="Isolate", columns="POS", values="AF")

# check that there are no NaNs anywhere
assert lineages_mat.isnull().values.sum() == 0

print(f"Added {len(lineages_mat)} lineages to the matrix")

if os.path.isfile(lineage_mat_fName):
    
    if len(set(old_lineages_mat.columns).symmetric_difference(lineages_mat.columns)) > 0:
        raise ValueError(f"{len(set(old_lineages_mat.columns).symmetric_difference(lineages_mat.columns))} columns do not match between the old and new lineage matrices")

    lineages_mat = pd.concat([old_lineages_mat, lineages_mat], axis=0)

# add MT_H37Rv
lineages_mat.loc['MT_H37Rv', :] = 0

# check that there are no NaNs anywhere
assert lineages_mat.isnull().values.sum() == 0
lineages_mat.to_csv(lineage_mat_fName)
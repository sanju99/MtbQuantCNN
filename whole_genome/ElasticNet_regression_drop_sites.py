import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle

from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Bio import SeqIO, Seq

# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
sys.path.append("utils")
from inSilicoMut_utils import *

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_genes = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_genes_v4.csv"))
h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

data_path = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"


_, drug, num_PCs = sys.argv

num_PCs = int(num_PCs)
df = pd.read_csv(os.path.join(data_path, drug, "isolate_variants_FULL_fixed.tsv"), sep="\t")

# # keep only those that would be used to train the CNN
df_phenos = pd.read_csv(os.path.join(data_path, drug, "data_for_model.csv"), usecols=["ROLLINGDB_ID", "category", f"{drug}_midpoint"]).set_index("ROLLINGDB_ID")
# df = df.query("Isolate in @df_phenos.index.values")

# # Zero out the low allele fractions because they are unreliable below 0.25
# df.loc[(df["QC"] > -1) & (df["QC"] < 0.25), "QC"] = 0
# assert df.query("QC > 0").QC.min() >= 0.25

# # separate SNPs and indels (to code them differently in the model)
# df_SNP = df.query("REF_len==1 & ALT_len==1").reset_index(drop=True)
# df_indel = df.query("~(REF_len==1 & ALT_len==1)").reset_index(drop=True)

# # add additional Gene column (the one from SNPEff is GENE)
# df_indel["Gene"] = df_indel["POS"].map(h37Rv_coords_dict)

# print(df_SNP.shape, df_indel.shape)

# drugs_loci = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/drugs_loci.csv")
# drugs_loci = drugs_loci.query("~Drugs.str.contains(@drug)")

# # add 1 to the start because it's 0-indexed
# drugs_loci["Start"] += 1
# assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

# # get all positions in resistance loci
# drug_res_sites = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
# drug_res_sites = list(itertools.chain.from_iterable(drug_res_sites))

# df_SNP = df_SNP.query("POS not in @drug_res_sites")
# del drug_res_sites

# remove_genes = ['gyrA',
#  'gyrB',
#  'eis',
#  'oxyR',
#  'ahpC',
#  'rrs',
#  'rrl',
#  'tlyA',
#  'acpM',
#  'kasA',
#  'embC',
#  'embA',
#  'embB',
#  'aftB',
#  'ubiA',
#  'ethA',
#  'ethR',
#  'rpoB',
#  'rpoC', 
#  'katG',
#  'fabG1',
#  'inhA',
#  'rpsL',
#  'gid',
#  'thyA']

# df_indel = df_indel.query("GENE not in @remove_genes")
# print(df_SNP.shape, df_indel.shape)

# def get_high_qual_variants(df, is_snp=True):

#     if is_snp:
#         group_col = "mutation"
#     else:
#         group_col = "Gene"
        
#     # get the counts of each mutation by quality metric in the dataset
#     mutation_QC_summary = pd.DataFrame(df.groupby(group_col)["QC"].value_counts())
    
#     # get the prevalence of each mutation (counts) in the dataset. Choose a random column to take the counts over (POS)
#     # then merge with the dataframe above. The df now has two columns: count = count by quality, total = total number of times the mutations is found in the dataset
#     mutation_QC_summary = mutation_QC_summary.merge(pd.DataFrame(df.groupby(group_col)["POS"].count()), left_index=True, right_index=True).reset_index().rename(columns={"POS": "total"})
    
#     # then get the low-quality mutations: more than 25% of the instance of the mutation are low-quality
#     low_qual_mutations = mutation_QC_summary.query("QC==-1").loc[(mutation_QC_summary.query("QC==-1")["count"] / mutation_QC_summary.query("QC==-1")["total"] > 0.25)][group_col].values
    
#     print(f"Removed {len(low_qual_mutations)}/{len(df[group_col].unique())} features with more than 25% low-quality occurrences")
#     df = df.query("mutation not in @low_qual_mutations")
    
#     # there are a few instances of 2 SNPs occurring on the same codon, but they individually have the same effect as the combination of the 2 SNPs
#     # they have different allele frequencies too, so they don't get removed in the drop_duplicates function that includes "QC"
#     # drop those here
#     return df.sort_values("QC", ascending=False).drop_duplicates(["Isolate", group_col], keep="first").reset_index(drop=True)



# df_SNP_high_qual = get_high_qual_variants(df_SNP, is_snp=True)
# df_indel_high_qual = get_high_qual_variants(df_indel, is_snp=False)


# ########### STEP 3: get synonymous variants to compute the genetic relatedness matrix
# df_syn = df_SNP_high_qual.query("EFFECT in ['synonymous_variant', 'intergenic_region', 'intragenic_variant', 'upstream_gene_variant', 'downstream_gene_variant']")

# # remove variants from drug resistance regions and homoplasic sites
# homoplasy_sites = pd.read_excel("/home/sak0914/who-analysis/data/Vargas_homoplasy.xlsx")

# if len(homoplasy_sites) == 1:
#     homoplasy_sites = homoplasy_sites[list(homoplasy_sites.keys())[0]]

# homoplasy_sites = homoplasy_sites["H37Rv Position"].values.astype(int)


# drugs_loci = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/drugs_loci.csv")

# # add 1 to the start because it's 0-indexed
# drugs_loci["Start"] += 1
# assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

# # get all positions in resistance loci
# drug_res_sites = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
# drug_res_sites = list(itertools.chain.from_iterable(drug_res_sites))

# prev_pos = len(df_syn.POS.unique())
# df_syn = df_syn.query("POS not in @homoplasy_sites & POS not in @drug_res_sites")
# print(f"Dropped {prev_pos - len(df_syn.POS.unique())} positions that are homoplasic or in drug-resistance regions")

# matrix_for_GRM = df_syn.pivot(index="Isolate", columns="mutation", values="QC").fillna(0).replace(-1, np.nan)
# prev_cols = matrix_for_GRM.shape[1]

# matrix_for_GRM = matrix_for_GRM.dropna(axis=1)
# print(f"Dropped {prev_cols-matrix_for_GRM.shape[1]} SNPs from the GRM with any missingness")
# assert sum(matrix_for_GRM.index.values != df_phenos.index.values) == 0

# scaler = StandardScaler()
# grm = np.cov(scaler.fit_transform(matrix_for_GRM.values))

# pca = PCA(n_components=num_PCs)
# pca.fit(scaler.fit_transform(grm))

# # save eigenvectors to use later as well
# pca_df = pd.DataFrame(pca.components_.T)
# pca_df.columns = [f"PC{num}" for num in np.arange(num_PCs)]
# pca_df.index = df_phenos.index.values
# pca_df.to_csv(os.path.join(data_path, drug, "whole_genome/PCA_eigenvec_df.csv"), index=False)

# def pivot_df_to_matrix(df_high_qual, is_snp=True, nonsyn_only=True):

#     if is_snp:
#         feat_col = "mutation"

#         if nonsyn_only:
#             df_high_qual = df_high_qual.query("EFFECT not in ['synonymous_variant', 'intergenic_region', 'intragenic_variant', 'upstream_gene_variant', 'downstream_gene_variant']")
#     else:
#         feat_col = "Gene"
        
#     # now, the NaNs are low-quality mutations
#     matrix = df_high_qual.pivot(index="Isolate", columns=feat_col, values="QC").fillna(0).replace(-1, np.nan)

#     # remove any features with NaNs (not isolates)
#     drop_cols = list(set(matrix.columns) - set(matrix.dropna(axis=1)))
#     print(f"Dropped {len(drop_cols)}/{matrix.shape[1]} features with any missingness")
#     matrix = matrix.dropna(axis=1)

#     # code for dropping only features with missingness above 25% of isolates
#     # matrix.dropna(axis=0, thresh=0.25*matrix.shape[1])
    
#     assert sum(pd.isnull(np.unique(matrix.values))) == 0
#     return matrix


# matrix_SNP = pivot_df_to_matrix(df_SNP_high_qual, is_snp=True, nonsyn_only=True)
# matrix_indel = pivot_df_to_matrix(df_indel_high_qual, is_snp=False, nonsyn_only=True)

# regression_matrix = pd.concat([matrix_SNP, matrix_indel], axis=1)
# print(f"Regression matrix shape: {regression_matrix.shape}")
# regression_matrix.to_pickle(os.path.join(data_path, drug, "whole_genome/regression_matrix_no_drug_res.pkl"))

regression_matrix = pd.read_pickle(os.path.join(data_path, drug, "whole_genome/regression_matrix_no_drug_res.pkl"))
pca_df = pd.read_csv(os.path.join(data_path, drug, "whole_genome/PCA_eigenvec_df.csv"), index_col=0)

scaler = StandardScaler()

X = scaler.fit_transform(pd.concat([regression_matrix.loc[df_phenos.index.values], pca_df.loc[df_phenos.index.values]], axis=1).values)
y = np.log2(df_phenos[f"{drug}_midpoint"].values)

# X_train = scaler.fit_transform(pd.concat([regression_matrix.loc[df_phenos.query("category=='original_train_set'").index.values],
#                                           pca_df.loc[df_phenos.query("category=='original_train_set'").index.values]
#                                          ], axis=1).values
#                               )

# X_test = scaler.fit_transform(pd.concat([regression_matrix.loc[df_phenos.query("category=='original_test_set'").index.values],
#                                          pca_df.loc[df_phenos.query("category=='original_test_set'").index.values]
#                                         ], axis=1).values
#                               )

# y_train = np.log2(df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values)
# y_test = np.log2(df_phenos.query("category=='original_test_set'")[f"{drug}_midpoint"].values)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X.shape, y.shape)

model = ElasticNetCV(l1_ratio=np.linspace(0, 1, 11),
                     alphas=np.logspace(-3, 3, 7),
                     cv=5,
                     fit_intercept=True, 
                     max_iter=10000,
                     verbose=2,
                     n_jobs=4
                    )

model.fit(X, y)
pickle.dump(model, open(os.path.join(data_path, drug, "whole_genome/full_feature_select.sav"), 'wb'))
print(model.l1_ratio_, model.alpha_)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle

from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Bio import SeqIO, Seq

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
sys.path.append("utils")
from inSilicoMut_utils import *

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_genes = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_genes_v4.csv"))
h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

exclude_regions = pd.read_csv("whole_genome/exclude_regions.txt", sep="\t", header=None)
exclude_regions.columns = ["CHROM", "START", "END"]

exclude_pos = []

for i, row in exclude_regions.iterrows():
    exclude_pos += list(np.arange(row["START"], row["END"]+1))


_, drug, num_PCs = sys.argv

out_dir = os.path.join(data_dir, drug, "whole_genome")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

num_PCs = int(num_PCs)
df_phenos = pd.read_csv(os.path.join(data_dir, drug, "data_for_model.csv"), usecols=["ROLLINGDB_ID", "category", f"{drug}_midpoint"]).set_index("ROLLINGDB_ID")

isolate_variants_df = pd.read_csv(os.path.join(data_dir, drug, "isolate_variants.csv.gz"), compression="gzip", dtype={"Isolate": str})
print(f"{isolate_variants_df.Isolate.nunique()} isolates")

# keep only those for training. The MIC-ML data for validation will be worked on later
df = isolate_variants_df.query("POS not in @exclude_pos and Isolate in @df_phenos.index.values").reset_index(drop=True)
print(f"{df.Isolate.nunique()} training isolates after removing {len(exclude_pos)} low mappability regions")
del isolate_variants_df

df["REF_len"] = [len(val) for val in df["REF"].values]
df["ALT_len"] = [len(val) for val in df["ALT"].values]

# Zero out the low allele fractions because they are unreliable below 0.25
# the QC column will be used in the final matrix. Missing variables will be NaN, then we drop those isolates
# can consider introducing an AF thresh in the future, where AFs >= thresh will be AF, and AFs < thresh will be 0
# this only makes sense for SNPs because there is not always an AF for indels (i.e. if it's a structural variant)
df["AF"] = df["AF"].replace(".", 0.76).astype(float)

# not sure why this happens
df.loc[df["AF"] > 1, "AF"] = 1

df.loc[(df["FILTER"].isin(['PASS', 'Amb'])) & (df["AF"] < 0.25), "QC"] = 0

# include support for heteroresistance -- encode intermediate AFs with the AFs themselves (instead of i.e. dropping them)
df.loc[(df["FILTER"].isin(['PASS', 'Amb'])) & (df["AF"] >= 0.25), "QC"] = df.loc[(df["FILTER"].isin(['PASS', 'Amb'])) & (df["AF"] >= 0.25)]["AF"]

# encode NaNs with -1 for now because during the pivot, 0 values (absent features) will be replace with NaN
df.loc[~df["FILTER"].isin(['PASS', 'Amb']), "QC"] = -1

# PRESENT = 1, ABSENT = 0. Only intermediate values encoded with their AFs
df.loc[df["QC"] > 0.75, "QC"] = 1

# separate SNPs and indels (to code them differently in the model)
df_SNP = df.query("REF_len==1 & ALT_len==1").reset_index(drop=True)
df_indel = df.query("~(REF_len==1 & ALT_len==1)").reset_index(drop=True)
del df

# add additional Gene column (the one from SNPEff is GENE)
df_indel["Gene"] = df_indel["POS"].map(h37Rv_coords_dict)

# add mutation column for SNPs so that each mutation is POS_REF_ALT
df_SNP["mutation"] = df_SNP["POS"].astype(str) + "_" + df_SNP["REF"] + "_" + df_SNP["ALT"]

indel_matrix = pd.DataFrame(df_indel.groupby(["Isolate", "Gene"])["QC"].max()).reset_index()
indel_matrix = indel_matrix.pivot(index="Isolate", columns="Gene", values="QC").fillna(0).replace(-1, np.nan)
del df_indel

if df_SNP.shape[0] != df_SNP.drop_duplicates(["Isolate", "mutation", "QC"]).shape[0]:
    raise ValueError("There are duplicates in the SNP dataframe!")

snp_matrix = df_SNP.pivot(index="Isolate", columns="mutation", values="QC").fillna(0).replace(-1, np.nan)
print(snp_matrix.shape, indel_matrix.shape)

snp_matrix.to_pickle(os.path.join(out_dir, "SNP_matrix.pkl"))
indel_matrix.to_pickle(os.path.join(out_dir, "indel_matrix.pkl"))


########### STEP 3: get synonymous variants to compute the genetic relatedness matrix


# df_syn = df_SNP.query("EFFECT in ['synonymous_variant', 'intergenic_region', 'intragenic_variant', 'upstream_gene_variant', 'downstream_gene_variant']")

# # remove variants from drug resistance regions and homoplasic sites
# homoplasy_sites = pd.read_excel("/home/sak0914/who-analysis/PCA/Vargas_homoplasy.xlsx")

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
# # assert sum(matrix_for_GRM.index.values != df_phenos.index.values) == 0

# scaler = StandardScaler()
# grm = np.cov(scaler.fit_transform(matrix_for_GRM.values))

# pca = PCA(n_components=num_PCs)
# pca.fit(scaler.fit_transform(grm))

# # save eigenvectors to use later as well
# pca_df = pd.DataFrame(pca.components_.T)
# pca_df.columns = [f"PC{num}" for num in np.arange(num_PCs)]
# pca_df.index = df_phenos.index.values
# pca_df.to_csv(os.path.join(out_dir, "PCA_eigenvec_df.csv"))

# # regression_matrix = pd.concat([matrix_SNP, matrix_indel], axis=1)
# # print(f"Regression matrix shape: {regression_matrix.shape}")
# # regression_matrix.to_pickle(os.path.join(data_dir, drug, "whole_genome/regression_matrix_no_drug_res.pkl"))

# regression_matrix = pd.read_pickle(os.path.join(data_dir, drug, "whole_genome/regression_matrix_no_drug_res.pkl"))
# pca_df = pd.read_csv(os.path.join(data_dir, drug, "whole_genome/PCA_eigenvec_df.csv"), index_col=0)

# scaler = StandardScaler()

# X = scaler.fit_transform(pd.concat([regression_matrix.loc[df_phenos.index.values], pca_df.loc[df_phenos.index.values]], axis=1).values)
# y = np.log2(df_phenos[f"{drug}_midpoint"].values)

# # X_train = scaler.fit_transform(pd.concat([regression_matrix.loc[df_phenos.query("category=='original_train_set'").index.values],
# #                                           pca_df.loc[df_phenos.query("category=='original_train_set'").index.values]
# #                                          ], axis=1).values
# #                               )

# # X_test = scaler.fit_transform(pd.concat([regression_matrix.loc[df_phenos.query("category=='original_test_set'").index.values],
# #                                          pca_df.loc[df_phenos.query("category=='original_test_set'").index.values]
# #                                         ], axis=1).values
# #                               )

# # y_train = np.log2(df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values)
# # y_test = np.log2(df_phenos.query("category=='original_test_set'")[f"{drug}_midpoint"].values)

# # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(X.shape, y.shape)

# model = ElasticNetCV(l1_ratio=np.linspace(0, 1, 11),
#                      alphas=np.logspace(-3, 3, 7),
#                      cv=5,
#                      fit_intercept=True, 
#                      max_iter=10000,
#                      verbose=2,
#                      n_jobs=4
#                     )

# model.fit(X, y)
# pickle.dump(model, open(os.path.join(data_dir, drug, "whole_genome/full_feature_select.sav"), 'wb'))
# print(model.l1_ratio_, model.alpha_)
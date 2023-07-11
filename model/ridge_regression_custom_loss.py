import pandas as pd
import numpy as np
import sys, glob, os, yaml, sparse, warnings, tracemalloc, pickle

from evcouplings.align import Alignment
import scipy.stats as st
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
import scipy.optimize

from Bio import SeqIO
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings("ignore")

# utils files are in the utils_file folder
sys.path.append("utils")
from data_utils import *
from model_utils import *
from analysis_utils import *


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))

binary = False
binary_thresh = kwargs["binary_thresh"]

drug = kwargs["drug"]
locus_list = kwargs["locus_list"]
output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
num_loci = len(locus_list)
include_lineage = kwargs["include_lineage"]
loss_type = kwargs["loss_type"]

fastas = [os.path.join(genotype_input_directory, locus + ".fasta") for locus in locus_list]
print(f"{len(fastas)} loci")
df_phenos = pd.read_csv(phenotype_file)
isolate_order = df_phenos["ROLLINGDB_ID"].values

# make a directory for the small fasta files
ridge_dir = os.path.join(output_path, "ridge")
bootstrap_dir = os.path.join(output_path, "ridge", "bootstrapping")

if not os.path.isdir(bootstrap_dir):
    os.makedirs(bootstrap_dir)

def subset_fasta_files():
    '''
        Iterate through each fasta file, select only the positions that are not identical in the dataset. Keep only the isolates with phenotype data, reorder the alignment to match isolate_order, save a reduced fasta file
    '''
    
    for file in fastas:
       
        print(file)
        aln = Alignment.from_file(open(file))
        
        # the fasta files contain sequences for all isolates. Keep only the isolates in the phenotypes file
        # need indices for splitting alignment in evcouplings
        keep_idx = [i for i, fName in enumerate(aln.ids) if os.path.basename(fName).replace(".eff", "").replace(".vcf", "") in isolate_order]
        assert len(keep_idx) == len(isolate_order)
        
        # if len(keep_idx) != len(isolate_order):
        #     raise ValueError(len(keep_idx), len(isolate_order))
        
        # this keeps only sites where the major allele frequency is less than 1. Major allele frequence = 1 means that all isolates are the same at the site
        # drop sites that are identical in all isolates because they have no impact on the regression
        indices_to_keep = np.where(aln.frequencies.max(axis=1) < 1)[0]
        subset_alignment = aln.select(columns=indices_to_keep, sequences=keep_idx)        
       
        print("original alignment shape", aln.matrix.shape)
        print("reduced alignment shape", subset_alignment.matrix.shape)

        # Cleanup giant variables
        del aln

        # First, correct the ids in the alignment so that they match the ROLLINGDB_IDs in the dataframe of phenotypes
        subset_alignment.ids = [os.path.basename(fName).replace(".eff", "").replace(".vcf", "") for fName in subset_alignment.ids]

        subset_alignment.id_to_index = {x:idx for idx,x in enumerate(subset_alignment.ids)}
    
        # Get the indices that would correctly reorder the alignment to match isolate_order
        reorder_index = [
            subset_alignment.id_to_index[x] if x in list(subset_alignment.id_to_index.keys()) else print(x) for x in isolate_order
        ]
        
        # Reorder based on reorder_index
        subset_alignment.ids = np.array(subset_alignment.ids)[reorder_index]
        assert sum(isolate_order != subset_alignment.ids) == 0

        subset_alignment.matrix = subset_alignment.matrix[reorder_index, :]

        # Get the name of the fasta file for saving
        name = os.path.basename(file).split(".")[0]

        # save the reduced file sequences and gene indices to new files
        subset_alignment.write(open(os.path.join(ridge_dir, f"{name}_reduced.fasta"), "w"))
        np.save(os.path.join(ridge_dir, f"{name}_indices.npy"), indices_to_keep)
        
        

# get the reduced files made in the previous function
reduced_fastas = glob.glob(os.path.join(ridge_dir, "*_reduced.fasta"))

if len(reduced_fastas) == len(fastas):
    print("Found reduced fasta files. Proceeding with modeling...")
else:
    subset_fasta_files()

# get the reduced files made in the previous function. Re-acquire the list of reduced fasta files since they've now been made
reduced_fastas = glob.glob(os.path.join(ridge_dir, "*_reduced.fasta"))

assert len(reduced_fastas) == len(fastas)

# Compute the total number of sites in our model by summing the length of all the alignment
total_sites = 0

for file in reduced_fastas:
    aln = Alignment.from_file(open(file))    
    total_sites += aln.L

print("total sites", total_sites)
total_seqs = aln.N

# Matrix to store the data for learning
X = np.zeros((total_seqs, total_sites), dtype=np.int8)

current_index = 0

for file in reduced_fastas:
    aln = Alignment.from_file(open(file), alphabet='-ACGT')
    
    # only use sequence alignments with sites for the model. Otherwise get a vectorize error
    if aln.L != 0:
        
        # Tells you which character is the most frequent in each site
        who_is_max = np.argmax(aln.frequencies, axis=1)

        # Major allele is encoded as 0, minor allele(s) as 1
        major_minor = aln.matrix_mapped != who_is_max

        # Add the encoding to the X matrix
        X[:, current_index:(current_index + major_minor.shape[1])] = major_minor

        # Keep track of how many sites in X we have filled in
        current_index = current_index + major_minor.shape[1]
    
# matrix for learning
np.save(os.path.join(ridge_dir, "combined_X.npy"), X)

# Make a table of all of the sites in the model for later mapping
# Note that the sites listed here are indexed within each fasta file - NOT the MTB genome
total_sites = []
genes = []


for file in reduced_fastas:
    
    gene_name = os.path.basename(file).split("_")[0]
    
    numpy_file = file.split("reduced.fasta")[0] + "indices.npy"

    sites = np.load(numpy_file)
    sites = sorted(list(sites))
    
    total_sites += list(sites)
    genes += [gene_name] * len(list(sites))

assert len(genes) == len(total_sites)
    
gene_sites = pd.DataFrame({
    "locus": genes,
    "sites": total_sites,
})

gene_sites.to_csv(os.path.join(ridge_dir, "site_indices.csv"))


##################### CROSS-VALIDATE AND RUN MODEL WITH CUSTOM LOSS FUNCTION #####################



def ridge_mic(X, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10):

    # need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
    df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
    df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)
    
    # Perform baseline model with all data
    scaler = StandardScaler()
        
    if include_lineage:
        print(f"Fitting model with lineages")
        # lineages = pd.get_dummies(df_phenos["Lineage"])
        # lineages.index = df_phenos["ROLLINGDB_ID"]
        
        lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages.values)) == 2
        lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]]
    
        X_train = np.concatenate([X[df_phenos.query("category=='original_train_set'").index, :], lineages.loc[df_train.ROLLINGDB_ID.values, :].values], axis=1)
        X_test = np.concatenate([X[df_phenos.query("category=='original_test_set'").index, :], lineages.loc[df_test.ROLLINGDB_ID.values, :].values], axis=1)
    
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    else:     
        X_train = scaler.fit_transform(X[df_phenos.query("category=='original_train_set'").index, :])   
        X_test = scaler.fit_transform(X[df_phenos.query("category=='original_test_set'").index, :])
        
    y_train = np.log2(df_train[f"{drug}_midpoint"].values)
    lower_bounds_train, upper_bounds_train = df_train[f"{drug}_lower_bound"].values, df_train[f"{drug}_upper_bound"].values

    print(X_train.shape, X_test.shape)
    print(f"Minimizing {loss_type} loss")
    
    reg_param_lst = np.logspace(-6, 6, 13)
    custom_Ridge_model = CustomRidgeCV(alphas=reg_param_lst, cv=5)
    custom_Ridge_model.fit(X_train, y_train, loss_type=loss_type, lower_bounds=lower_bounds_train, upper_bounds=upper_bounds_train)
    
    print(f"Regularization parameter: {custom_Ridge_model.alpha_}")
    pickle.dump(custom_Ridge_model, open(os.path.join(ridge_dir, "model.sav"), "wb"))
    custom_Ridge_model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
    
    y_pred = custom_Ridge_model.predict(X_test)
    summary_df = create_summary_df(df_test, y_pred, drug, binary_thresh, num_loci, model_name="LinReg", binarize=True, save_fName=os.path.join(ridge_dir, "test_predictions.csv"))
    summary_df["CV"] = 0
    
    bootstrap_df = []

    print("Performing bootstrapping...")
    for i in range(num_bootstrap):
        
        train_idx = np.random.choice(np.arange(0, len(X_train)), size=len(X_train), replace=True)
        
        X_bs = X_train[train_idx, :]
        y_bs = y_train[train_idx]
        lower_bounds_bs = lower_bounds_train[train_idx]
        upper_bounds_bs = upper_bounds_train[train_idx]
        
        # use regularization parameter determined above
        bs_model = CustomRidge(alpha=custom_Ridge_model.alpha_)
        bs_model.fit(X_bs, y_bs, loss_type=loss_type, lower_bounds=lower_bounds_bs, upper_bounds=upper_bounds_bs)
        
        pickle.dump(bs_model, open(os.path.join(bootstrap_dir, f"model_{i}.sav"), "wb"))
        bs_model = pickle.load(open(os.path.join(bootstrap_dir, f"model_{i}.sav"), "rb"))
        y_pred_bs = bs_model.predict(X_test)
        
        bs_summary_df = create_summary_df(df_test, y_pred_bs, drug, binary_thresh, num_loci, "LinReg", binarize=True, save_fName=None)
        bs_summary_df["CV"] = i + 1
        bootstrap_df.append(bs_summary_df)
        
    bootstrap_df = pd.concat(bootstrap_df)
    final_df = pd.concat([summary_df, bootstrap_df], axis=0)
    final_df[["Lineage", "Num_Loci"]] = [int(include_lineage), num_loci]

    return final_df


X = np.load(os.path.join(ridge_dir, "combined_X.npy"))

results_df = ridge_mic(X, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10)
results_df.to_csv(os.path.join(ridge_dir, "ridge_results.csv"), index=False)
    
# # delete all large files to save space (this script is very fast, so time is not an issue)
# for fName in reduced_fastas:
#     os.remove(fName)
    
# for fName in glob.glob(os.path.join(ridge_dir, "*.npy")):
#     os.remove(fName)
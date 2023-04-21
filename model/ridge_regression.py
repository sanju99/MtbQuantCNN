import pandas as pd
import numpy as np
from matplotlib import transforms
import sys, glob, os, yaml, sparse, warnings, tracemalloc

from evcouplings.align import Alignment
import scipy.stats as st
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV, SGDClassifier, SGDRegressor
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold

from Bio import SeqIO
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

# utils files are in the utils_file folder
sys.path.append("utils")
from model_utils import *


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))

binary = kwargs["binary"]
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
if not os.path.isdir(ridge_dir):
    os.makedirs(ridge_dir)

def subset_fasta_files():
    '''
        Iterate through each fasta file, select only the positions that are not identical in the dataset. Keep only the isolates with phenotype data, reorder the alignment to match isolate_order, save a reduced fasta file
    '''
    
    for file in fastas:
       
        print(file)
        aln = Alignment.from_file(open(file))
        
        # the fasta files contain sequences for all isolates. Keep only the isolates in the phenotypes file
        # need indices for splitting alignment in evcouplings
        keep_ids = [i for i, name in enumerate(aln.ids) if os.path.basename(name).split(".")[0] in isolate_order]
        # assert len(keep_ids) == len(isolate_order)

        # drop sites that are identical in all isolates because they have no impact on the regression
        # indices_to_keep = np.where(aln.frequencies.max(axis=1) < 1)[0]
        
        # this keeps only sites where the major allele frequency is less than 1. Major allele frequence = 1 means that all isolates are the same at the site
        
        # keep sites with a minor allele fraction of at least 0.1%
        indices_to_keep = np.where(aln.frequencies.max(axis=1) < 1)[0]
        subset_alignment = aln.select(columns=indices_to_keep, sequences=keep_ids)        
       
        print("original alignment shape", aln.matrix.shape)
        print("reduced alignment shape", subset_alignment.matrix.shape)

        # Cleanup giant variables
        del aln

        # First, correct the ids in the alignment so that they match the ROLLINGDB_IDs in the dataframe of phenotypes
        subset_alignment.ids = [os.path.basename(path).split(".")[0] for path in subset_alignment.ids]
        
        # First, correct the ids in the alignment so that they match the ROLLINGDB_IDs in the dataframe of phenotypes
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
        
        

# # get the reduced files made in the previous function
# reduced_fastas = glob.glob(os.path.join(ridge_dir, "*_reduced.fasta"))

# if len(reduced_fastas) == len(fastas):
#     print("Found reduced fasta files. Proceeding with modeling...")
# else:
#     subset_fasta_files()

# get the reduced files made in the previous function. Re-acquire the list of reduced fasta files since they've now been made
subset_fasta_files()
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


######## Run model ########


def ridge_mic(X, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10):

    # need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
    df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
    df_val = df_phenos.query("category=='original_test_set'").reset_index(drop=True)
    
    # Perform baseline model with all data
    scaler = StandardScaler()
        
    if include_lineage:
        print(f"Fitting model with lineages")
        # lineages = pd.get_dummies(df_phenos["Lineage"])
        # lineages.index = df_phenos["ROLLINGDB_ID"]
        
        lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages.values)) == 2
        lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]]
        
#         # the following checks are a bit extra, but including them anyway to be very careful
#         # check that the sum of each row (isolate) is 1, i.e. each isolate has only 1 lineage
#         assert lineages.sum(axis=1).unique() == np.array([1])

#         # minimum number of samples in a particular lineage group should not be 0
#         assert lineages.sum(axis=0).min() > 0
    
        X_train = np.concatenate([X[df_phenos.query("category=='original_train_set'").index, :], lineages.loc[df_train.ROLLINGDB_ID.values, :].values], axis=1)
        X_test = np.concatenate([X[df_phenos.query("category=='original_test_set'").index, :], lineages.loc[df_val.ROLLINGDB_ID.values, :].values], axis=1)
    
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    else:     
        X_train = scaler.fit_transform(X[df_phenos.query("category=='original_train_set'").index, :])   
        X_test = scaler.fit_transform(X[df_phenos.query("category=='original_test_set'").index, :])

    print(X_train.shape, X_test.shape)
    
    y_train = np.log2(df_train[f"{drug}_midpoint"].values)
    y_test = np.log2(df_val[f"{drug}_midpoint"].values)
    
    # get the regularization parameter
    if loss_type == "L1":
        loss_func = "neg_mean_absolute_error"
    elif loss_type == "L2":
        loss_func = "neg_mean_squared_error"
    
    print(f"Minimizing {loss_type} loss")
    
    model = RidgeCV(alphas=np.logspace(-6, 6, 13), scoring=loss_func)
    model.fit(X_train, y_train)
    pickle.dump(model, open(os.path.join(out_dir, "model.sav"), "wb"))
    print(f"Regularization parameter: {model.alpha_}")
    
    y_pred = np.squeeze(model.predict(X_test))
    
    summary_df = create_summary_df(df_phenos.query("category=='original_test_set'").reset_index(drop=True), 
                                   y_pred, 
                                   drug, 
                                   binary_thresh, 
                                   num_loci, 
                                   model_name="CNN", 
                                   binarize=True, 
                                   save_fName=os.path.join(ridge_dir, "test_predictions.csv")
                                  )

    summary_df["CV"] = 0

    bootstrap_df = []

    print("Performing bootstrapping...")
    for i in range(num_bootstrap):
        
        train_idx = np.random.choice(np.arange(0, len(X_train)), size=len(X_train), replace=True)
        
        X_bs = X_train[train_idx, :]
        y_bs = y_train[train_idx]
        
        # use regularization parameter determined above
        bs_model = Ridge(alpha=model.alpha_)
        bs_model.fit(X_bs, y_bs)
        y_bs_pred = np.squeeze(bs_model.predict(X_test))
                
        bs_pred_df = pd.DataFrame({"y_pred": np.squeeze(y_bs_pred), 
                                   "y_test": y_test, 
                                   "lower": df_phenos.query("category=='original_test_set'")[f"{drug}_lower_bound"], 
                                   "upper": df_phenos.query("category=='original_test_set'")[f"{drug}_upper_bound"]
                                  }) 
            
        bs_binned_mae, bs_binned_mse, bs_within_doubling = boundedLoss_predict(bs_pred_df, "y_pred", "y_test", "lower", "upper")
        
        bs_summary_df = pd.DataFrame({"Drug": drug,
                                   "Model": "LinReg",
                                   "Num_Loci": num_loci,
                                   "Binned_MAE": bs_binned_mae,
                                   "Binned_MSE": bs_binned_mse,
                                   "MAE": np.mean(np.abs(bs_pred_df.y_test - bs_pred_df.y_pred)),
                                   "MSE": np.mean((bs_pred_df.y_test - bs_pred_df.y_pred)**2),
                                    "Within_doubling": bs_within_doubling,
                                   "Spearman": st.spearmanr(bs_pred_df.y_test, bs_pred_df.y_pred)[0],
                                   "Pearson": st.pearsonr(bs_pred_df.y_test, bs_pred_df.y_pred)[0],
                                  }, index=[0])

        bs_binary_metrics_df = compute_binary_metrics(bs_pred_df["y_test"], bs_pred_df["y_pred"], binary_thresh, binarize=True)
        bs_summary_df = pd.concat([bs_summary_df, bs_binary_metrics_df], axis=1)
        bs_summary_df["CV"] = i + 1
        bootstrap_df.append(bs_summary_df)
        
    bootstrap_df = pd.concat(bootstrap_df)
    final_df = pd.concat([summary_df, bootstrap_df], axis=0)
    final_df[["Lineage", "Num_Loci"]] = [int(include_lineage), num_loci]

#     print("Performing crossvalidation....")

#     crossValidation_df = []

#     cv_splits = StratifiedKFold(n_splits=num_bootstrap)

#     # include df_train["Binary"] as the y argument so that it stratifies by it
#     # so stratify the training dataset by binary resistance phenotype
#     for rep, (train_idx, val_idx) in enumerate(cv_splits.split(df_train.index, df_train["Binary"])):

#         X_cv_train = scaler.fit_transform(X_train[train_idx])
#         X_cv_val = scaler.fit_transform(X_train[val_idx])
        
#         y_cv_train = y_train[train_idx]
#         y_cv_val = y_train[val_idx]
        
#         cv_model = Ridge(alpha=model.alpha_)
#         cv_model.fit(X_cv_train, y_cv_train)
#         y_cv_pred = np.squeeze(cv_model.predict(X_cv_val))
                
#         cv_pred_df = pd.DataFrame({"y_pred": y_cv_pred,
#                                    "y_test": y_cv_val, 
#                                    "lower": df_train.iloc[val_idx, :][f"{drug}_lower_bound"], 
#                                    "upper": df_train.iloc[val_idx, :][f"{drug}_upper_bound"]
#                                   }) 
            
#         cv_binned_mae, cv_binned_mse, cv_within_doubling = boundedLoss_predict(cv_pred_df, "y_pred", "y_test", "lower", "upper")
        
#         cv_summary_df = pd.DataFrame({"Drug": drug,
#                                    "Model": "LinReg",
#                                    "Num_Loci": num_loci,
#                                    "Binned_MAE": cv_binned_mae,
#                                    "Binned_MSE": cv_binned_mse,
#                                    "MAE": np.mean(np.abs(cv_pred_df.y_test - cv_pred_df.y_pred)),
#                                    "MSE": np.mean((cv_pred_df.y_test - cv_pred_df.y_pred)**2),
#                                    "Within_doubling": cv_within_doubling,
#                                    "Pearson": st.pearsonr(cv_pred_df.y_test, cv_pred_df.y_pred)[0],
#                                   }, index=[0])

#         cv_binary_metrics_df = compute_binary_metrics(cv_pred_df["y_test"], cv_pred_df["y_pred"], binary_thresh, binarize=True)
#         cv_summary_df = pd.concat([cv_summary_df, cv_binary_metrics_df], axis=1)
#         cv_summary_df["CV"] = rep + 1
#         crossValidation_df.append(cv_summary_df)
        
        
    # crossValidation_df = pd.concat(crossValidation_df)
    # final_df = pd.concat([summary_df, crossValidation_df], axis=0)
    # final_df[["Lineage", "Num_Loci"]] = [int(include_lineage), num_loci]

    return final_df




# def ridge_binary(X, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10):
    
#     print(f"Critical concentration: {binary_thresh} ug/mL")
#     # Perform baseline model with all data
#     scaler = StandardScaler()
        
#     if include_lineage:
#         lineages = pd.get_dummies(df_phenos["Lineage"])
#         lineages.index = df_phenos["ROLLINGDB_ID"]
        
#         # the following checks are a bit extra, but including them anyway to be very careful
#         # check that the sum of each row (isolate) is 1, i.e. each isolate has only 1 lineage
#         assert lineages.sum(axis=1).unique() == np.array([1])

#         # minimum number of samples in a particular lineage group should not be 0
#         assert lineages.sum(axis=0).min() > 0
    
#         X_train = np.concatenate([X[df_phenos.query("category=='original_train_set'").index, :], lineages.loc[df_phenos.query("category=='original_train_set'").ROLLINGDB_ID.values, :].values], axis=1)
#         X_test = np.concatenate([X[df_phenos.query("category=='original_test_set'").index, :], lineages.loc[df_phenos.query("category=='original_test_set'").ROLLINGDB_ID.values, :].values], axis=1)
    
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.fit_transform(X_test)
#     else:     
#         X_train = scaler.fit_transform(X[df_phenos.query("category=='original_train_set'").index, :])   
#         X_test = scaler.fit_transform(X[df_phenos.query("category=='original_test_set'").index, :])

#     y_train = (df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"] > binary_thresh).astype(int).values
#     y_test = (df_phenos.query("category=='original_test_set'")[f"{drug}_midpoint"] > binary_thresh).astype(int).values
    
#     model = LogisticRegressionCV(penalty='l2', 
#                                  Cs=np.logspace(-6, 6, 13), 
#                                  class_weight=class_weighting_dictionary(y_train), 
#                                  multi_class='ovr', 
#                                  scoring='neg_log_loss', 
#                                  max_iter=100000
#                                 )
    
#     model.fit(X_train, y_train)
#     y_pred = np.squeeze(model.predict(X_test))
    
#     pred_df = pd.DataFrame({"Isolate": df_phenos.query("category=='original_test_set'")["ROLLINGDB_ID"].values,
#                             "y_pred": np.squeeze(y_pred), 
#                             "y_test": y_test
#                            }) 
#     pred_df.to_csv(os.path.join(ridge_dir, "binary_test_predictions.csv"), index=False)
    
#     summary_df = compute_binary_metrics(y_test, y_pred, binary_thresh, binarize=False)
#     summary_df["CV"] = 0
        
#     bootstrap_df = []

#     print("Performing bootstrapping...")
#     for i in range(num_bootstrap):
        
#         train_idx = np.random.choice(np.arange(0, len(X_train)), size=len(X_train), replace=True)
        
#         X_bs = X_train[train_idx, :]
#         y_bs = y_train[train_idx]
        
#         # use regularization parameter determined above
#         bs_model = LogisticRegression(penalty="l2", C=model.C_[0], multi_class='ovr', class_weight=class_weighting_dictionary(y_bs), max_iter=100000)
#         bs_model.fit(X_bs, y_bs)
#         y_bs_pred = np.squeeze(bs_model.predict(X_test))

#         bs_binary_metrics_df = compute_binary_metrics(y_test, y_bs_pred, binary_thresh, binarize=False)
#         bs_binary_metrics_df["CV"] = i + 1
#         bootstrap_df.append(bs_binary_metrics_df)

        
#     bootstrap_df = pd.concat(bootstrap_df)
#     final_df = pd.concat([summary_df, bootstrap_df])
#     final_df[["Lineage", "Num_Loci"]] = [int(include_lineage), num_loci]
#     final_df.insert(0, "Model", "LogReg")
#     final_df.insert(0, "Drug", drug)

#     return final_df



###### 

X = np.load(os.path.join(ridge_dir, "combined_X.npy"))
X.shape, df_phenos.shape

if binary:
    results_df = ridge_binary(X, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10)
    results_df.to_csv(os.path.join(ridge_dir, "binary_ridge_results.csv"), index=False)
else:
    results_df = ridge_mic(X, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10)
    results_df.to_csv(os.path.join(ridge_dir, "ridge_results.csv"), index=False)
    
# delete all large files to save space (this script is very fast, so time is not an issue)
for fName in reduced_fastas:
    os.remove(fName)
    
for fName in glob.glob(os.path.join(ridge_dir, "*.npy")):
    os.remove(fName)
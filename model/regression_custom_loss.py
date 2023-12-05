import numpy as np
import pandas as pd
import glob, os, subprocess, vcf, shutil, sparse, yaml, sys, pickle, warnings, tracemalloc
import scipy.stats as st
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from model_utils import *
from analysis_utils import *

from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

tracemalloc.start()

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")


_, config_file = sys.argv

kwargs = yaml.safe_load((open(config_file)))
drug = kwargs["drug"]
include_lineage = kwargs["include_lineage"]
binary_thresh = kwargs["binary_thresh"]
locus_list = kwargs["locus_list"]
num_loci = len(locus_list)
genotype_input_directory = kwargs["genotype_input_directory"]
df_phenos = pd.read_csv(kwargs['phenotype_file'])
loss_type = "L1"

include_peptide_length = True
lowAF = False

if lowAF:
    suffix = "_lowAF"
else:
    suffix = ""

# naming consistency
output_path = kwargs["output_path"]
output_path = output_path.replace("_lineage", "").replace("_peptide", "")

if include_peptide_length:
    output_path += "_peptide"
    
if include_lineage:
    output_path += "_lineage"

ridge_dir = os.path.join(output_path, "ridge")
    
if not os.path.isdir(ridge_dir):
    os.makedirs(ridge_dir)

print(f"Saving results to {ridge_dir}")
seq_data_path = output_path.replace("_lineage", "").replace("_peptide", "")

# make dataframes of coordinates
gene_coords, _ = get_gene_coords(locus_list, genotype_input_directory)
h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, genotype_input_directory)

if os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_full.npz")): 
    print("Input one-hot encodings file exists. Proceeding with modeling \n")    
else:
    print("Making input one-hot encodings file...\n")
    make_geno_pheno_files(seq_data_path, **kwargs)

# read in matrices of input sequences
full_matrix = sparse.load_npz(f"{seq_data_path}/pkl_sparse_full{suffix}.npz").todense()
ref_matrix = sparse.load_npz(f"{seq_data_path}/pkl_sparse_ref.npz").todense()

# same input file for models with and without lineages because these are just sequences
full_mat_file = os.path.join(ridge_dir.replace("_lineage", "").replace("_peptide", ""), f"full_seq_matrix{suffix}.pkl")
ref_mat_file = os.path.join(ridge_dir.replace("_lineage", "").replace("_peptide", ""), f"ref_seq_matrix{suffix}.pkl")


if os.path.isfile(full_mat_file) and os.path.isfile(ref_mat_file):

    print(f"Found existing input sequence matrix")
    df_seq = pd.read_pickle(full_mat_file)
    df_ref = pd.read_pickle(ref_mat_file)

else:
    print(f"Creating input sequence matrix")
    df_seq = []
    df_ref = []
    
    for locus in locus_list:

        single_locus_matrix, single_locus_ref_matrix = get_single_locus_Reg_input(locus, locus_list, df_phenos, full_matrix, ref_matrix, h37Rv_coords)
        df_seq.append(single_locus_matrix)
        df_ref.append(single_locus_ref_matrix)

    df_seq = pd.concat(df_seq, axis=1)
    df_seq.index = df_phenos["ROLLINGDB_ID"].values
    df_seq.to_pickle(full_mat_file)

    df_ref = pd.concat(df_ref, axis=1)
    df_ref.to_pickle(ref_mat_file)


# # make peptide lengths dataframe if it has not already been created
# if include_peptide_length and not os.path.isfile(os.path.join(seq_data_path, "gene_peptide_lengths.csv")):

#     if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
#         all_loci_seq = create_all_loci_matrices(config_file)
#         pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))
    
#     locus_peptide_lengths = make_CDS_length_df(locus_list, genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
    
#     # keep index because that's the samples column
#     locus_peptide_lengths.to_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"))


def ridge_cv_select_regularization(df_seq, df_phenos, drug, include_lineage, binary_thresh, num_loci):

    print(df_seq.shape)
    
    if include_lineage:
        print("    Fitting model with lineages")
        lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages.values)) == 2
        lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]]
        lineages.columns = [f"lineageSNP_{col}" for col in lineages.columns]

        df_seq = df_seq.merge(lineages, left_index=True, right_index=True, how="left")
        print(df_seq.shape)

    if include_peptide_length:
        print("    Fitting model with peptide lengths")
        locus_peptide_lengths = pd.read_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"), index_col=[0])

        # reorder the isolates to match the rest of the data, then get the values to make it a matrix
        locus_peptide_lengths = locus_peptide_lengths.loc[df_phenos["ROLLINGDB_ID"]]
        df_seq = df_seq.merge(locus_peptide_lengths, left_index=True, right_index=True, how="left")
        print(df_seq.shape)
    
    print(f"    Minimizing {loss_type} loss")
    reg_param_lst = np.logspace(-3, 3, 7)
    losses_df = pd.DataFrame(columns=["alpha", "split", "val_loss"])
    cv_splits = StratifiedKFold(n_splits=5)

    y = np.log2(df_phenos[f"{drug}_midpoint"].values)

    # use 5-fold cross-validation, stratifying by binary resistance phenotype
    # use the same 5 splits for all regularization parameters to reduce variance
    for split, (train_idx, test_idx) in enumerate(cv_splits.split(df_phenos.index, df_phenos["Binary"])):

        # after splitting the data, remove redundancies in the training data, then X_cv_test must have only the features from X_cv_train
        df_seq_train = df_seq.iloc[train_idx, :]

        # df_ref and h37Rv_coords are global variables because they are for the full H37Rv reference genome
        df_seq_train = remove_redundant_sites_for_Reg(df_seq_train, df_ref, h37Rv_coords)
        
        # get just the values
        X_cv_train = df_seq_train.values

        # get test indices, then keep only the features from the train matrix after removing redundancies
        X_cv_test = df_seq.iloc[test_idx, :][df_seq_train.columns].values
        print(f"Split {split} data shapes: {X_cv_train.shape}, {X_cv_test.shape}")

        y_cv_train = y[train_idx]
        y_cv_test = y[test_idx]

        # perform mean / SD scaling using the training set
        train_mean = X_cv_train.mean()
        train_sd = X_cv_train.std()
            
        X_cv_train = (X_cv_train - train_mean) / train_sd
        X_cv_test = (X_cv_test - train_mean) / train_sd

        lower_bounds_cv_test = df_phenos.iloc[test_idx, :][f"{drug}_lower_bound"].values
        upper_bounds_cv_test = df_phenos.iloc[test_idx, :][f"{drug}_upper_bound"].values
        
        for i, alpha in enumerate(reg_param_lst):
    
            # fit a model on the training data using the given regularization parameter
            cv_model = Ridge(alpha=alpha)
            # cv_model = CustomRidge(alpha, 
            #                         df_phenos.iloc[train_idx, :][f"{drug}_lower_bound"].values, 
            #                         df_phenos.iloc[train_idx, :][f"{drug}_upper_bound"].values, 
            #                         loss_type=loss_type, 
            #                         solver="auto"
            #                        )
            cv_model.fit(X_cv_train, y_cv_train)
    
            # get predictions on the test set, then compute binned error. Use the same functional form as for the CNN
            y_hat_cv = np.squeeze(cv_model.predict(X_cv_test))

            losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "split": split, "val_loss": boundedLoss_Reg(y_hat_cv, y_cv_test, lower_bounds_cv_test, upper_bounds_cv_test, loss_type=loss_type)}, index=[0])])
            losses_df.to_csv(os.path.join(ridge_dir, "reg_param_losses.csv"), index=False)

    select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
    print(f"    Regularization parameter: {select_alpha}, minimum validation loss: {losses_df.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")
    return select_alpha



def ridge_mic(df_seq, df_phenos, drug, include_lineage, binary_thresh, num_loci, alpha):

    print(df_seq.shape)
    
    if include_lineage:
        print("    Fitting model with lineages")

        lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages.values)) == 2 # only relevant if we are binary encoding the lineages, but they have been switched to AF
        lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]]
        lineages.columns = [f"lineageSNP_{col}" for col in lineages.columns]

        df_seq = df_seq.merge(lineages, left_index=True, right_index=True, how="left")

    if include_peptide_length:
        print("    Fitting model with peptide lengths")
        locus_peptide_lengths = pd.read_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"), index_col=[0])

        # reorder the isolates to match the rest of the data, then get the values to make it a matrix
        locus_peptide_lengths = locus_peptide_lengths.loc[df_phenos["ROLLINGDB_ID"]]
        df_seq = df_seq.merge(locus_peptide_lengths, left_index=True, right_index=True, how="left")
        print(df_seq.shape)

    print(df_seq.columns)
    
    train_idx = df_phenos.query("category=='original_train_set'").index.values
    test_idx = df_phenos.query("category=='original_test_set'").index.values

    df_seq_train = df_seq.iloc[train_idx, :]
    df_seq_test = df_seq.iloc[test_idx, :]

    # only use the lowAF for predicting using an existing model, not for training
    if not lowAF:
        # remove redundant sites
        df_seq_train = remove_redundant_sites_for_Reg(df_seq_train, df_ref, h37Rv_coords)
    
        # save the features to use for validation data prediction, then use only those for the test matrix
        keep_features = df_seq_train.columns
        pd.Series(keep_features).to_csv(os.path.join(ridge_dir, "model_features.txt"), index=False, sep="\t", header=None)
            
        print(f"    Minimizing {loss_type} loss")
        y_train = np.log2(df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values)
            
        X_train = df_seq_train.values
            
        # perform mean / SD scaling using the training set, then save the values for later use, if needed
        train_mean = X_train.mean()
        train_sd = X_train.std()
        np.save(os.path.join(ridge_dir, "train_mean_sd.npy"), np.array([train_mean, train_sd]))
            
        X_train = (X_train - train_mean) / train_sd
        print(f"    Train: {X_train.shape}")
        
        # fit a new model with the selected alpha parameter
        model = Ridge(alpha=alpha)
        # model = CustomRidge(alpha, 
        #                     df_phenos.query("category=='original_train_set'")[f"{drug}_lower_bound"], 
        #                     df_phenos.query("category=='original_train_set'")[f"{drug}_upper_bound"], 
        #                     loss_type=loss_type, 
        #                     solver="auto"
        #                    )
        model.fit(X_train, y_train)
        pickle.dump(model, open(os.path.join(ridge_dir, "model.sav"), "wb"))
    else:
        # get the mean and standard deviation from the original X_train to standardize the input matrix with
        train_mean, train_sd = np.load(os.path.join(ridge_dir, "train_mean_sd.npy"))
        
        # get the features used for training the original model with AF = 75%
        keep_features = pd.read_csv(os.path.join(ridge_dir, "model_features.txt"), sep="\t", header=None)[0].values

    df_seq_test = df_seq_test[keep_features]
    X_test = df_seq_test.values
    print(f"Test: {X_test.shape}")
    X_test = (X_test - train_mean) / train_sd

    model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
    y_pred = model.predict(X_test)
    
    summary_df = create_summary_df(df_phenos.query("category=='original_test_set'"), y_pred, drug, binary_thresh, num_loci, model_name="LinReg", binarize=True, save_fName=os.path.join(ridge_dir, f"test_predictions{suffix}.csv"))
    summary_df["CV"] = 0

    if loss_type == "L1":
        print(f"    Final Binned MAE: {summary_df['Binned_MAE'].values[0]}")
    else:
        print(f"    Final Binned MSE: {summary_df['Binned_MSE'].values[0]}")

    return summary_df


if os.path.isfile(os.path.join(ridge_dir, "reg_param_losses.csv")):
    losses_df = pd.read_csv(os.path.join(ridge_dir, "reg_param_losses.csv"))
    select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
else:
    select_alpha = ridge_cv_select_regularization(df_seq, df_phenos, drug, include_lineage, binary_thresh, num_loci)

results_df = ridge_mic(df_seq, df_phenos, drug, include_lineage, binary_thresh, num_loci, select_alpha)
results_df[["Lineage", "Num_Loci"]] = [int(include_lineage), num_loci]
results_df.to_csv(os.path.join(ridge_dir, f"results{suffix}.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
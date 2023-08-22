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

from sklearn.linear_model import Ridge, RidgeCV

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
loss_type = kwargs["loss_type"]
fasta_dir = kwargs["genotype_input_directory"]

df_phenos = pd.read_csv(kwargs['phenotype_file'])

out_dir = kwargs["output_path"]
ridge_dir = os.path.join(out_dir, "ridge")
bootstrap_dir = os.path.join(out_dir, "ridge", "bootstrapping")
    
if not os.path.isdir(bootstrap_dir):
    os.makedirs(bootstrap_dir)


gene_coords, _ = get_gene_coords(locus_list, fasta_dir)
h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)

# read in matrices of input sequences
train_matrix = sparse.load_npz(f"{out_dir.replace('_lineage', '')}/pkl_sparse_train.npz").todense()
test_matrix = sparse.load_npz(f"{out_dir.replace('_lineage', '')}/pkl_sparse_test.npz").todense()
ref_matrix = sparse.load_npz(f"{out_dir.replace('_lineage', '')}/pkl_sparse_ref.npz").todense()

# make dataframes of coordinates
gene_coords, _ = get_gene_coords(locus_list, fasta_dir)
h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)

def get_single_locus_Reg_input(locus, locus_list, train_matrix, test_matrix, ref_matrix):

    locus_idx = locus_list.index(locus)

    train_samples = train_matrix.shape[0]
    test_samples = test_matrix.shape[0]
    
    one_hot_encodings = train_matrix.shape[1]
    longest_locus = train_matrix.shape[2]
    num_loci = train_matrix.shape[3]
    assert one_hot_encodings == 5

    # turn the matrices into dataframes for easy manipulation
    df_train = pd.DataFrame(np.reshape(train_matrix[:, :, :, locus_idx], (train_samples, one_hot_encodings * longest_locus), order='F'))
    df_test = pd.DataFrame(np.reshape(test_matrix[:, :, :, locus_idx], (test_samples, one_hot_encodings * longest_locus), order='F'))
    df_ref = pd.DataFrame(np.reshape(ref_matrix[:, :, :, locus_idx], (1, one_hot_encodings * longest_locus), order='F'))

    # need to get all the nucleotide positions to name the columns. This makes manipulation easier and is also useful to keep track of which positions went into the model (interpretability)
    # k is an iterator to keep track of indels
    seq_coords = []
    k = 0
    
    for coord in h37Rv_coords[:, locus_idx]:

        # indels -- position is NaN, so give unique names that are a concatenation of the locus and an index
        if pd.isnull(coord):
            coord = f"{locus}_{k}"
            k += 1
        else:
            coord = str(int(coord))
            
        seq_coords += [f"{coord}_{nuc}" for nuc in BASE_TO_COLUMN.keys()]
    
    assert len(seq_coords) == len(df_train.columns)
    df_train.columns = seq_coords
    df_test.columns = seq_coords

    # this is a dataframe of length 1
    df_ref.columns = seq_coords
    h37Rv_ref_seq = df_ref[df_ref.columns[(df_ref.loc[0] == 1)]]

    # keep only variables that are not the same everywhere because there is no signal
    df_train_keep = df_train.loc[:, df_train.nunique() > 1]    
    keep_pos = ['_'.join(val.split("_")[:-1]) for val in df_train_keep.columns]
    
    # when value_counts = 1, it's for indels, where the only options are indel or not. The four nucleotides are 0 for all samples and get dropped in the previous step
    single_allele_pos =  pd.Series(keep_pos).value_counts()[pd.Series(keep_pos).value_counts() == 2].index.values
    multi_allele_pos = pd.Series(keep_pos).value_counts()[pd.Series(keep_pos).value_counts() > 2].index.values
    
    # for positions with only two alleles (REF and ALT, essentially), only need to keep one because they are redundant information and perfectly correlated
    single_allele_pos_keep_cols = []
    
    # preferentially keep the alternative allele because it makes interpretability easier
    for pos in single_allele_pos:

        single_pos_cols = [col for col in df_train_keep.columns if "_".join(col.split("_")[:-1]) == pos]
        df_keep_single_pos = df_train_keep[single_pos_cols]
        
        alt_col = set(df_keep_single_pos.columns) - set(h37Rv_ref_seq[h37Rv_ref_seq.columns[h37Rv_ref_seq.columns.str.contains(pos)]].columns)
        assert len(alt_col) == 1
        single_allele_pos_keep_cols += list(alt_col)
    
    assert len(single_allele_pos_keep_cols) == len(single_allele_pos)

    # create training and testing dataframes from the columns to keep (which is based on df_train only)
    df_train_final = pd.concat([df_train_keep[single_allele_pos_keep_cols], df_train_keep[df_train_keep.columns[df_train_keep.columns.str.contains('|'.join(multi_allele_pos))]]], axis=1)

    # for the test dataframe, keep only the columns determined from the train dataframe
    return df_train_final, df_test[df_train_final.columns], seq_coords, df_train_final.columns


    
# same input file for models with and without lineages because these are just sequences
train_mat_file = os.path.join(ridge_dir.replace("_lineage", ""), "train_seq_matrix.pkl")
test_mat_file = os.path.join(ridge_dir.replace("_lineage", ""), "test_seq_matrix.pkl")
feat_names_file = os.path.join(ridge_dir.replace("_lineage", ""), "all_feature_names.txt")
model_feat_names_file = os.path.join(ridge_dir.replace("_lineage", ""), "model_feature_names.txt")

if os.path.isfile(feat_names_file) and os.path.isfile(model_feat_names_file):#os.path.isfile(train_mat_file) and os.path.isfile(test_mat_file):

    print(f"Found existing input sequence matrices")
    X_train = pd.read_pickle(train_mat_file)
    X_test = pd.read_pickle(test_mat_file)

else:
    print(f"Creating input sequence matrices")
    X_train = []
    X_test = []

    features_lst = []
    model_features_lst = []
    
    for locus in locus_list:
        single_locus_train, single_locus_test, single_locus_features, single_locus_model_features = get_single_locus_Reg_input(locus, locus_list, train_matrix, test_matrix, ref_matrix)
    
        X_train.append(single_locus_train)
        X_test.append(single_locus_test)
        features_lst += list(single_locus_features)
        model_features_lst += list(single_locus_model_features)
    
    X_train = pd.concat(X_train, axis=1)
    X_test = pd.concat(X_test, axis=1)
    
    # no changes were made to sample ordering, so use the exact indexes of the isolates from df_phenos
    X_train.index = df_phenos.query("category=='original_train_set'")["ROLLINGDB_ID"]
    X_test.index = df_phenos.query("category=='original_test_set'")["ROLLINGDB_ID"]
    
    X_train.to_pickle(train_mat_file)
    X_test.to_pickle(test_mat_file)

    # save the feature names (both the full list and the list used to fit the model) for the validation data later
    pd.Series(features_lst).to_csv(feat_names_file, sep="\t", index=False, header=None)
    pd.Series(model_features_lst).to_csv(model_feat_names_file, sep="\t", index=False, header=None)



def ridge_mic(X_train, X_test, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10):

    df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
    df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)
    
    if include_lineage:
        print(f"Fitting model with lineages")

        lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages.values)) == 2
        lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]]

        X_train = X_train.merge(lineages, left_index=True, right_index=True, how="left")
        X_test = X_test.merge(lineages, left_index=True, right_index=True, how="left")
        
    X_train = X_train.values
    X_test = X_test.values
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
    # perform mean / SD scaling using the training set
    train_mean = X_train.mean()
    train_sd = X_train.std()
        
    X_train = (X_train - train_mean) / train_sd
    X_test = (X_test - train_mean) / train_sd
    y_train = np.log2(df_train[f"{drug}_midpoint"].values)
    y_test = np.log2(df_test[f"{drug}_midpoint"].values)
    
    lower_bounds_train, upper_bounds_train = df_train[f"{drug}_lower_bound"].values, df_train[f"{drug}_upper_bound"].values
    lower_bounds_test, upper_bounds_test = df_test[f"{drug}_lower_bound"].values, df_test[f"{drug}_upper_bound"].values

    print(f"Minimizing {loss_type} loss")
    
    reg_param_lst = np.logspace(-6, 6, 13)
    losses_df = pd.DataFrame(columns=["alpha", "val_loss"])
    
    for i, alpha in enumerate(reg_param_lst):

        # fit a model on the training data using the given regularization parameter
        cv_model = CustomRidge(alpha=alpha)
        cv_model.fit(X_train, y_train, loss_type=loss_type, lower_bounds=lower_bounds_train, upper_bounds=upper_bounds_train)

        # get predictions on the test set, then compute binned mean squared error
        y_hat_cv = cv_model.predict(X_test)
        losses_df.loc[i, :] = [alpha, boundedLoss_Reg(y_hat_cv, y_test, lower_bounds_test, upper_bounds_test, loss_type=loss_type)]

    select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
    
    print(f"Regularization parameter: {select_alpha}, minimum CV validation loss: {losses_df.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")

    # fit a new model with the selected alpha parameter
    model = CustomRidge(alpha=select_alpha)
    model.fit(X_train, y_train, loss_type=loss_type, lower_bounds=lower_bounds_train, upper_bounds=upper_bounds_train)
    pickle.dump(model, open(os.path.join(ridge_dir, "model.sav"), "wb"))

    model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
    y_pred = model.predict(X_test)
    
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
        bs_model = CustomRidge(alpha=select_alpha)
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


results_df = ridge_mic(X_train, X_test, df_phenos, drug, include_lineage, binary_thresh, num_loci, num_bootstrap=10)
results_df.to_csv(os.path.join(ridge_dir, "ridge_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
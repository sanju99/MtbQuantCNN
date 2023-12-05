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
loss_type = "L1"
genotype_input_directory = kwargs["genotype_input_directory"]

df_phenos = pd.read_csv(kwargs['phenotype_file'])

include_peptide_length = False
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
bootstrap_dir = os.path.join(output_path, "ridge", "bootstrapping")
    
if not os.path.isdir(bootstrap_dir):
    os.makedirs(bootstrap_dir)

print(f"Saving results to {bootstrap_dir}")
seq_data_path = output_path.replace("_lineage", "").replace("_peptide", "")

gene_coords, _ = get_gene_coords(locus_list, genotype_input_directory)
h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, genotype_input_directory)

if os.path.isfile(os.path.join(output_path.replace("_lineage", ""), "pkl_sparse_full.npz")): 
    print("Input one-hot encodings file exists. Proceeding with modeling \n")    
else:
    print("Making input one-hot encodings file...\n")
    make_geno_pheno_files(seq_data_path, **kwargs)

# read in matrices of input sequences
full_matrix = sparse.load_npz(f"{seq_data_path}/pkl_sparse_full.npz").todense()
ref_matrix = sparse.load_npz(f"{seq_data_path}/pkl_sparse_ref.npz").todense()

# make dataframes of coordinates
gene_coords, _ = get_gene_coords(locus_list, genotype_input_directory)
h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, genotype_input_directory)    

# same input file for models with and without lineages because these are just sequences
full_mat_file = os.path.join(ridge_dir.replace("_lineage", "").replace("_peptide", ""), "full_seq_matrix.pkl")
ref_mat_file = os.path.join(ridge_dir.replace("_lineage", "").replace("_peptide", ""), "ref_seq_matrix.pkl")
# feat_names_file = os.path.join(ridge_dir, "all_feature_names.txt")

# make peptide lengths dataframe if it has not already been created
if include_peptide_length and not os.path.isfile(os.path.join(seq_data_path, "gene_peptide_lengths.csv")):

    if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
        all_loci_seq = create_all_loci_matrices(config_file)
        pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))
    
    locus_peptide_lengths = make_CDS_length_df(locus_list, genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
    
    # keep index because that's the samples column
    locus_peptide_lengths.to_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"))

df_seq = pd.read_pickle(full_mat_file)
df_ref = pd.read_pickle(ref_mat_file)

# get regularization parameter for the full dataset
main_model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
select_alpha = main_model.alpha
del main_model

if include_lineage:
    print(f"    Fitting model with lineages")

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
    
        
# reset index so that it is not the sample anymore, but the absolute index
df_seq_train = df_seq.loc[df_phenos.query("category=='original_train_set'")['ROLLINGDB_ID'].values].reset_index()
df_seq_test = df_seq.loc[df_phenos.query("category=='original_test_set'")['ROLLINGDB_ID'].values].reset_index()
del df_seq


def perform_one_bootstrap_rep(df_seq_train, df_seq_test, df_phenos, rep, drug, binary_thresh, num_loci):

    bs_train_idx = np.random.choice(df_seq_train.index.values, size=len(df_seq_train), replace=True)
    df_seq_bs_train = df_seq_train.iloc[bs_train_idx, :]
    
    # remove redundant sites
    df_seq_bs_train = remove_redundant_sites_for_Reg(df_seq_bs_train, df_ref, h37Rv_coords)

    # save the features to use for validation data prediction, then use only those for the test matrix
    keep_features = df_seq_bs_train.columns
    pd.Series(keep_features).to_csv(os.path.join(bootstrap_dir, f"model_features_{rep}.txt"), index=False, sep="\t", header=None)
    df_seq_bs_test = df_seq_test[keep_features]
        
    print(f"    Minimizing {loss_type} loss using L2 regularization with strength {select_alpha}")
    y_train = np.log2(df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values[bs_train_idx])
        
    X_train = df_seq_bs_train.values
    X_test = df_seq_bs_test.values
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
        
    # perform mean / SD scaling using the training set
    train_mean = X_train.mean()
    train_sd = X_train.std()
        
    X_train = (X_train - train_mean) / train_sd
    X_test = (X_test - train_mean) / train_sd
    
    # fit a new model with the selected alpha parameter
    model = Ridge(alpha=select_alpha)
    model.fit(X_train, y_train)
    pickle.dump(model, open(os.path.join(bootstrap_dir, f"model_{rep}.sav"), "wb"))

    model = pickle.load(open(os.path.join(bootstrap_dir, f"model_{rep}.sav"), "rb"))
    y_pred = model.predict(X_test)
    
    summary_df = create_summary_df(df_phenos.query("category=='original_test_set'"), y_pred, drug, binary_thresh, num_loci, model_name="LinReg", binarize=True, save_fName=os.path.join(bootstrap_dir, f"predictions_{rep}.csv"))
    summary_df["CV"] = rep + 1

    return summary_df, pd.DataFrame({"Mean": train_mean, "SD": train_sd}, index=[0])


bootstrap_df = []
bootstrap_train_mean_sd = []
num_bootstrap = 10

for rep in range(num_bootstrap):

    print(f"\nTraining replicate {rep+1}/{num_bootstrap}")
    bs_summary_df, train_mean_sd_df = perform_one_bootstrap_rep(df_seq_train, df_seq_test, df_phenos, rep, drug, binary_thresh, num_loci)

    if loss_type == "L1":
        print(f"    Final Binned MAE: {bs_summary_df['Binned_MAE'].values[0]}")
    else:
        print(f"    Final Binned MSE: {bs_summary_df['Binned_MSE'].values[0]}")
        
    bootstrap_df.append(bs_summary_df)
    bootstrap_train_mean_sd.append(train_mean_sd_df)

    
bootstrap_df = pd.concat(bootstrap_df, axis=0)
bootstrap_df[["Lineage", "Num_Loci"]] = [int(include_lineage), num_loci]
bootstrap_df.to_csv(os.path.join(bootstrap_dir, "results.csv"), index=False)

bootstrap_train_mean_sd = pd.concat(bootstrap_train_mean_sd, axis=0)
bootstrap_train_mean_sd.to_csv(os.path.join(bootstrap_dir, "train_mean_sd.csv"), index=False)

# returns a tuple: current, peak memory in bytes
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
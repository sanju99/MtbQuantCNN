import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle, tracemalloc

import Bio.SeqUtils
import Bio.Data
from Bio import SeqIO
from Bio.Seq import Seq
from evcouplings.align import Alignment

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlepad'] = 10
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# load all utils functions
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
from model_utils import *
from data_utils import *
from dataloader import MtbGeneDataset
from inSilicoMut_utils import *

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_path = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
vcf_dir = "/n/scratch3/users/s/sak0914/annotated_VCF"

who_variants_clean = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")

tracemalloc.start()


def get_train_test_val_lineages(df_train, df_test, df_val, lineage_fName="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv"):
    
    lineages = pd.read_csv(lineage_fName, index_col=[0])
    assert len(np.unique(lineages.values)) == 2

    train_lineages = lineages.loc[df_train["ROLLINGDB_ID"].values]
    assert sum(train_lineages.index.values != df_train["ROLLINGDB_ID"].values) == 0

    test_lineages = lineages.loc[df_test["ROLLINGDB_ID"].values]
    assert sum(test_lineages.index.values != df_test["ROLLINGDB_ID"].values) == 0

    val_lineages = lineages.loc[df_val["ROLLINGDB_ID"].values]
    assert sum(val_lineages.index.values != df_val["ROLLINGDB_ID"].values) == 0
    
    return train_lineages, test_lineages, val_lineages


def get_combined_model_inputs(X, lineages_matrix, model_type, include_lineage):
    
    if model_type == "CNN":
        
        if include_lineage:
            model_inputs = [X, lineages_matrix.values, np.zeros(len(X)), np.zeros(len(X))]
        else:
            model_inputs = [X, np.zeros(len(X)), np.zeros(len(X))]
        
    elif model_type == "Regression":
        
        scaler = StandardScaler()
        
        if include_lineage:
            model_inputs = scaler.fit_transform(np.concatenate([X, lineages_matrix.values], axis=1))
        else:
            model_inputs = scaler.fit_transform(X)
        
    else:
        raise ValueError(f"{model_type} is not a valid model type!")
        
    return model_inputs



###################### CATALOG BASED CLASSIFICATION ######################


def mutation_catalog_with_bootstrapping(df, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats=["Sensitivity", "Specificity", "AUC", "Accuracy", "Balanced_Acc"]):
    
    df = df.rename(columns={"ROLLINGDB_ID": "Isolate"}).reset_index(drop=True)
    cat1_mutations = who_variants_df.query("drug == @drug & confidence=='1) Assoc w R'").mutation.values
    isolates_R = isolate_variants_df.query("mutation in @cat1_mutations & FILTER == 'PASS' & Isolate in @df.Isolate.values").Isolate.values
        
    df_pred_catalog = df[["Isolate", f"{drug}_midpoint"]]
    df_pred_catalog["y_test"] = (df[f"{drug}_midpoint"] > binary_thresh).astype(int)
    df_pred_catalog["y_pred"] = df_pred_catalog["Isolate"].map(dict(zip(isolates_R, np.ones(len(isolates_R))))).fillna(0).astype(int)
    
    df_stats = compute_binary_metrics(df_pred_catalog["y_test"], df_pred_catalog["y_pred"], binary_thresh, binarize=False)[return_stats]
    df_stats["CV"] = 0
    bs_lst = []
    
    # perform bootstrapping with 10 replicates
    for i in range(10):
        
        bs_sample_idx = np.random.choice(df.index.values, size=len(df), replace=True)
        bs_df = df.iloc[bs_sample_idx, :]
        bs_isolates_R = isolate_variants_df.query("mutation in @cat1_mutations & FILTER == 'PASS' & Isolate in @bs_df.Isolate.values").Isolate.values
        
        bs_pred_catalog = bs_df[["Isolate", f"{drug}_midpoint"]]
        bs_pred_catalog["y_test"] = (bs_df[f"{drug}_midpoint"] > binary_thresh).astype(int)
        bs_pred_catalog["y_pred"] = bs_pred_catalog["Isolate"].map(dict(zip(bs_isolates_R, np.ones(len(bs_isolates_R))))).fillna(0).astype(int)
        
        bs_df_stats = compute_binary_metrics(bs_pred_catalog["y_test"], bs_pred_catalog["y_pred"], binary_thresh, binarize=False)[return_stats]
        bs_df_stats["CV"] = i + 1
        bs_lst.append(bs_df_stats)

    df_return = pd.concat([df_stats, pd.concat(bs_lst, axis=0)], axis=0).reset_index(drop=True)
    df_return["Model"] = "Catalog"
    return df_return



def classify_using_mutation_catalog(drug, data_path, who_variants_df, isolate_variants_df, binary_thresh, return_stats=["Sensitivity", "Specificity", "AUC", "Accuracy", "Balanced_Acc"]):

    df_train = pd.read_csv(os.path.join(data_path, drug, "data_for_model.csv")).query("category=='original_train_set'")
    df_test = pd.read_csv(os.path.join(data_path, drug, "data_for_model.csv")).query("category=='original_test_set'")
    df_val = pd.read_csv(os.path.join(data_path, drug, "validation_data_for_model.csv"))
        
    df_train = mutation_catalog_with_bootstrapping(df_train, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats)
    df_train["Dataset"] = "Train"
    
    df_test = mutation_catalog_with_bootstrapping(df_test, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats)
    df_test["Dataset"] = "Test"
    
    df_val = mutation_catalog_with_bootstrapping(df_val, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats)
    df_val["Dataset"] = "Validation"
    
    return pd.concat([df_train, df_test, df_val], axis=0).reset_index(drop=True)





def get_new_aln_for_regression(isolate_order,
                               locus_list,
                               results_dir,
                               fasta_dir
                              ):
    
    aln = Alignment.from_file(open(os.path.join(fasta_dir, f"{locus_list[0]}.fasta")))
    indices_to_keep = np.load(os.path.join(results_dir, f"ridge/{locus_list[0]}_indices.npy"))
    
    # the fasta files contain sequences for all isolates. Keep only the isolates in the phenotypes file
    # need indices for splitting alignment in evcouplings
    keep_ids = [i for i, name in enumerate(aln.ids) if os.path.basename(name).split(".")[0] in isolate_order]
    # assert len(keep_ids) == len(isolate_order)

    # this keeps only sites where the major allele frequency is less than 1. Major allele frequence = 1 means that all isolates are the same at the site
    # drop sites that are identical in all isolates because they have no impact on the regression
    subset_alignment = aln.select(columns=indices_to_keep, sequences=keep_ids)        
    
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

    total_seqs = subset_alignment.N
    total_sites = subset_alignment.L
    
    # Matrix to store the data for learning
    X = np.zeros((total_seqs, total_sites), dtype=np.int8)

    current_index = 0

    # only use sequence alignments with sites for the model. Otherwise get a vectorize error
    if subset_alignment.L != 0:

        # Tells you which character is the most frequent in each site
        who_is_max = np.argmax(subset_alignment.frequencies, axis=1)

        # Major allele is encoded as 0, minor allele(s) as 1
        major_minor = subset_alignment.matrix_mapped != who_is_max

        # Add the encoding to the X matrix
        X[:, current_index:(current_index + major_minor.shape[1])] = major_minor

        # Keep track of how many sites in X we have filled in
        current_index = current_index + major_minor.shape[1]
        
    return X



def get_new_aln_for_CNN(df,
                        locus_list,
                        fasta_dir
                       ):
    
    # argument = directory that contains the fasta file
    df_genos = make_genotype_df(locus_list, fasta_dir)
    df_genos.index = [name.replace("-", "_").split(".")[0] for name in df_genos.index]
    
    df["ROLLINGDB_ID"] = [name.replace("-", "_") for name in df["ROLLINGDB_ID"]]
    
    # the additional new strains to predict MICs for
    df_genos = df_genos.loc[df["ROLLINGDB_ID"].values]
    
    assert len(df_genos) == len(df)

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[locus + "_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))
        
    return create_X(df_genos)




def get_inputs_for_regression(drug,
                              config_file,
                             ):

    kwargs = yaml.safe_load(open(config_file, "r"))
    
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    results_dir = kwargs["output_path"]
    fasta_dir = os.path.join(os.path.dirname(results_dir), "fastas")
    include_lineage = kwargs["include_lineage"]
    
    df_train = pd.read_csv(kwargs["phenotype_file"]).query("category=='original_train_set'")    
    df_test = pd.read_csv(kwargs["phenotype_file"]).query("category=='original_test_set'")    
    df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))
    
    X = np.load(os.path.join(results_dir, "ridge", "combined_X.npy"))
    X_train = X[df_train.index.values, :]
    X_test = X[df_test.index.values, :]
    
    X_val = get_new_aln_for_regression(df_val["ROLLINGDB_ID"].values,
                                       locus_list,
                                       results_dir,
                                       fasta_dir
                                      )
        
    scaler = StandardScaler()
        
    train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)
    
    X_train = get_combined_model_inputs(X_train, train_lineages, "Regression", include_lineage)
    X_test = get_combined_model_inputs(X_test, test_lineages, "Regression", include_lineage)
    X_val = get_combined_model_inputs(X_val, val_lineages, "Regression", include_lineage)
        
    return X_train, X_test, X_val, df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val.reset_index(drop=True)




def get_inputs_for_CNN(drug,
                       config_file,
                      ):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    results_dir = kwargs["output_path"]
    fasta_dir = os.path.join(os.path.dirname(results_dir), "fastas")
    include_lineage = kwargs["include_lineage"]

    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    
    df_train = pd.read_csv(os.path.join(data_dir, "data_for_model.csv")).query("category=='original_train_set'").reset_index(drop=True)    
    df_test = pd.read_csv(os.path.join(data_dir, "data_for_model.csv")).query("category=='original_test_set'").reset_index(drop=True)    
    df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))

    X_train = sparse.load_npz(os.path.join(results_dir, "pkl_sparse_train.npz")).todense()
    X_test = sparse.load_npz(os.path.join(results_dir, "pkl_sparse_test.npz")).todense()
    
    if not os.path.isfile(os.path.join(results_dir, "pkl_sparse_val.npz")):
        
        X_val = get_new_aln_for_CNN(df_val,
                                    locus_list,
                                    fasta_dir
                                   )
        sparse.save_npz(os.path.join(results_dir, "pkl_sparse_val.npz"), sparse.COO(X_val))
        
    else:
        X_val = sparse.load_npz(os.path.join(results_dir, "pkl_sparse_val.npz")).todense()

    train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)
    
    X_train = get_combined_model_inputs(X_train, train_lineages, "CNN", include_lineage)
    X_test = get_combined_model_inputs(X_test, test_lineages, "CNN", include_lineage)
    X_val = get_combined_model_inputs(X_val, val_lineages, "CNN", include_lineage)
        
    return X_train, X_test, X_val, df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val.reset_index(drop=True)




def get_results_single_model(X, df, dataset, model_type, config_file, bootstrap=True):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    results_dir = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    num_loci = len(kwargs["locus_list"])
    bounded_loss = kwargs["bounded_loss"]
    filter_size = kwargs["filter_size"]
    include_lineage = kwargs["include_lineage"]
    binary = kwargs["binary"]
    BATCH_SIZE = kwargs["batch_size"]
    
    if dataset not in ["Train", "Test", "Validation"]:
        raise ValueError(f"{dataset} is not a valid dataset name")
    
    # at this point, X is a list of length 3 or 4. The first element is the input matrix, the second is the lineages matrix, and the last 2 are the lower and upper bounds
    # if there are no lineages in the model, there is no second element, and the 3rd and 4th are shifted up
    if model_type == "CNN":
        
        if include_lineage:
            num_lineages = X[1].shape[1]
        else:
            num_lineages = 0
        
        longest_locus = X[0].shape[2]
        
        model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
        model.load_weights(os.path.join(results_dir, "best_model.h5"))
        y_pred = model.predict(X, batch_size=BATCH_SIZE).flatten()
        
    else:
        model = pickle.load(open(os.path.join(results_dir, "ridge", "model.sav"), "rb"))
        y_pred = np.squeeze(model.predict(X))
    
    summary_df = create_summary_df(df, y_pred, drug, binary_thresh, num_loci, model_type, binarize=True, save_fName=None)
    summary_df[["Dataset", "CV"]] = [dataset, 0]
    summary_df = [summary_df]
    
    # perform bootstrapping of all 3 datasets
    if bootstrap:
        print(f"Predicting bootstrapped {model_type} models for the {dataset} dataset...")
        for i in range(10):

            if model_type == "CNN":
                model.load_weights(os.path.join(results_dir, "bootstrapping", f"model_{i}.h5"))
                y_pred = model.predict(X, batch_size=BATCH_SIZE).flatten()
            else:
                model = pickle.load(open(os.path.join(results_dir, "ridge", "bootstrapping", f"model_{i}.sav"), "rb"))
                y_pred = np.squeeze(model.predict(X))

            bs_summary = create_summary_df(df, y_pred, drug, binary_thresh, num_loci, model_type, binarize=True, save_fName=None)
            bs_summary[["Dataset", "CV"]] = [dataset, i + 1]
            summary_df.append(bs_summary)

    return pd.concat(summary_df, axis=0)





def get_results_all_models(drug, config_file, model_type, bootstrap=True):
    
    if model_type not in ["CNN", "Regression"]:
        raise ValueError(f"{model_type} is not a valid model type")

    # get input matrices for the training, testing, and validation sets
    if model_type == "Regression":
        data_func = get_inputs_for_regression
    else:
        data_func = get_inputs_for_CNN
        
    # inputs are lists with 
    X_train, X_test, X_val, df_train, df_test, df_val = data_func(drug, config_file)
    
    if model_type == "Regression":
        print(X_train.shape, X_test.shape, X_val.shape)
    else:
        print(X_train[0].shape, X_test[0].shape, X_val[0].shape)

    train_summary = get_results_single_model(X_train, df_train, "Train", model_type, config_file, bootstrap)
    test_summary = get_results_single_model(X_test, df_test, "Test", model_type, config_file, bootstrap)
    val_summary = get_results_single_model(X_val, df_val, "Validation", model_type, config_file, bootstrap)
    
    df_combined = pd.concat([train_summary, test_summary, val_summary]).reset_index(drop=True)
    
    del_cols = ["Drug", "Num_Loci"]
    
    for col in del_cols:
        if col in df_combined.columns:
            del df_combined[col]
            
    return df_combined





# # python3 validation_stats.py Rifampicin RIF ../config_rif.yaml rpoBC_variants_10543isolates.csv
# # python3 validation_stats.py Moxifloxacin MXF ../config_mxf_lineage.yaml gyrBA_variants_8569isolates.csv
# _, drug, drug_abbr, config_file, isolate_variants_file = sys.argv

# isolate_variants = pd.read_csv(os.path.join(data_path, drug_abbr, isolate_variants_file))

# binary_thresh = yaml.safe_load(open(config_file, "r"))["binary_thresh"]

# if not os.path.isfile(f"{drug}/validation/catalog_stats.csv"):
#     catalog_stats = classify_using_mutation_catalog(drug_abbr, data_path, who_variants_clean, isolate_variants, binary_thresh, return_stats=["Sensitivity", "Specificity", "AUC", "AUC_PR", "Accuracy", "Balanced_Acc"])
#     catalog_stats.to_csv(f"{drug}/validation/catalog_stats.csv", index=False)
# else:
#     catalog_stats = pd.read_csv(f"{drug}/validation/catalog_stats.csv")
    
# if not os.path.isfile(f"{drug}/validation/Reg_stats.csv"):
#     Reg_stats = get_results_all_models(drug_abbr, config_file, "Regression", bootstrap=True)
#     Reg_stats.to_csv(f"{drug}/validation/Reg_stats.csv", index=False)
# else:
#     Reg_stats = pd.read_csv(f"{drug}/validation/Reg_stats.csv")

# if not os.path.isfile(f"{drug}/validation/CNN_stats.csv"):
#     CNN_stats = get_results_all_models(drug_abbr, config_file, "CNN", bootstrap=True)
#     CNN_stats.to_csv(f"{drug}/validation/CNN_stats.csv", index=False)
# else:
#     CNN_stats = pd.read_csv(f"{drug}/validation/CNN_stats.csv")
    
# print(catalog_stats.shape, Reg_stats.shape, CNN_stats.shape)

# # returns a tuple: current, peak memory in bytes 
# script_memory = tracemalloc.get_traced_memory()[1] / 1e9
# tracemalloc.stop()
# print(f"Maximum memory used: {script_memory} GB")
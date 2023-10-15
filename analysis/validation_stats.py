import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle, tracemalloc

import Bio.SeqUtils
import Bio.Data
from Bio import SeqIO
from Bio.Seq import Seq

import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# load all utils functions
sys.path.append("utils")
from model_utils import *
from data_utils import *
from analysis_utils import *
from dataloader import MtbGeneDataset
from inSilicoMut_utils import *

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
vcf_dir = "/n/scratch3/users/s/sak0914/annotated_VCF"

who_variants_clean = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")

tracemalloc.start()



def get_CNN_results(config_file, keep_idx=None):
    '''
    Only need to used the base model to get predictions for the validation dataset
    '''
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    results_dir = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    num_loci = len(kwargs["locus_list"])
    bounded_loss = kwargs["bounded_loss"]
    filter_size = kwargs["filter_size"]
    include_lineage = kwargs["include_lineage"]
    BATCH_SIZE = kwargs["batch_size"]
    binary = kwargs["binary"]

    _, X_test, X_val, _, df_test, df_val = get_inputs_for_CNN(config_file, keep_idx=keep_idx)

    longest_locus = X_test[0].shape[2]

    # at this point, X is a list of length 3 or 4. The first element is the input matrix, the second is the lineages matrix, and the last 2 are the lower and upper bounds
    # if there are no lineages in the model, there is no second element, and the 3rd and 4th are shifted up        
    if include_lineage:
        num_lineages = X_test[1].shape[1]
    else:
        num_lineages = 0
            
    model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    model.load_weights(os.path.join(results_dir, "best_model.h5"))

    if X_val is None:
        print(X_test[0].shape)
    else:
        print(X_test[0].shape, X_val[0].shape)        

        # get and save predictions for the validation data on the base model (not bootstrapped). Also print to see
        # don't save predictions if working on the subset, just print them to see
        if keep_idx is None:
            print(create_summary_df(df_val, model.predict(X_val, batch_size=BATCH_SIZE).flatten(), drug, binary_thresh, num_loci, "CNN", binarize=True, save_fName=os.path.join(results_dir, "validation", "CNN_predictions.csv")))
        else:
            print(create_summary_df(df_val, model.predict(X_val, batch_size=BATCH_SIZE).flatten(), drug, binary_thresh, num_loci, "CNN", binarize=True, save_fName=None))
            
    # iterate through the bootstrap models
    results = []
    
    for i in range(10):

        # load the weights of the bootstrapped model
        model.load_weights(os.path.join(results_dir, "bootstrapping", f"model_{i}.h5"))

        # get predictions for the test set (redundant, but it's easy to put them in the same dataframe as the validation set this way)
        bs_summary_df = create_summary_df(df_test, model.predict(X_test, batch_size=BATCH_SIZE).flatten(), drug, binary_thresh, num_loci, "CNN", binarize=True, save_fName=None)
        bs_summary_df[["Dataset", "CV"]] = ["Test", i + 1]
        results.append(bs_summary_df)

        # get predictions for the validation set
        if X_val is not None:
            bs_summary_df = create_summary_df(df_val, model.predict(X_val, batch_size=BATCH_SIZE).flatten(), drug, binary_thresh, num_loci, "CNN", binarize=True, save_fName=None)
            bs_summary_df[["Dataset", "CV"]] = ["Validation", i + 1]
            results.append(bs_summary_df)

    return pd.concat(results, axis=0)

    

def get_Reg_results(config_file, keep_idx=None):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    ridge_dir = os.path.join(kwargs["output_path"], "ridge")
    bootstrap_dir = os.path.join(ridge_dir, "bootstrapping")
    binary_thresh = kwargs["binary_thresh"]
    locus_list = kwargs["locus_list"]
    num_loci = len(locus_list)
    bounded_loss = kwargs["bounded_loss"]
    filter_size = kwargs["filter_size"]
    include_lineage = kwargs["include_lineage"]
    binary = kwargs["binary"]
    BATCH_SIZE = kwargs["batch_size"]
    fasta_dir = kwargs["genotype_input_directory"]

    # make dataframes of coordinates
    gene_coords, _ = get_gene_coords(locus_list, fasta_dir)
    h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)    

    Reg_train, Reg_test, Reg_val, df_train, df_test, df_val = get_inputs_for_regression(config_file)

    if keep_idx is not None:
        if df_val is not None:
            df_val = df_val.iloc[keep_idx]
            Reg_val = Reg_val.iloc[keep_idx, :]

    if include_lineage:
        train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)
    else:
        train_lineages = None
        test_lineages = None
        val_lineages = None

    model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))

    # read in the feature names to train the base model
    feature_names = pd.read_csv(os.path.join(ridge_dir, "model_features.txt"), sep="\t", header=None)[0].values
    
    X_train = prepare_model_inputs(Reg_train, "Regression", include_lineage, feature_names, lineages_matrix=train_lineages)  

    # standard scale using the mean and SD of the training data
    train_mean = X_train.mean()
    train_sd = X_train.std()

    X_test = prepare_model_inputs(Reg_test, "Regression", include_lineage, feature_names, lineages_matrix=test_lineages)
    X_test = (X_test - train_mean) / train_sd

    # combine model inputs with lineages (if so), keep only features used to train the original model, and scale using the mean and SD of the train matrix
    if Reg_val is not None:
        X_val = prepare_model_inputs(Reg_val, "Regression", include_lineage, feature_names, lineages_matrix=val_lineages)
        X_val = (X_val - train_mean) / train_sd
        print(X_test.shape, X_val.shape)

        # get and save predictions for the validation data on the base model (not bootstrapped). Also print to see
        # don't save predictions if working on the subset, just print them to see
        if keep_idx is None:
            print(create_summary_df(df_val, np.squeeze(model.predict(X_val)), drug, binary_thresh, num_loci, "Regression", binarize=True, save_fName=os.path.join(os.path.dirname(ridge_dir), "validation", "Reg_predictions.csv")))
        else:
            print(create_summary_df(df_val, np.squeeze(model.predict(X_val)), drug, binary_thresh, num_loci, "Regression", binarize=True, save_fName=None))
    else:
        print(X_test.shape)
        
    del X_train
    del train_mean
    del train_sd
        
    # dataframe of the mean and SD of each bootstrapped sample. Use these for scaling the validation data
    bs_train_mean_sd = pd.read_csv(os.path.join(bootstrap_dir, "train_mean_sd.csv"))
    
    # iterate through the bootstrap models
    results = []
    
    for i in range(10):

        # load the bootstrapped model
        bs_model = pickle.load(open(os.path.join(bootstrap_dir, f"model_{i}.sav"), "rb"))
        bs_feature_names = pd.read_csv(os.path.join(bootstrap_dir, f"model_features_{i}.txt"), sep="\t", header=None)[0].values

        # get the mean and SD of the exact bootstrap training data used to train each model
        train_mean, train_sd = bs_train_mean_sd.iloc[i, :].values

        # combine model inputs with lineages (if so), keep only features used to train the original model, and scale using the mean and SD of the train matrix
        X_test = prepare_model_inputs(Reg_test, "Regression", include_lineage, bs_feature_names, lineages_matrix=test_lineages)
        X_test = (X_test - train_mean) / train_sd

        # get predictions for the test set (redundant, but it's easy to put them in the same dataframe as the validation set this way)
        bs_summary_df = create_summary_df(df_test, np.squeeze(bs_model.predict(X_test)), drug, binary_thresh, num_loci, "Regression", binarize=True, save_fName=None)
        bs_summary_df[["Dataset", "CV"]] = ["Test", i + 1]
        results.append(bs_summary_df)

        if Reg_val is not None:
            X_val = prepare_model_inputs(Reg_val, "Regression", include_lineage, bs_feature_names, lineages_matrix=val_lineages)
            X_val = (X_val - train_mean) / train_sd
            print(X_test.shape, X_val.shape)

            # get predictions for the validation set
            bs_summary_df = create_summary_df(df_val, np.squeeze(bs_model.predict(X_val)), drug, binary_thresh, num_loci, "Regression", binarize=True, save_fName=None)
            bs_summary_df[["Dataset", "CV"]] = ["Validation", i + 1]
            results.append(bs_summary_df)
        else:
            print(X_test.shape)

    return pd.concat(results, axis=0)

    



_, config_file, val_subset = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
output_path = kwargs["output_path"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]

if not os.path.isdir(os.path.join(output_path, "validation")):
    os.makedirs(os.path.join(output_path, "validation"))

print(f"Saving results to {os.path.join(output_path, 'validation')}")

if val_subset == "True":

    if not os.path.isfile(os.path.join(data_dir, drug, "validation_data_for_model.csv")):
        print(f"There is no validation data for {drug}. Quitting...")
        exit()
    
    df_val = pd.read_csv(os.path.join(data_dir, drug, "validation_data_for_model.csv"))
    keep_idx = df_val.query(f"~(WHO_Cat1_mutation == 1 & {drug}_midpoint < @binary_thresh)").index.values
    print(f"Removing {len(df_val) - len(keep_idx)}/{len(df_val)} isolates with Category 1 mutations and MIC midpoints less than the CC of {binary_thresh}")
    save_suffix = "_noCat1LowMIC"
    
    if len(keep_idx) == 0:
        print("There are no susceptible validation samples with Category 1 mutations. Exiting...")
        exit()
else:
    keep_idx = None
    save_suffix = ""
    

# same classifier with and without lineage, so don't run it for lineage models
# get catalog results for all three datasets -- train, test, and validation because they were not previously computed, but the quant model metrics were already computed for train and test
# therefore, the CNN and Regression stats will only be computed on the validation data
if not include_lineage:
    catalog_stats = classify_using_mutation_catalog(drug, data_dir, who_variants_clean, binary_thresh, valOnlykeepidx=keep_idx, return_stats=["Sensitivity", "Specificity", "Precision", "Accuracy", "Balanced_Acc"])
    catalog_stats.to_csv(os.path.join(output_path, f"validation/catalog_stats{save_suffix}.csv"), index=False)

# CNN_stats = get_CNN_results(config_file, keep_idx=keep_idx)
# CNN_stats.to_csv(os.path.join(output_path, f"validation/CNN_stats{save_suffix}.csv"), index=False)

# Reg_stats = get_Reg_results(config_file, keep_idx=keep_idx)
# Reg_stats.to_csv(os.path.join(output_path, f"validation/Reg_stats{save_suffix}.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
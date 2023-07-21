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
sys.path.append("utils")
from model_utils import *
from data_utils import *
from analysis_utils import *
from dataloader import MtbGeneDataset
from inSilicoMut_utils import *

from sklearn.linear_model import LinearRegression

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_path = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
vcf_dir = "/n/scratch3/users/s/sak0914/annotated_VCF"

who_variants_clean = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")

tracemalloc.start()




def get_results_single_model(X, drug, df, dataset, model_type, config_file, bootstrap=True):
    '''
    X matrices have already been standard scaled using the mean and SD of the training data
    '''
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug_abbr = kwargs["drug"]
    results_dir = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    num_loci = len(kwargs["locus_list"])
    bounded_loss = kwargs["bounded_loss"]
    filter_size = kwargs["filter_size"]
    include_lineage = kwargs["include_lineage"]
    binary = kwargs["binary"]
    BATCH_SIZE = kwargs["batch_size"]
    
    # whether to binarize predictions later
    if binary:
        binarize = False
        model_prefix = "binary_"
    else:
        binarize = True
        model_prefix = ""
    
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
        model.load_weights(os.path.join(results_dir, f"{model_prefix}best_model.h5"))
        y_pred = model.predict(X, batch_size=BATCH_SIZE).flatten()        
    else:
        model = pickle.load(open(os.path.join(results_dir, "ridge", f"{model_prefix}model.sav"), "rb"))
        y_pred = np.squeeze(model.predict(X))


    # save the predictions for the validation dataset to use later
    if dataset == "Validation":
        df_pred = pd.DataFrame({"Isolate": df["ROLLINGDB_ID"].values,
                                "y_pred": y_pred,
                                "y_test": np.log2(df[f"{drug_abbr}_midpoint"].values)
                               })
        df_pred.to_csv(f"analysis/{drug}/validation/{model_type}_predictions.csv", index=False)
    
    summary_df = create_summary_df(df, y_pred, drug_abbr, binary_thresh, num_loci, model_type, binarize=binarize, save_fName=None)
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
                model = pickle.load(open(os.path.join(results_dir, "ridge", "bootstrapping", f"{model_prefix}model_{i}.sav"), "rb")) 
                y_pred = np.squeeze(model.predict(X))

            bs_summary_df = create_summary_df(df, y_pred, drug_abbr, binary_thresh, num_loci, model_type, binarize=True, save_fName=None)
            bs_summary_df[["Dataset", "CV"]] = [dataset, i + 1]
            summary_df.append(bs_summary_df)

    return pd.concat(summary_df, axis=0)





def get_results_all_datasets(drug, drug_abbr, config_file, model_type, bootstrap=True):
    
    if model_type not in ["CNN", "Reg"]:
        raise ValueError(f"{model_type} is not a valid model type")

    # get input matrices for the training, testing, and validation sets
    if model_type == "Reg":
        data_func = get_inputs_for_regression
    else:
        data_func = get_inputs_for_CNN
        
    # inputs are lists with 
    X_train, X_test, X_val, df_train, df_test, df_val = data_func(drug_abbr, config_file)
    
    if model_type == "Reg":
        print(X_train.shape, X_test.shape, X_val.shape)
    else:
        print(X_train[0].shape, X_test[0].shape, X_val[0].shape)

    # standard scale using the mean and SD of the training data
    train_mean = X_train.mean()
    train_sd = X_train.std()

    X_train = (X_train - train_mean) / train_sd
    X_test = (X_test - train_mean) / train_sd
    X_val = (X_val - train_mean) / train_sd
    
    train_summary = get_results_single_model(X_train, drug, df_train, "Train", model_type, config_file, bootstrap)
    test_summary = get_results_single_model(X_test, drug, df_test, "Test", model_type, config_file, bootstrap)
    val_summary = get_results_single_model(X_val, drug, df_val, "Validation", model_type, config_file, bootstrap)
    
    df_combined = pd.concat([train_summary, test_summary, val_summary]).reset_index(drop=True)
    
    del_cols = ["Drug", "Num_Loci"]
    
    for col in del_cols:
        if col in df_combined.columns:
            del df_combined[col]
            
    return df_combined





# python3 analysis/validation_stats.py Rifampicin RIF config_files/config_rif.yaml rpoBC_variants_10543isolates.csv
# python3 analysis/validation_stats.py Moxifloxacin MXF config_files/config_mxf_lineage.yaml gyrBA_variants_8569isolates.csv
_, drug, drug_abbr, config_file, isolate_variants_file = sys.argv

# if ".csv" in isolate_variants_file:
#     isolate_variants = pd.read_csv(os.path.join(data_path, drug_abbr, isolate_variants_file))
# else:
#     isolate_variants = pd.read_csv(os.path.join(data_path, drug_abbr, isolate_variants_file), sep="\t")

binary_thresh = yaml.safe_load(open(config_file, "r"))["binary_thresh"]

if not os.path.isfile(f"analysis/{drug}/validation/catalog_stats.csv"):
    catalog_stats = classify_using_mutation_catalog(drug_abbr, data_path, who_variants_clean, isolate_variants, binary_thresh, return_stats=["Sensitivity", "Specificity", "AUC", "AUC_PR", "Accuracy", "Balanced_Acc"])
    catalog_stats.to_csv(f"analysis/{drug}/validation/catalog_stats.csv", index=False)
else:
    catalog_stats = pd.read_csv(f"analysis/{drug}/validation/catalog_stats.csv")
    
# linear regression
if not os.path.isfile(f"{drug}/validation/Reg_stats.csv"):
    Reg_stats = get_results_all_datasets(drug, drug_abbr, config_file, "Reg", bootstrap=True)
    Reg_stats.to_csv(f"analysis/{drug}/validation/Reg_stats.csv", index=False)
else:
    Reg_stats = pd.read_csv(f"{drug}/validation/Reg_stats.csv")

# quantitative CNN
if not os.path.isfile(f"analysis/{drug}/validation/CNN_stats.csv"):
    CNN_stats = get_results_all_datasets(drug, drug_abbr, config_file, "CNN", bootstrap=True)
    CNN_stats.to_csv(f"analysis/{drug}/validation/CNN_stats.csv", index=False)
else:
    CNN_stats = pd.read_csv(f"analysis/{drug}/validation/CNN_stats.csv")
    
print(catalog_stats.shape, Reg_stats.shape, CNN_stats.shape)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
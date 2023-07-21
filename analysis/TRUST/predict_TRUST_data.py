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
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
sys.path.append("utils")
from model_utils import *
from data_utils import *
from inSilicoMut_utils import *

from sklearn.linear_model import LinearRegression

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_path = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

who_variants_clean = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")


def get_TRUST_predictions(config_file,
                          isolates_lst,
                         ):

    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]

    output_path = kwargs["output_path"]
    locus_list = kwargs["locus_list"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    include_lineage = kwargs["include_lineage"]

    # insilico validation doesn't make much sense to include lineages, unless you want to see how each lineage is predicted
    trust_dir = os.path.join(data_path, drug, "TRUST/fastas")
    df_genos = make_genotype_df(locus_list, trust_dir)
    df_genos = df_genos.loc[isolates_lst]

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[f"{locus}_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))

    # if not os.path.isfile(os.path.join(trust_dir, "pkl_sparse_data.npz")):
    X_cnn = create_X(df_genos)

    # get longest locus from the pickle file
    X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz'))

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37rv.shape[2]
    num_loci = X_h37rv.shape[-1]
    del X_h37rv

    if include_lineage:
        lineages_matrix = pd.read_csv("analysis/TRUST/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages_matrix.values)) == 2
        lineages_matrix = lineages_matrix.loc[isolates_lst]

        # check ordering
        assert sum(lineages_matrix.index.values != df_genos.index.values) == 0
        num_lineages = lineages_matrix.shape[1]
    else:
        num_lineages = 0
   
    print(f"Predicting using models with {num_lineages} lineages")
    
    cnn_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    cnn_model.load_weights(os.path.join(output_path, "best_model.h5"))
    reg_model = pickle.load(open(os.path.join(output_path, "ridge", "model.sav"), "rb"))

    # load the phenotypes dataframe to subset the input dataframe to get the training data mean and SD for standard scaling
    df_train = pd.read_csv(kwargs["phenotype_file"]).query("category=='original_train_set'")        
    X = np.load(os.path.join(output_path, "ridge", "combined_X.npy"))
    X_train = X[df_train.index.values, :]

    train_mean = X_train.mean()
    train_sd = X_train.std()
    del X_train
                             
    X_reg = get_new_aln_for_regression(isolates_lst, 
                                       locus_list,
                                       output_path,
                                       trust_dir
                                      )
                             
    if include_lineage:
        X_cnn = [X_cnn, lineages_matrix, np.zeros(len(X_cnn)), np.zeros(len(X_cnn))]
        X_reg = np.concatenate([X_reg, lineages_matrix.values], axis=1)
    else:
        X_cnn = [X_cnn, np.zeros(len(X_cnn)), np.zeros(len(X_cnn))]
        
    X_reg = (X_reg - train_mean) / train_sd
                             
    # X and df_genos are in the same order because df_genos is passed in as an argument to the function to make X
    df_genos[f"log2_MIC_CNN"] = cnn_model.predict(X_cnn, batch_size=BATCH_SIZE).flatten()
    df_genos[f"log2_MIC_Reg"] = np.squeeze(reg_model.predict(X_reg))
    
    for locus in locus_list:
        del df_genos[locus]
        del df_genos[f"{locus}_one_hot"]
        
    df_genos[f"MIC_CNN"] = np.exp2(df_genos[f"log2_MIC_CNN"])
    df_genos[f"MIC_Reg"] = np.exp2(df_genos[f"log2_MIC_Reg"])
    return drug, df_genos.reset_index().rename(columns={"index":"SampleID"})



_, config_file = sys.argv

df_trust_combined = pd.read_csv("analysis/TRUST/combined_samples_patients.csv")
isolates_lst = df_trust_combined.query("comment=='-'")["SampleID"].values
print(len(isolates_lst))

drug, TRUST_pred = get_TRUST_predictions(config_file,
                                         isolates_lst,
                                        )

TRUST_pred.to_csv(f"{data_path}/{drug}/TRUST/MIC_predictions.csv", index=False)
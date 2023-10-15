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
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
sys.path.append("utils")
from model_utils import *
from data_utils import *
from inSilicoMut_utils import *

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_path = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"


def get_TRUST_predictions(config_file,
                          isolates_lst,
                         ):

    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]

    fasta_dir = kwargs["genotype_input_directory"]
    output_path = kwargs["output_path"]
    ridge_dir = os.path.join(output_path, "ridge")
    locus_list = kwargs["locus_list"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    include_lineage = kwargs["include_lineage"]

    trust_dir = os.path.join(output_path, "validation", "TRUST")
    print(f"Saving results to {trust_dir}")

    if not os.path.isdir(trust_dir):
        os.makedirs(trust_dir)

    if not os.path.isfile(os.path.join(trust_dir.replace("_lineage", ""), "pkl_sparse_CNN.npz")):

        print("Making pickle file for TRUST data CNN model")
        df_genos = make_genotype_df(locus_list, fasta_dir)
        df_genos = df_genos.loc[isolates_lst]
    
        # Apply one-hot encoding function to get each isolate sequence
        print('making one hot encoding for...')
        for locus in locus_list:
            print("...", locus)
            lengths = [len(seq) for seq in df_genos[locus]]
            assert len(np.unique(lengths)) == 1
            df_genos[f"{locus}_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))

        df_genos.to_csv(os.path.join(trust_dir.replace("_lineage", ""), "df_genos.csv"))
        
        X_cnn = create_X(df_genos)
        sparse.save_npz(os.path.join(trust_dir.replace("_lineage", ""), "pkl_sparse_CNN.npz"), sparse.COO(X_cnn), compressed=False)
    else:
        X_cnn = sparse.load_npz(os.path.join(trust_dir.replace("_lineage", ""), 'pkl_sparse_CNN.npz')).todense()
        df_genos = pd.read_csv(os.path.join(trust_dir.replace("_lineage", ""), "df_genos.csv"), index_col=[0])
        
    # shape = num_samples x 5 x longest_locus x num_loci
    num_samples = X_cnn.shape[0]
    one_hot_encodings = X_cnn.shape[1]
    assert one_hot_encodings == 5
    longest_locus = X_cnn.shape[2]
    num_loci = X_cnn.shape[-1]

    if include_lineage:
        lineages_matrix = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        lineages_matrix = lineages_matrix.loc[isolates_lst]
        assert np.min(lineages_matrix.values) >= 0
        num_lineages = lineages_matrix.shape[1]
    else:
        num_lineages = 0
   
    print(f"Predicting using models with {num_lineages} lineages")

    if not os.path.isfile(os.path.join(trust_dir.replace("_lineage", ""), "seq_matrix_Reg.pkl")):

        print("Creating pickle file for TRUST data regression model")
        
        # features determined from the train dataset of the ridge regression. Use these exact features for the validation data inputs
        all_features = pd.read_csv(os.path.join(ridge_dir.replace("_lineage", ""), "all_feature_names.txt"), sep="\t", header=None)[0].values
        model_features = pd.read_csv(os.path.join(ridge_dir.replace("_lineage", ""), "model_feature_names.txt"), sep="\t", header=None)[0].values
    
        X_reg = []
        
        for locus_idx, locus in enumerate(locus_list):
                                     
            # convert to a dataframe so that we get the correct columns determined from the train matrix
            X_reg.append(pd.DataFrame(np.reshape(X_cnn[:, :, :, locus_idx], (num_samples, one_hot_encodings * longest_locus), order='F')))
    
        # combine the data for all the loci along the columns axis
        X_reg = pd.concat(X_reg, axis=1)
    
        # keep only columns used to train the original model
        X_reg.columns = all_features
        X_reg = X_reg[model_features]
        X_reg.to_pickle(os.path.join(trust_dir.replace("_lineage", ""), "seq_matrix_Reg.pkl"))

    X_reg = pd.read_pickle(os.path.join(trust_dir.replace("_lineage", ""), "seq_matrix_Reg.pkl")).values
                                 
    if include_lineage:
        X_cnn = [X_cnn, lineages_matrix, np.zeros(len(X_cnn)), np.zeros(len(X_cnn))]
        X_reg = np.concatenate([X_reg, lineages_matrix.values], axis=1)
    else:
        X_cnn = [X_cnn, np.zeros(len(X_cnn)), np.zeros(len(X_cnn))]

    # load the training data to get the mean and SD for standard scaling
    X_train = pd.read_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "train_seq_matrix.pkl")).values
    train_mean = X_train.mean()
    train_sd = X_train.std()
    del X_train
                             
    X_reg = (X_reg - train_mean) / train_sd

    cnn_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    cnn_model.load_weights(os.path.join(output_path, "best_model.h5"))
    reg_model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
                         
    # X and df_genos are in the same order because df_genos is passed in as an argument to the function to make X
    df_genos[f"log2_MIC_CNN"] = cnn_model.predict(X_cnn, batch_size=BATCH_SIZE).flatten()
    df_genos[f"log2_MIC_Reg"] = np.squeeze(reg_model.predict(X_reg))
    
    for locus in locus_list:
        del df_genos[locus]
        del df_genos[f"{locus}_one_hot"]
        
    df_genos[f"MIC_CNN"] = np.exp2(df_genos[f"log2_MIC_CNN"])
    df_genos[f"MIC_Reg"] = np.exp2(df_genos[f"log2_MIC_Reg"])
    df_genos.reset_index().rename(columns={"index":"SampleID"}).to_csv(f"{trust_dir}/MIC_predictions.csv", index=False)



_, config_file = sys.argv

df_trust_combined = pd.read_csv("analysis/TRUST/combined_samples_patients.csv")
isolates_lst = df_trust_combined.query("comment=='-'")["SampleID"].values
print(f"Getting CNN and regression predictions for {len(isolates_lst)} TRUST isolates\n")

get_TRUST_predictions(config_file, isolates_lst)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle

import Bio.SeqUtils
import Bio.Data
from Bio import SeqIO
from Bio.Seq import Seq

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlepad'] = 10
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

# load all utils functions
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
sys.path.append("utils")
from data_utils import *
from model_utils import *
from inSilicoMut_utils import *

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_path = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")
who_variants["gene"] = [mut.split("_")[0] for mut in who_variants["mutation"].values]

coll_2014 = pd.read_csv("/home/sak0914/who-analysis/data/coll2014_SNP_scheme.tsv", sep="\t")
coll_2014["lineage"] = coll_2014["#lineage"].str.replace("lineage", "")
del coll_2014["#lineage"]

scaler = StandardScaler()



def get_insilico_mutation_predictions(config_file,
                                      mutations_file,
                                      data_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs",
                                      fasta_dir="inSilico_analysis",
                                     ):

    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]
    locus_list = kwargs["locus_list"]
    drug = kwargs["drug"]
    include_lineage = kwargs["include_lineage"]

    output_path = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
        
    # argument = directory that contains the fasta file
    inSilico_dir = os.path.join(data_dir, drug, fasta_dir)

    # the additional new strains to predict MICs for
    keep_strains = pd.read_csv(os.path.join(inSilico_dir, mutations_file), sep="\t", header=None)[0].values
    
    # the names are abolute paths to the VCF files, so just get the ID
    # remove all possible file extensions from the mutation names
    keep_strains = [os.path.basename(fName).replace(".eff", "").replace(".vcf", "") for fName in keep_strains]

    # insilico validation doesn't make much sense to include lineages, unless you want to see how each lineage is predicted
    if include_lineage:
        lineages_mat = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        num_lineages = lineages_mat.shape[1]
        del lineages_mat
    else:
        num_lineages = 0

    print(f"Predicting with models trained using {num_lineages} lineage SNPs")

    df_genos = make_genotype_df(locus_list, inSilico_dir)
    
    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[f"{locus}_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))

    # X_h37rv_Reg = df_genos.loc["MT_H37Rv"]
    df_genos = df_genos.loc[keep_strains]

    X_CNN = create_X(df_genos)
    print(f"Predicting MICs for {len(X_CNN)} in silico sequences")
        
    # get longest locus from the pickle file
    X_h37rv_CNN = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz')).todense()

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37rv_CNN.shape[2]
    num_loci = X_h37rv_CNN.shape[-1]

    CNN_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    CNN_model.load_weights(os.path.join(output_path, "best_model.h5"))
    
    X_Reg = get_new_aln_for_regression(keep_strains,
                                       locus_list,
                                       output_path,
                                       inSilico_dir
                                      )

    reg_model = pickle.load(open(os.path.join(output_path, "ridge", "model.sav"), "rb"))

    # get prediction for H37Rv
    if include_lineage:
        h37Rv_pred_CNN = CNN_model.predict([X_h37rv_CNN, np.zeros((X_h37rv_CNN.shape[0], num_lineages)), np.zeros(X_h37rv_CNN.shape[0]), np.zeros(X_h37rv_CNN.shape[0])])
    else:
        h37Rv_pred_CNN = CNN_model.predict([X_h37rv_CNN, np.zeros(X_h37rv_CNN.shape[0]), np.zeros(X_h37rv_CNN.shape[0])])
    # h37Rv_pred_Reg = reg_model.predict()
    
    # X and df_genos are in the same order because df_genos is passed in as an argument to the function to make X
    if include_lineage:
        df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, np.zeros((X_CNN.shape[0], num_lineages)), np.zeros(X_CNN.shape[0]), np.zeros(X_CNN.shape[0])], batch_size=BATCH_SIZE).flatten()
        df_genos["Reg_log_MIC"] = reg_model.predict(scaler.fit_transform(np.concatenate([X_Reg, np.zeros((X_Reg.shape[0], num_lineages))], axis=1)))
    else:
        df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, np.zeros(X_CNN.shape[0]), np.zeros(X_CNN.shape[0])], batch_size=BATCH_SIZE).flatten()
        df_genos["Reg_log_MIC"] = reg_model.predict(scaler.fit_transform(X_Reg))
    
    # return dataframe with predictions
    prev_len = len(df_genos)
    df_genos = df_genos.merge(who_variants.query("drug==@drug")[["mutation", "confidence", "genome_index"]], left_index=True, right_on="mutation")
    assert len(df_genos.mutation.unique()) == prev_len
    
    for locus in locus_list:
        del df_genos[locus]
        del df_genos[f"{locus}_one_hot"]

    # add H37Rv prediction to df_genos
    df_genos.loc[-1, ["mutation", "CNN_log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred_CNN)]
    return df_genos.reset_index(drop=True)



def get_lineage_MIC_predictions(drug, 
                                config_file,
                               ):
    
    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]

    output_path = kwargs["output_path"]
    print(output_path)
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    
    lineages = coll_2014.copy()
    lineages["Count"] = 1
    lineages = lineages.pivot(index="lineage", columns="position", values="Count").fillna(0).astype(int)
    
    num_lineages = lineages.shape[1]
    assert np.min(lineages.sum(axis=1).values) == 1
    assert np.max(lineages.sum(axis=1).values) == 1
    
    # # add reference lineage, which is all 0s
    # lineages = pd.concat([lineages, pd.DataFrame(np.zeros((1, lineages.shape[1])), columns=lineages.columns, index=["MT_H37Rv"])])
                                   
    # get longest locus from the pickle file and the number of inputs for regression from the .npy file
    X_h37Rv_CNN = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz')).todense()
    X_Reg = np.load(os.path.join(output_path, "ridge/combined_X.npy"))
    
    # copy so that there is one reference sequence for each lineage
    X_CNN = np.repeat(X_h37Rv_CNN, lineages.shape[0], axis=0)

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37Rv_CNN.shape[2]
    num_loci = X_h37Rv_CNN.shape[-1]

    CNN_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    CNN_model.load_weights(os.path.join(output_path, "best_model.h5"))
    # reg_model = pickle.load(open(os.path.join(output_path, "ridge", "model.sav"), "rb"))

    # no lineage SNPs for H37Rv, so put in vectors of 0 for lineage SNPs (and also lower and upper bounds, but those don't matter)
    h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, np.zeros(len(lineages.values)).reshape(1, -1), np.zeros(X_h37Rv_CNN.shape[0]), np.zeros(X_h37Rv_CNN.shape[0])])
    # h37Rv_pred_Reg = reg_model.predict()

    # X_Reg = np.zeros((len(lineages.values, X_Reg.shape[1])))

    # # X and df_genos are in the same order because df_genos is passed in as an argument to the function to make X
    # predicted_mics = best_model.predict([X, lineages.values, np.zeros(X.shape[0]), np.zeros(X.shape[0])], batch_size=BATCH_SIZE).flatten()
    
    df_results = pd.DataFrame({"Lineage": lineages.index.values,
                               "CNN_log_MIC": CNN_model.predict([X_CNN, lineages.values, np.zeros(X_CNN.shape[0]), np.zeros(X_CNN.shape[0])], batch_size=BATCH_SIZE).flatten(),
                               # "Reg_log_MIC": reg_model.predict(scaler.fit_transform(X_Reg))
                              })
    # ref_pred = df_results.query("Lineage=='MT_H37Rv'")["pred_log2_MIC"].values[0]    
    # add H37Rv prediction to df_genos
    df_results.loc[-1, ["Lineage", "CNN_log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred_CNN)]
    return df_results.reset_index(drop=True)




# Moxifloxacin config_files/config_mxf.yaml gyrBA_WHO_mutations.txt gyrBA_WHO_predictions.csv
_, drug, config_file, WHO_mutations_file, out_file = sys.argv

if not os.path.isdir(f"analysis/{drug}/insilico_mutagenesis"):
    os.makedirs(f"analysis/{drug}/insilico_mutagenesis")

if os.path.isfile(WHO_mutations_file):
    df_pred = get_insilico_mutation_predictions(config_file,
                                                WHO_mutations_file,
                                               )
    df_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/{out_file}", index=False)

df_lineage_pred = get_lineage_MIC_predictions(drug, config_file)
df_lineage_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/lineage_predictions.csv", index=False)
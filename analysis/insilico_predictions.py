import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, shutil

import Bio.SeqUtils
import Bio.Data
from Bio import SeqIO
from Bio.Seq import Seq

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlepad'] = 10
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

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


# inclusive
# 759611 767320
# _, START, END = sys.argv

# START = int(START)
# END = int(END)


def get_insilico_mutation_predictions(drug, 
                                     locus_list, 
                                     config_file,
                                     mutations_file,
                                     data_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs",
                                     fasta_dir="inSilico_analysis",
                                    ):

    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]

    output_path = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]

    # insilico validation doesn't make much sense to include lineages, unless you want to see how each lineage is predicted
    num_lineages = 0
        
    # argument = directory that contains the fasta file
    inSilico_dir = os.path.join(data_dir, drug, fasta_dir)
    df_genos = make_genotype_df(locus_list, inSilico_dir)
    
    # the additional new strains to predict MICs for
    keep_strains = pd.read_csv(os.path.join(inSilico_dir, mutations_file), sep="\t", header=None)[0].values
    
    # the names are abolute paths to the VCF files, so just get the ID
    # WHO mutations have a "." in them, so need to keep all the split values until the last one, then join
    keep_strains = [".".join(os.path.basename(fName).split(".")[:-1]) for fName in keep_strains]
    df_genos = df_genos.loc[keep_strains]

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[f"{locus}_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))

    X = create_X(df_genos)
    print(f"Predicting MICs for {len(X)} in silico sequences")
        
    # get longest locus from the pickle file
    X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz')).todense()

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37rv.shape[2]
    num_loci = X_h37rv.shape[-1]

    best_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    best_model.load_weights(os.path.join(output_path, "best_model.h5"))

    # print prediction for H37Rv
    h37Rv_pred = best_model.predict([X_h37rv, np.zeros(X_h37rv.shape[0]), np.zeros(X_h37rv.shape[0])])
    
    # X and df_genos are in the same order because df_genos is passed in as an argument to the function to make X
    predicted_mics = best_model.predict([X, np.zeros(X.shape[0]), np.zeros(X.shape[0])], batch_size=BATCH_SIZE).flatten()
    df_genos["log_MIC"] = predicted_mics
    
    # return dataframe with predictions
    prev_len = len(df_genos)
    df_genos = df_genos.merge(who_variants.query("drug==@drug")[["mutation", "confidence", "genome_index"]], left_index=True, right_on="mutation")
    assert len(df_genos.mutation.unique()) == prev_len
    
    for locus in locus_list:
        del df_genos[locus]
        del df_genos[f"{locus}_one_hot"]

    # add H37Rv prediction to df_genos
    df_genos.loc[-1, ["mutation", "log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred)]
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
    
    # add reference lineage, which is all 0s
    lineages = pd.concat([lineages, pd.DataFrame(np.zeros((1, lineages.shape[1])), columns=lineages.columns, index=["MT_H37Rv"])])

    # get longest locus from the pickle file
    X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz')).todense()
    
    # copy so that there is one reference sequence for each lineage
    X = np.repeat(X_h37rv, lineages.shape[0], axis=0)

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37rv.shape[2]
    num_loci = X_h37rv.shape[-1]
    del X_h37rv

    best_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    best_model.load_weights(os.path.join(output_path, "best_model.h5"))

    # X and df_genos are in the same order because df_genos is passed in as an argument to the function to make X
    predicted_mics = best_model.predict([X, lineages.values, np.zeros(X.shape[0]), np.zeros(X.shape[0])], batch_size=BATCH_SIZE).flatten()
    
    df_results = pd.DataFrame({"Lineage": lineages.index.values, "pred_log2_MIC": predicted_mics}).sort_values("pred_log2_MIC", ascending=False).reset_index(drop=True)
    ref_pred = df_results.query("Lineage=='MT_H37Rv'")["pred_log2_MIC"].values[0]    
    return df_results



_, drug, drug_abbr, locus = sys.argv


# df_pred = get_insilico_mutation_predictions(drug_abbr",
#                                             [locus],
#                                             "config_rif.yaml",
#                                             "rpoBC_WHO_mutations.txt"
#                                            )

df_lineage_pred = get_lineage_MIC_predictions(drug_abbr, 
                                              "config_mxf_lineage.yaml",
                                             )

# df_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/{_locus}WHO_predictions.csv", index=False)
df_lineage_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/lineage_predictions.csv", index=False)
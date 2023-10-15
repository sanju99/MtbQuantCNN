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
    include_peptide_length = kwargs["include_peptide_length"]

    output_path = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
        
    # argument = directory that contains the fasta file
    inSilico_dir = os.path.join(data_dir, drug, fasta_dir)
    
    # for cases when you just want the lineage predictions (i.e. LEVO)
    if not os.path.isfile(os.path.join(inSilico_dir, mutations_file)):
        return None

    # the additional new strains to predict MICs for
    keep_mutations = pd.read_csv(os.path.join(inSilico_dir, mutations_file), sep="\t", header=None)[0].values
    
    # the names are abolute paths to the VCF files, so just get the ID
    # remove all possible file extensions from the mutation names
    keep_mutations = [os.path.basename(fName).replace(".eff", "").replace(".vcf", "") for fName in keep_mutations]

    # MLP block: lineages and peptide lengths
    additional_data_len = 0
    
    if include_lineage:
        lineages_mat = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        num_lineages = lineages_mat.shape[1]
        additional_data_len += num_lineages
        del lineages_mat

    if include_peptide_length:
        
        # check if the file of peptide lengths for all mutations exists. If not, create it
        if not os.path.isfile(os.path.join(inSilico_dir, file_prefix + "_peptide_lengths.csv")):

            WHO_catalog_variants = pd.read_csv(os.path.join(inSilico_dir, file_prefix + "_nucleotide_variants.csv"))

            # this function also add H37Rv (named MT_H37Rv)
            WHO_mutations_peptide_lengths = get_peptide_lengths_WHO_mutations(drug, [file_prefix.replace('_WHO', '')], locus_list, data_dir, WHO_catalog_variants)
            WHO_mutations_peptide_lengths.to_csv(os.path.join(inSilico_dir, file_prefix + "_peptide_lengths.csv"))

        WHO_mutations_peptide_lengths = pd.read_csv(os.path.join(inSilico_dir, file_prefix + "_peptide_lengths.csv"), index_col=[0])

        # separate H37Rv to keep the code readable
        H37Rv_peptide_lengths = np.reshape(WHO_mutations_peptide_lengths.loc["MT_H37Rv"].values, (1, -1))
        
        # match order to the sequence inputs. keep_mutations doesn't contain MT_H37Rv anyway
        WHO_mutations_peptide_lengths = WHO_mutations_peptide_lengths.loc[keep_mutations]
        additional_data_len += WHO_mutations_peptide_lengths.shape[1]
        
    print(f"Predicting with models trained using an MLP block with {additional_data_len} features")

    df_genos = make_genotype_df(locus_list, inSilico_dir)
    df_genos = df_genos.loc[keep_mutations]

    if os.path.isfile(os.path.join(inSilico_dir, "pkl_sparse_WHO.npz")):
        X_CNN = sparse.load_npz(os.path.join(inSilico_dir, "pkl_sparse_WHO.npz")).todense()
    else:
        # Apply one-hot encoding function to get each isolate sequence
        print('making one hot encoding for...')
        for locus in locus_list:
            print("...", locus)
            lengths = [len(seq) for seq in df_genos[locus]]
            assert len(np.unique(lengths)) == 1
            df_genos[f"{locus}_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))        
    
        X_CNN = create_X(df_genos)
        sparse.save_npz(os.path.join(inSilico_dir, "pkl_sparse_WHO.npz"), sparse.COO(X_CNN), compressed=True)
    
    print(f"Predicting MICs for {len(X_CNN)} in silico sequences")
        
    # get longest locus from the pickle file
    X_h37rv_CNN = sparse.load_npz(os.path.join(output_path.replace("_lineage", "").replace("_peptide", ""), 'pkl_sparse_ref.npz')).todense()

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37rv_CNN.shape[2]
    num_loci = X_h37rv_CNN.shape[-1]

    # get regularization parameter
    losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))
    
    # get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
    losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
    select_alpha = np.round(losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0], 6)  # not sure why, but some of the alphas are like 0.999999999 instead of 1
    print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}\n")

    # use bounded_loss = False so that you don't have to put in empty bounds arrays. bounded_loss is only relevant for training
    CNN_model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss=False, filter_size=filter_size, reg_strength=select_alpha)
    CNN_model.load_weights(os.path.join(output_path, "best_model.h5"))

    if additional_data_len == 0:
        h37Rv_pred_CNN = CNN_model.predict([X_h37rv_CNN])
        df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN], batch_size=BATCH_SIZE).flatten()
    else:
        if include_lineage:
            if include_peptide_length:

                h37Rv_pred_CNN = CNN_model.predict([X_h37rv_CNN, 
                                                    np.concatenate([np.zeros((X_h37rv_CNN.shape[0], num_lineages)), 
                                                                    H37Rv_peptide_lengths
                                                                   ], axis=1)])
                df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, 
                                                             np.concatenate([np.zeros((X_CNN.shape[0], num_lineages)), 
                                                                             WHO_mutations_peptide_lengths.values
                                                                            ], axis=1)
                                                            ], batch_size=BATCH_SIZE).flatten()
            else:
                h37Rv_pred_CNN = CNN_model.predict([X_h37rv_CNN, np.zeros((X_h37rv_CNN.shape[0], additional_data_len))])
                df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, np.zeros((X_CNN.shape[0], additional_data_len))], batch_size=BATCH_SIZE).flatten()
        else:
            # SAME TODO AS ABOVE
            if include_peptide_length:
                h37Rv_pred_CNN = CNN_model.predict([X_h37rv_CNN, H37Rv_peptide_lengths])
                df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, WHO_mutations_peptide_lengths.values], batch_size=BATCH_SIZE).flatten()
                
    # return dataframe with predictions
    prev_len = len(df_genos)
    df_genos = df_genos.merge(who_variants.query("drug==@drug")[["mutation", "confidence", "genome_index"]], left_index=True, right_on="mutation")
    assert len(df_genos.mutation.unique()) == prev_len
    
    for locus in locus_list:
        if locus in df_genos.columns:
            del df_genos[locus]
        if f"{locus}_one_hot" in df_genos.columns:
            del df_genos[f"{locus}_one_hot"]

    # add H37Rv prediction to df_genos
    df_genos.loc[-1, ["mutation", "CNN_log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred_CNN)]
    return df_genos.reset_index(drop=True)



def get_lineage_MIC_predictions(config_file,
                                data_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs",
                                fasta_dir="inSilico_analysis"
                               ):
    
    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]
    output_path = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    inSilico_dir = os.path.join(data_dir, drug, fasta_dir)

    additional_data_len = 0
    lineages = coll_2014.copy()
    lineages["Count"] = 1
    lineages = lineages.pivot(index="lineage", columns="position", values="Count").fillna(0).astype(int)
    
    num_lineages = lineages.shape[1]
    additional_data_len += num_lineages
    assert np.min(lineages.sum(axis=1).values) == 1
    assert np.max(lineages.sum(axis=1).values) == 1
                      
    # get longest locus from the pickle file and the number of inputs for regression from the .npy file
    X_h37Rv_CNN = sparse.load_npz(os.path.join(output_path.replace("_lineage", ""), 'pkl_sparse_ref.npz')).todense()

    # copy so that there is one reference sequence for each lineage
    X_CNN = np.repeat(X_h37Rv_CNN, lineages.shape[0], axis=0)

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37Rv_CNN.shape[2]
    num_loci = X_h37Rv_CNN.shape[-1]

    if include_peptide_length:
        
        WHO_mutations_peptide_lengths = pd.read_csv(os.path.join(inSilico_dir, file_prefix + "_peptide_lengths.csv"), index_col=[0])

        # for the lineage model, you just need the H37Rv lengths because the model is testing the effects of single lineage SNPs, keeping protein lengths constant
        H37Rv_peptide_lengths = np.reshape(WHO_mutations_peptide_lengths.loc["MT_H37Rv"].values, (1, -1))
        additional_data_len += WHO_mutations_peptide_lengths.shape[1]
        del WHO_mutations_peptide_lengths

        # copy so that the H37Rv protein lengths are used with all lineage SNPs
        WHO_mutations_peptide_lengths = np.repeat(H37Rv_peptide_lengths, lineages.shape[0], axis=0)
        
    print(f"Predicting with models trained using an MLP block with {additional_data_len} features")

    # get regularization parameter
    losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))
    
    # get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
    losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
    select_alpha = np.round(losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0], 6)  # not sure why, but some of the alphas are like 0.999999999 instead of 1
    print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}\n")

    # use bounded_loss = False so that you don't have to put in empty bounds arrays. bounded_loss is only relevant for training
    CNN_model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss=False, filter_size=filter_size, reg_strength=select_alpha)
    CNN_model.load_weights(os.path.join(output_path, "best_model.h5"))

    df_results = pd.DataFrame({"Lineage": lineages.index.values})

    if include_peptide_length:
        h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, 
                                            np.concatenate([np.zeros((X_h37Rv_CNN.shape[0], num_lineages)), 
                                                            H37Rv_peptide_lengths
                                                           ], axis=1)])
        df_results["CNN_log_MIC"] = CNN_model.predict([X_CNN, 
                                                     np.concatenate([lineages.values, 
                                                                     WHO_mutations_peptide_lengths
                                                                    ], axis=1)
                                                    ], batch_size=BATCH_SIZE).flatten()
    else:
        h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, np.zeros((X_h37Rv_CNN.shape[0], additional_data_len))])
        df_results["CNN_log_MIC"] = CNN_model.predict([X_CNN, lineages.values], batch_size=BATCH_SIZE).flatten()

    # add H37Rv prediction to df_genos
    df_results.loc[-1, ["Lineage", "CNN_log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred_CNN)]
    return df_results.reset_index(drop=True)


# python3 -u analysis/insilico_predictions.py config_files/config_mxf.yaml gyrBA_WHO
# python3 -u analysis/insilico_predictions.py config_files/config_pza_peptide.yaml pncA_WHO

_, config_file, file_prefix = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
include_lineage = kwargs["include_lineage"]
include_peptide_length = kwargs["include_peptide_length"]

suffix = ""

# lineage first, then peptide suffixed
if include_lineage:
    suffix += "_lineage"

if include_peptide_length:
    suffix += "_peptide"

out_file = file_prefix + "_predictions" + suffix + ".csv"
print(f"Saving CNN predictions to {out_file}")

if not os.path.isdir(f"analysis/{drug}/insilico_mutagenesis"):
    os.makedirs(f"analysis/{drug}/insilico_mutagenesis")

df_pred = get_insilico_mutation_predictions(config_file,
                                            f"{file_prefix}_mutations.txt",
                                           )

if df_pred is not None:
    df_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/{out_file}", index=False)

if include_lineage:
    df_lineage_pred = get_lineage_MIC_predictions(config_file)
    
    if include_peptide_length:
        df_lineage_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/lineage_predictions_peptide.csv", index=False)
    else:
        df_lineage_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/lineage_predictions.csv", index=False)
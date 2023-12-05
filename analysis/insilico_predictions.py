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
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

coll_2014 = pd.read_csv("/home/sak0914/who-analysis/data/coll2014_SNP_scheme.tsv", sep="\t")
coll_2014["lineage"] = coll_2014["#lineage"].str.replace("lineage", "")
del coll_2014["#lineage"]


def get_insilico_mutation_predictions(config_file):

    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]
    locus_list = kwargs["locus_list"]
    drug = kwargs["drug"]
    include_lineage = kwargs["include_lineage"]

    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    output_path = kwargs["output_path"]
    
    # naming consistency
    output_path = output_path.replace("_lineage", "").replace("_peptide", "")
    
    if include_peptide_length:
        output_path += "_peptide"
        
    if include_lineage:
        output_path += "_lineage"
        
    inSilico_dir = os.path.join(data_dir, drug, "inSilico_analysis")
    
    # for cases when you just want the lineage predictions (i.e. LEVO)
    if not os.path.isfile(os.path.join(inSilico_dir, f"{file_prefix}_nucleotide_variants.csv")):
        return None

    # the additional new strains to predict MICs for. Don't sanitize the variant names until after creating the peptide lengths dataframe so that the names are consistent
    WHO_catalog_variants = pd.read_csv(os.path.join(inSilico_dir, f'{file_prefix}_nucleotide_variants.csv'))

        # MLP block: lineages and peptide lengths
    additional_data_len = 0
    
    if include_lineage:
        lineages_mat = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        num_lineages = lineages_mat.shape[1]
        additional_data_len += num_lineages
        del lineages_mat

    if include_peptide_length:
        
        # check if the file of peptide lengths for all mutations exists. If not, create it
        if not os.path.isfile(os.path.join(inSilico_dir, f"{file_prefix}_peptide_lengths.csv")):

            # this function also add H37Rv (named MT_H37Rv)
            WHO_mutations_peptide_lengths = get_peptide_lengths_WHO_mutations(drug, locus_list, [file_prefix.replace('_WHO', '')], data_dir, WHO_catalog_variants)
            WHO_mutations_peptide_lengths.to_csv(os.path.join(inSilico_dir, f"{file_prefix}_peptide_lengths.csv"))

        WHO_mutations_peptide_lengths = pd.read_csv(os.path.join(inSilico_dir, f"{file_prefix}_peptide_lengths.csv"), index_col=[0])

        # separate H37Rv to keep the code readable
        H37Rv_peptide_lengths = np.reshape(WHO_mutations_peptide_lengths.loc["MT_H37Rv"].values, (1, -1))
        
        # match order to the sequence inputs
        WHO_mutations_peptide_lengths = WHO_mutations_peptide_lengths.loc[WHO_catalog_variants["mutation"].values]
        WHO_mutations_peptide_lengths.index = [val.replace('.', '_').replace('*', '+') for val in WHO_mutations_peptide_lengths.index.values]
        additional_data_len += WHO_mutations_peptide_lengths.shape[1]

    # sanitize the names to be consistent with the file names (and therefore, the .npz file)
    WHO_catalog_variants["mutation"] = [val.replace('.', '_').replace('*', '+') for val in WHO_catalog_variants['mutation'].values]
        
    print(f"Predicting with models trained using an MLP block with {additional_data_len} features")

    df_genos = make_genotype_df(locus_list, inSilico_dir)
    df_genos = df_genos.loc[WHO_catalog_variants["mutation"].values]

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
    X_h37Rv_CNN = sparse.load_npz(os.path.join(output_path.replace("_lineage", "").replace("_peptide", ""), 'pkl_sparse_ref.npz')).todense()

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37Rv_CNN.shape[2]
    num_loci = X_h37Rv_CNN.shape[-1]

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
        h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN])
        df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN], batch_size=BATCH_SIZE).flatten()
    else:
        if include_lineage:
            if include_peptide_length:

                h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, 
                                                    np.concatenate([np.zeros((X_h37Rv_CNN.shape[0], num_lineages)), 
                                                                    H37Rv_peptide_lengths
                                                                   ], axis=1)])
                df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, 
                                                             np.concatenate([np.zeros((X_CNN.shape[0], num_lineages)), 
                                                                             WHO_mutations_peptide_lengths.values
                                                                            ], axis=1)
                                                            ], batch_size=BATCH_SIZE).flatten()
            else:
                h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, np.zeros((X_h37Rv_CNN.shape[0], additional_data_len))])
                df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, np.zeros((X_CNN.shape[0], additional_data_len))], batch_size=BATCH_SIZE).flatten()
        else:
            # SAME TODO AS ABOVE
            if include_peptide_length:
                h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, H37Rv_peptide_lengths])
                df_genos["CNN_log_MIC"] = CNN_model.predict([X_CNN, WHO_mutations_peptide_lengths.values], batch_size=BATCH_SIZE).flatten()
                
    # return dataframe with predictions
    prev_len = len(df_genos)
    df_genos = df_genos.merge(WHO_catalog_variants, left_index=True, right_on="mutation")
    assert df_genos.mutation.nunique() == prev_len
    
    for locus in locus_list:
        if locus in df_genos.columns:
            del df_genos[locus]
        if f"{locus}_one_hot" in df_genos.columns:
            del df_genos[f"{locus}_one_hot"]

    # add H37Rv prediction to df_genos
    df_genos.loc[-1, ["mutation", "CNN_log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred_CNN)]
    return df_genos.reset_index(drop=True)



def get_lineage_MIC_predictions(config_file):
    '''
    Get predictions for H37Rv + individual lineage SNPs (sanity check to make sure that no single synonymous SNP predicted resistance)
    '''
    
    # load in previously trained model
    kwargs = yaml.safe_load(open(config_file, "r"))
    drug = kwargs["drug"]
    filter_size = kwargs["filter_size"]
    N_epochs = kwargs["N_epochs"]
    BATCH_SIZE = kwargs["batch_size"]
    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    locus_list = kwargs["locus_list"]
    original_fasta_dir = kwargs["genotype_input_directory"]
    output_path = kwargs["output_path"]
    
    # naming consistency
    output_path = output_path.replace("_lineage", "").replace("_peptide", "")
    
    if include_peptide_length:
        output_path += "_peptide"
        
    if include_lineage:
        output_path += "_lineage"

    inSilico_dir = os.path.join(data_dir, drug, "inSilico_analysis")

    additional_data_len = 0
    lineages = coll_2014.copy()
    lineages["Count"] = 1
    lineages = lineages.pivot(index="lineage", columns="position", values="Count").fillna(0).astype(int)
    
    num_lineages = lineages.shape[1]
    additional_data_len += num_lineages
    assert np.min(lineages.sum(axis=1).values) == 1
    assert np.max(lineages.sum(axis=1).values) == 1
                      
    # get longest locus from the pickle file and the number of inputs for regression from the .npy file
    X_h37Rv_CNN = sparse.load_npz(os.path.join(output_path.replace("_lineage", "").replace("_peptide", ""), 'pkl_sparse_ref.npz')).todense()

    # copy so that there is one reference sequence for each lineage
    X_CNN = np.repeat(X_h37Rv_CNN, lineages.shape[0], axis=0)

    # shape = 1 x 5 x longest_locus x num_loci
    longest_locus = X_h37Rv_CNN.shape[2]
    num_loci = X_h37Rv_CNN.shape[-1]

    if include_peptide_length:

        # only get the H37Rv peptide legnths. For lineage predictions, everything should be kept constant except for the lineage SNPs
        H37Rv_peptide_lengths, _ = make_H37Rv_CDS_length_df(locus_list, original_fasta_dir)
        H37Rv_peptide_lengths = np.reshape(H37Rv_peptide_lengths["Length"].values, (1, -1))

        # so both the sequence input and peptide lengths should be the same (H37Rv values) for all 62 lineage SNPs
        H37Rv_peptide_lengths_repeat = np.repeat(H37Rv_peptide_lengths, lineages.shape[0], axis=0)
        additional_data_len += H37Rv_peptide_lengths.shape[1]

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

    # NEED TO INCLUDE THE ASTYPE(FLOAT) PART FOR THE PEPTIDE MODELS BECAUSE OTHERWISE YOU GET THIS WEIRD TENSOR TYPE ERROR:
    # ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float)
    if include_peptide_length:
        h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, 
                                            np.concatenate([np.zeros((X_h37Rv_CNN.shape[0], num_lineages)), 
                                                            H37Rv_peptide_lengths
                                                           ], axis=1).astype(float)])
        df_results["CNN_log_MIC"] = CNN_model.predict([X_CNN, 
                                                       np.concatenate([lineages.values, 
                                                                       H37Rv_peptide_lengths_repeat
                                                                    ], axis=1).astype(float)
                                                      ], batch_size=BATCH_SIZE).flatten()
    else:
        h37Rv_pred_CNN = CNN_model.predict([X_h37Rv_CNN, np.zeros((X_h37Rv_CNN.shape[0], additional_data_len))])
        df_results["CNN_log_MIC"] = CNN_model.predict([X_CNN, lineages.values], batch_size=BATCH_SIZE).flatten()

    # add H37Rv prediction to df_genos
    df_results.loc[-1, ["Lineage", "CNN_log_MIC"]] = ["MT_H37Rv", np.squeeze(h37Rv_pred_CNN)]
    return df_results.reset_index(drop=True)


_, config_file, include_peptide_length, file_prefix = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
include_lineage = kwargs["include_lineage"]

if include_peptide_length.upper() == "TRUE":
    include_peptide_length = True
else:
    include_peptide_length = False

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

df_pred = get_insilico_mutation_predictions(config_file)

# non-LEVO drugs
if df_pred is not None:
    df_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/{out_file}", index=False)

if include_lineage:
    df_lineage_pred = get_lineage_MIC_predictions(config_file)
    
    if include_peptide_length:
        df_lineage_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/lineage_predictions_peptide.csv", index=False)
    else:
        df_lineage_pred.to_csv(f"analysis/{drug}/insilico_mutagenesis/lineage_predictions.csv", index=False)
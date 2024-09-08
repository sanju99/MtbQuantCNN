import sys, argparse, glob, os, yaml, sparse, tracemalloc, pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")

# utils files are in the utils directory
sys.path.append("utils")
from data_utils import *
from analysis_utils import *
from model_utils import *
from dataloader import MtbGeneDataset

from Bio import SeqIO, Seq
import warnings
warnings.filterwarnings("ignore")

# don't log warnings like compiled metrics aren't available because they clog up the logs file
tf.get_logger().setLevel('ERROR')

# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

# boolean argument for including lineage SNPs, default value False. If you include the flag, it is considered True
parser.add_argument('--lineage', action='store_true', help='Flag to add lineage SNPs to model')

# boolean argument for including peptide lengths, default value False. If you include the flag, it is considered True
parser.add_argument('--peptide-lengths', dest='peptide_lengths', action='store_true', help='Flag to add peptide lengths to model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--tier2', action='store_true', help='Flag to add tier 2 loci to the model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--amino-acid', dest='amino_acid', action='store_true', help='Flag to add amino acid biophysical properties to the model')

parser.add_argument('--train', dest='train_model', action='store_true', help='If true, train a new model. If false, just create input matrices.')

parser.add_argument('--patience', default=200, type=int, help='Number of patience epochs for model training')

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')


cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
train_model = cmd_line_args.train_model
patience_epochs = cmd_line_args.patience
AF_thresh = cmd_line_args.AF_thresh

# use the non-75% AF thresh for the test data generator if specified
if AF_thresh > 1:
    AF_thresh /= 100
    
kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
tier1_loci = kwargs["tier1_loci"]

if include_tier2:
    tier2_loci = kwargs["tier2_loci"]
else:
    tier2_loci = []

filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]

if 'output_path' in kwargs.keys():
    output_path = kwargs["output_path"]
else:
    output_path = f"/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/{drug}"

loss_type = "L1"
binary = False
bounded_loss = True
N_epochs = 10000

df_phenos = pd.read_csv(phenotype_file)
df_train_val = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)
df_test = df_phenos.query("category == 'test_set'").reset_index(drop=True)

seq_data_path = output_path

if include_peptide_lengths:
    output_path += "_peptide"
    
if include_lineage:
    output_path += "_lineage"

if include_tier2:
    output_path += "_tier2"

if include_amino_acid_properties:
    output_path += "_amino_acid"

if not os.path.isdir(seq_data_path):
    os.makedirs(seq_data_path)

cv_dir = os.path.join(output_path, "cross_validation")

if not os.path.isdir(cv_dir):
    os.makedirs(cv_dir)

if AF_thresh != 0.75:
    test_seq_data_path = os.path.join(seq_data_path, f"AF_thresh_{int(AF_thresh*100)}")

    if not os.path.isdir(test_seq_data_path):
        os.makedirs(test_seq_data_path)
    
    # test_predictions_fName = os.path.join(output_path, f"test_predictions_AF_thresh_{int(AF_thresh*100)}.csv")
    test_results_fName = os.path.join(output_path, f"results_AF_thresh_{int(AF_thresh*100)}.csv")
else:
    test_seq_data_path = seq_data_path
    # test_predictions_fName = os.path.join(output_path, "test_predictions.csv")
    test_results_fName = os.path.join(output_path, "results.csv")

print(f"Saving results to {output_path}")

# input files that need to be present
if not os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_train_val.npz")) or not os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_test.npz")):
    print("Making nucleotide one-hot encodings files...\n")

    # make for all loci
    make_nucleotide_matrices(drug, 
                             kwargs["tier1_loci"] + kwargs['tier2_loci'],
                             seq_data_path,
                             df_phenos,
                             genotype_input_directory,
                             split_groups=True
                            )

# make other AF matrices
if AF_thresh != 0.75 and not os.path.isfile(os.path.join(test_seq_data_path, 'pkl_sparse_test.npz')):
    
    make_nucleotide_matrices(drug, 
                     kwargs["tier1_loci"] + kwargs['tier2_loci'],
                     test_seq_data_path,
                     df_phenos,
                     os.path.join(os.path.dirname(genotype_input_directory), f"AF_thresh_{int(AF_thresh*100)}", "fastas"), # remove fastas from genotype_input_directory
                     split_groups=True
                    )

    # delete the train_val file because we only need the test file
    os.remove(os.path.join(test_seq_data_path, 'pkl_sparse_train_val.npz'))

# make peptide lengths dataframe if it has not already been created
if include_peptide_lengths and not os.path.isfile(os.path.join(seq_data_path, "gene_peptide_lengths.csv")):

    print("Creating gene peptide lengths dataframe")

    if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
        all_loci_seq = create_all_loci_matrices(config_file)
        pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))

    # make dataframe for both tiers of genes, then subset them appropriately
    locus_peptide_lengths = make_CDS_length_df(drug, kwargs["tier1_loci"] + kwargs["tier2_loci"], genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
    
    # keep index because that's the samples column
    locus_peptide_lengths.to_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"))


if include_amino_acid_properties:

    genes_lst = get_genes_lst(kwargs["tier1_loci"] + kwargs['tier2_loci'])
    print(f"Genes list: {','.join(genes_lst)}")

    # make the amino acid property files if they don't exist
    if not os.path.isfile(os.path.join(seq_data_path, "pkl_AA_train_val.npy")) or not os.path.isfile(os.path.join(seq_data_path, "pkl_AA_test.npy")):

        # need to make the full pickle file of all sequences to translate, then get the amino acid properties
        if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
            all_loci_seq = create_all_loci_matrices(config_file)
            pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))

        # make protein FASTA files for all loci, both tiers
        create_AA_alns(drug, kwargs["tier1_loci"] + kwargs["tier2_loci"], genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
        
        make_AA_property_matrices(drug,
                                  genes_lst,
                                  seq_data_path, 
                                  df_phenos, 
                                  genotype_input_directory,
                                  split_groups=True
                                 )

    # make other AF matrices
    if AF_thresh != 0.75 and not os.path.isfile(os.path.join(test_seq_data_path, 'pkl_AA_test.npy')):

        # need to make the full pickle file of all sequences to translate, then get the amino acid properties
        if not os.path.isfile(os.path.join(test_seq_data_path, "seqDict.pkl")):
            
            all_loci_seq = create_all_loci_matrices(config_file, 
                                                    fasta_dir=os.path.join(os.path.dirname(genotype_input_directory), f"AF_thresh_{int(AF_thresh*100)}", "fastas"),
                                                   )
            
            pickle.dump(all_loci_seq, open(os.path.join(test_seq_data_path, "seqDict.pkl"), "wb"))

        # make protein FASTA files for all loci, both tiers
        create_AA_alns(drug, 
                       kwargs["tier1_loci"] + kwargs["tier2_loci"], 
                       os.path.join(os.path.dirname(genotype_input_directory), f"AF_thresh_{int(AF_thresh*100)}", "fastas"), # AF thresh dir name
                       os.path.join(test_seq_data_path, "seqDict.pkl")
                      )
        
        make_AA_property_matrices(drug,
                                  genes_lst,
                                  test_seq_data_path, 
                                  df_phenos, 
                                  os.path.join(os.path.dirname(genotype_input_directory), f"AF_thresh_{int(AF_thresh*100)}", "fastas"), # AF thresh dir name
                                  split_groups=True
                                 )

        # delete the train_val file because we only need the test file
        os.remove(os.path.join(test_seq_data_path, 'pkl_AA_train_val.npy'))
        

test_generator = MtbGeneDataset(
    drug,
    df_test,
    os.path.join(test_seq_data_path, 'pkl_sparse_test.npz'),
    os.path.join(test_seq_data_path, 'pkl_AA_test.npy'),
    seq_data_path=test_seq_data_path, # use this for the test dataset if AF_thresh is not 0.75
    binary=binary,
    cc=binary_thresh,
    tier1_loci=tier1_loci,
    tier2_loci=tier2_loci,
    include_lineage=include_lineage,
    include_peptide_lengths=include_peptide_lengths,
    include_amino_acid_properties=include_amino_acid_properties,
    bounded_loss=bounded_loss,
    shuffle_batches=False, # don't need to shuffle test data because order doesn't matter
)

num_loci = len(test_generator.nuc_locus_list)
longest_locus = test_generator.longest_locus
longest_protein = test_generator.longest_protein
num_proteins = test_generator.num_proteins
num_peptide_lengths = test_generator.num_peptide_lengths
additional_data_len = test_generator.mlp_data_shape
print(f"MLP input shape: {additional_data_len}")

num_cv_splits = 5
cv_model_results = []

if train_model:

    # don't need to shuffle because samples within a batch are shuffled by the dataloader
    kfold_splits = StratifiedKFold(n_splits=num_cv_splits, shuffle=False)

    # stratify by the binary resistance phenotype only. You can pass in a dummy variable for X, which is np.zeros(len(df_train_val)) here
    for split, (train_idx, val_idx) in enumerate(kfold_splits.split(np.zeros(len(df_train_val)), df_train_val["Binary"])): 

        print(f"\nTraining {split+1}/{num_cv_splits} cross-validation splits")
        
        print(f"    CV train R: {df_train_val.iloc[train_idx]['Binary'].mean()}")
        print(f"    CV val R: {df_train_val.iloc[val_idx]['Binary'].mean()}")
    
        train_generator = MtbGeneDataset(
            drug,
            df_train_val, 
            os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'), 
            os.path.join(seq_data_path, 'pkl_AA_train_val.npy'),
            seq_data_path=seq_data_path,
            binary=binary,
            cc=binary_thresh,
            tier1_loci=tier1_loci,
            tier2_loci=tier2_loci,
            data_idx=train_idx, # get the train indices from the cross-validation splits
            include_lineage=include_lineage, 
            include_peptide_lengths=include_peptide_lengths, 
            include_amino_acid_properties=include_amino_acid_properties, 
            bounded_loss=bounded_loss, 
            shuffle_batches=True
        )
      
        val_generator = MtbGeneDataset(
            drug,
            df_train_val,
            os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'),
            os.path.join(seq_data_path, 'pkl_AA_train_val.npy'),
            seq_data_path=seq_data_path,
            binary=binary,
            cc=binary_thresh,
            tier1_loci=tier1_loci,
            tier2_loci=tier2_loci,
            data_idx=val_idx, # get the train indices from the cross-validation splits
            include_lineage=include_lineage,
            include_peptide_lengths=include_peptide_lengths,
            include_amino_acid_properties=include_amino_acid_properties, 
            bounded_loss=bounded_loss,
            shuffle_batches=False, # don't need to shuffle validation data because order in which you get the losses doesn't matter
        )
        
        if include_amino_acid_properties:
            model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
        else:
            model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)

        # train a model for on each split
        train_single_CNN(model, loss_type, N_epochs, train_generator, val_generator, len(train_idx), len(val_idx), save_model_fName=os.path.join(cv_dir, f"model_{split}.h5"), save_history_fName=os.path.join(cv_dir, f"history_{split}.csv"), patience_epochs=patience_epochs, return_min_loss=False)

        # load in the finished model
        model.load_weights(os.path.join(cv_dir, f"model_{split}.h5"))
    
        # get final model predictions on the TEST dataset, which has been until now set aside
        y_pred = model.predict(
            x=test_generator,
            workers=4,
            use_multiprocessing=True,
        )

        # predictions will be saved for isolates with Span_CC = 1, but the summary df will not include them in the computation
        summary_df = create_summary_df(df_test,
                                       y_pred, 
                                       drug,
                                       binary_thresh, 
                                       num_loci, 
                                       model_name="CNN", 
                                       binarize=True, 
                                       save_fName=os.path.join(cv_dir, f"test_predictions_{split}.csv"),
                                      )
        summary_df["CV"] = split
        cv_model_results.append(summary_df)
        
        if loss_type == "L1":
            print(f"    Final loss on the test set: {summary_df['Binned_MAE'].values[0]}")
        else:
            print(f"    Final loss on the test set: {summary_df['Binned_MSE'].values[0]}")
                    
        del train_generator
        del val_generator


    # save the cross-validation model results
    pd.concat(cv_model_results).to_csv(test_results_fName, index=False)


####################################### TO GET A FINAL MODEL FOR DOWNSTREAM ANALYSES, USE THE ENTIRE TRAINING SET AND TUNE IT USING THE TEST SET #######################################


    # # the model performance metrics have already been determined using cross-validation, and this is what will be used to compare model performances, so I believe this should be fine.
    # print("\nFitting full model on training set")

    # # this is the full training dataset
    # train_generator = MtbGeneDataset(
    #             drug,
    #             df_train_val, 
    #             os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'), 
    #             os.path.join(seq_data_path, 'pkl_AA_train_val.npy'),
    #             seq_data_path=seq_data_path,
    #             binary=binary,
    #             cc=binary_thresh,
    #             tier1_loci=tier1_loci,
    #             tier2_loci=tier2_loci,
    #             include_lineage=include_lineage, 
    #             include_peptide_lengths=include_peptide_lengths, 
    #             include_amino_acid_properties=include_amino_acid_properties, 
    #             bounded_loss=bounded_loss, 
    #             shuffle_batches=True
    #         )
    
    # if include_amino_acid_properties:
    #     model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
    # else:
    #     model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)
    
    # # train a model on the full dataset, tuning on the test dataset. Just save the model and history dataframe, don't need the loss
    # train_single_CNN(model,
    #                  loss_type, 
    #                  N_epochs, 
    #                  train_generator, 
    #                  test_generator, 
    #                  len(df_train_val), 
    #                  len(df_test),
    #                  save_model_fName=os.path.join(output_path, "best_model.h5"), 
    #                  save_history_fName=os.path.join(output_path, "history.csv"), 
    #                  patience_epochs=patience_epochs, 
    #                  return_min_loss=False
    #                 )

    K.clear_session()


# save model predictions
if os.path.isfile(os.path.join(output_path, "best_model.h5")):

    if include_amino_acid_properties:
        model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
    else:
        model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)
        
    # load in the weights of the best model
    model.load_weights(os.path.join(output_path, "best_model.h5"))
    
    y_pred = model.predict(
                            x=test_generator,
                            workers=4,
                            use_multiprocessing=True,
                        )
    
    # predictions will be saved for isolates with Span_CC = 1, but the summary df will not include them in the computation
    summary_df = create_summary_df(df_test,
                                   y_pred, 
                                   drug,
                                   binary_thresh, 
                                   num_loci, 
                                   model_name="CNN", 
                                   binarize=True, 
                                   save_fName=os.path.join(output_path, "test_predictions.csv"),
                                  )    

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
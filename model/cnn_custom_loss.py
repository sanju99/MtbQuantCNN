import sys, argparse, glob, os, yaml, sparse, tracemalloc, pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")
results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"

# utils files are in the utils directory
sys.path.append("utils")
from data_utils import *
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

parser.add_argument('--train', dest='train_model', action='store_true', help='If true, train a model on the full data.')

parser.add_argument('--CV', dest='perform_cross_validation', action='store_true', help="If true, perform 5-fold cross-validation.")

parser.add_argument('--epochs', default=10000, type=int, help='Maximum number of epochs to train the model')

parser.add_argument('--patience', default=200, type=int, help='Number of patience epochs for model training')

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--augment', dest='augment', action='store_true', help='If True, use the {drug}_augment directory')

parser.add_argument('--binary', dest='binary', action='store_true', help='If True, use the {drug}_binary directory and train a binary model')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
train_model = cmd_line_args.train_model
perform_cross_validation = cmd_line_args.perform_cross_validation
patience_epochs = cmd_line_args.patience
N_epochs = cmd_line_args.epochs
AF_thresh = cmd_line_args.AF_thresh
augment = cmd_line_args.augment
binary = cmd_line_args.binary

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

loss_type = "L1"

if train_model:
    bounded_loss = True
else:
    bounded_loss = False

output_path = f"{results_dir}/{drug}"

if augment:
    output_path += "_augment"

    # the last folder in the directory name is "fastas", and we need to append "augment" to the second to last level
    genotype_input_directory = os.path.join(os.path.dirname(genotype_input_directory) + "_augment", os.path.basename(genotype_input_directory))

    # same thing for the phenotypes file
    phenotype_file = os.path.join(os.path.dirname(phenotype_file) + "_augment", os.path.basename(phenotype_file))

    
if binary:
    output_path += "_binary"

    # the last folder in the directory name is "fastas", and we need to append "augment" to the second to last level
    genotype_input_directory = os.path.join(os.path.dirname(genotype_input_directory) + "_binary", os.path.basename(genotype_input_directory))

    # same thing for the phenotypes file
    phenotype_file = os.path.join(os.path.dirname(phenotype_file) + "_binary", os.path.basename(phenotype_file))

    
df_phenos = pd.read_csv(phenotype_file)
df_train_val = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)
df_test = df_phenos.query("category == 'test_set'").reset_index(drop=True)
print(df_phenos['Binary'].value_counts())
    
seq_data_path = output_path

if include_peptide_lengths:
    output_path += "_peptide"
    
if include_lineage:
    output_path += "_lineage"

if include_tier2:
    output_path += "_tier2"

if include_amino_acid_properties:
    output_path += "_amino_acid"

if AF_thresh != 0.75:

    # separate input paths for different AF
    seq_data_path = f"{seq_data_path}_AF{int(AF_thresh*100)}"

    output_path = f"{output_path}_AF{int(AF_thresh*100)}"

    genotype_input_directory = f"{genotype_input_directory.replace('fastas', 'AF_thresh_25/fastas')}"

if not os.path.isdir(seq_data_path):
    os.makedirs(seq_data_path)
    
cv_dir = os.path.join(output_path, "cross_validation")

if not os.path.isdir(cv_dir):
    os.makedirs(cv_dir)

print(f"Saving results to {output_path}")

# input files that need to be present
if not os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_train_val.npz")) or not os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_test.npz")):
    
    print(f"Making nucleotide one-hot encodings files using FASTA files in {genotype_input_directory}...\n")

    # make for all loci
    make_nucleotide_matrices(drug, 
                             kwargs["tier1_loci"] + kwargs['tier2_loci'],
                             seq_data_path,
                             df_phenos,
                             genotype_input_directory,
                             split_groups=True
                            )


if include_amino_acid_properties:

    genes_lst = get_genes_lst(kwargs["tier1_loci"] + kwargs['tier2_loci'])
    print(f"Genes list: {','.join(genes_lst)}")

    # make the amino acid property files if they don't exist
    if not os.path.isfile(os.path.join(seq_data_path, "pkl_AA_train_val.npy")) or not os.path.isfile(os.path.join(seq_data_path, "pkl_AA_test.npy")):

        # need to make the full pickle file of all sequences to translate, then get the amino acid properties
        if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):

            print(f"{len(df_phenos['ROLLINGDB_ID'].values)} isolates")
            print(f"genotype input directory: {genotype_input_directory}")
            
            all_loci_seq = create_all_loci_matrices(kwargs['tier1_loci'] + kwargs['tier2_loci'], 
                                                    genotype_input_directory, 
                                                    df_phenos['ROLLINGDB_ID'].values
                                                   )
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


    # compute the mean and SD of the training set to scale validation and test data later
    # scale across the sample axis (0) and the length of the amino acid sequence (2). Don't scale different biophysical properties together (1), or different genes together (3)
    train_mean_fName = os.path.join(seq_data_path, "AA_train_mean.npy")
    train_std_fName = os.path.join(seq_data_path, "AA_train_std.npy")

    if not os.path.isfile(train_mean_fName) or not os.path.isfile(train_std_fName):
    
        train_idx = df_train_val.index.values
    
        # this is the training matrix, which was just made above by the make_AA_property_matrices function
        X_amino_acid_train = np.load(os.path.join(seq_data_path, "pkl_AA_train_val.npy"))
        X_amino_acid_train = X_amino_acid_train[train_idx, :]
        
        train_mean = X_amino_acid_train.mean(axis=(0, 2))
        train_std = X_amino_acid_train.std(axis=(0, 2))
        del X_amino_acid_train
    
        np.save(train_mean_fName, train_mean)
        np.save(train_std_fName, train_std)
        

test_generator = MtbGeneDataset(
    drug,
    df_test,
    os.path.join(seq_data_path, 'pkl_sparse_test.npz'),
    os.path.join(seq_data_path, 'pkl_AA_test.npy'),
    seq_data_path=seq_data_path,
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

if perform_cross_validation:
    
    num_cv_splits = 5
    cv_model_results = []

    kfold_splits = StratifiedKFold(n_splits=num_cv_splits, shuffle=True)

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
        if not binary:
            train_single_CNN(model, loss_type, N_epochs, train_generator, val_generator, len(train_idx), len(val_idx), save_model_fName=os.path.join(cv_dir, f"model_{split}.h5"), save_history_fName=os.path.join(cv_dir, f"history_{split}.csv"), patience_epochs=patience_epochs, return_min_loss=False)
        else:
            model.compile(optimizer=Adam(learning_rate = np.exp(-1.0 * 9)), loss="binary_crossentropy", metrics=["accuracy"])
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_epochs, restore_best_weights=True)
            
            # get class weights on the training data to balance training
            balanced_class_weights = compute_class_weight(class_weight='balanced', 
                                                          classes=df_train_val.iloc[train_idx].Binary.unique(), 
                                                          y=df_train_val.iloc[train_idx].Binary.values
                                                         )
            
            balanced_class_weight_dict = dict(zip(df_train_val.iloc[train_idx].Binary.unique(), balanced_class_weights))
                        
            history = model.fit(
                                train_generator,
                                epochs=N_epochs,
                                batch_size=BATCH_SIZE,
                                validation_data=val_generator,
                                callbacks=[early_stop],
                                class_weight=balanced_class_weight_dict,
                                verbose=1
                            )
            
            # save the model
            model.save(os.path.join(cv_dir, f"model_{split}.h5"))
            
            # save the history
            df_history = pd.DataFrame(history.history)
            df_history.to_csv(os.path.join(cv_dir, f"history_{split}.csv"))
            
        # load in the finished model
        model.load_weights(os.path.join(cv_dir, f"model_{split}.h5"))

        # get final model predictions on the TEST dataset, which has been until now set aside
        y_pred = model.predict(
            x=test_generator,
            workers=4,
            use_multiprocessing=True,
        )

        if binary:
            
            df_binary_pred = df_test.copy()
            df_binary_pred['y_pred'] = y_pred

            # determine the threshold that maximizes sensitivity and specificity. This function adds a column y_pred_label, the binarized predictions, to the dataframe
            threshold_val, df_binary_pred = get_threshold_val(df_binary_pred, 'y_pred', 'Binary', spec_thresh=None)
            df_binary_pred = df_binary_pred.rename(columns={'Binary': 'y_test'})
            
            # save the threshold val. Need it to get predictions on new data because need to use the same binarization threshold. Save as an array of length 1
            np.save(os.path.join(cv_dir, f"binarization_threshold_{split}.npy"), [threshold_val])
            
            # save only relevant columns
            df_binary_pred[['ROLLINGDB_ID', 'y_test', 'y_pred', 'y_pred_label']].to_csv(os.path.join(cv_dir, f"test_predictions_{split}.csv"), index=False)

            # add the binary metrics, like sens, spec accuracy, F1, etc. using the compute_binary_metrics function
            summary_df = compute_binary_metrics(df_binary_pred['y_test'], df_binary_pred['y_pred_label'], binary_thresh, binarize=False)

            # add AUC
            summary_df['AUC'] = sklearn.metrics.roc_auc_score(df_binary_pred['y_test'], df_binary_pred['y_pred'])
                    
        else:
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

        if not binary:
            if loss_type == "L1":
                print(f"    Final loss on the test set: {summary_df['Binned_MAE'].values[0]}")
            else:
                print(f"    Final loss on the test set: {summary_df['Binned_MSE'].values[0]}")

        del train_generator
        del val_generator
        del df_binary_pred
        del df_history

    # save the cross-validation model results
    pd.concat(cv_model_results).to_csv(os.path.join(output_path, "results.csv"), index=False)
    
    K.clear_session()


####################################### TO GET A FINAL MODEL FOR DOWNSTREAM ANALYSES, USE THE ENTIRE TRAINING SET AND TUNE IT USING THE TEST SET #######################################

    
if train_model:
    
    # the model performance metrics have already been determined using cross-validation, and this is what will be used to compare model performances, so I believe this should be fine.
    print("\nFitting full model on training set")

    # this is the full training dataset
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
                include_lineage=include_lineage, 
                include_peptide_lengths=include_peptide_lengths, 
                include_amino_acid_properties=include_amino_acid_properties, 
                bounded_loss=bounded_loss, 
                shuffle_batches=True
            )
    
    if include_amino_acid_properties:
        model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
    else:
        model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)
        
    # train a model on the full dataset, tuning on the test dataset. Just save the model and history dataframe, don't need the loss
    if not binary:
        train_single_CNN(model,
                         loss_type, 
                         N_epochs, 
                         train_generator, 
                         test_generator, 
                         len(df_train_val), 
                         len(df_test),
                         save_model_fName=os.path.join(output_path, "best_model.h5"), 
                         save_history_fName=os.path.join(output_path, "history.csv"), 
                         patience_epochs=patience_epochs, 
                         return_min_loss=False
                        )
    else:
        model.compile(optimizer=Adam(learning_rate = np.exp(-1.0 * 9)), loss="binary_crossentropy", metrics=["accuracy"])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_epochs, restore_best_weights=True)

        # get class weights on the training data to balance training
        balanced_class_weights = compute_class_weight(class_weight='balanced', 
                                                      classes=df_train_val.query("category=='train_set'").Binary.unique(), 
                                                      y=df_train_val.query("category=='train_set'").Binary.values
                                                     )
        
        balanced_class_weight_dict = dict(zip(df_train_val.query("category=='train_set'").Binary.unique(), balanced_class_weights))

        history = model.fit(
                            train_generator,
                            epochs=N_epochs,
                            batch_size=BATCH_SIZE,
                            validation_data=test_generator,
                            callbacks=[early_stop],
                            class_weight=balanced_class_weight_dict,
                            verbose=1
                        )
        
        # save the model
        model.save(os.path.join(output_path, "best_model.h5"))

        df_history = pd.DataFrame(history.history)
        df_history.to_csv(os.path.join(output_path, "history.csv"))

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
       
    if binary:
        df_binary_pred = df_test.copy()
        df_binary_pred['y_pred'] = y_pred
        
        # determine the threshold that maximizes sensitivity and specificity. This function adds a column y_pred_label, the binarized predictions, to the dataframe
        threshold_val, df_binary_pred = get_threshold_val(df_binary_pred, 'y_pred', 'Binary', spec_thresh=None)
        df_binary_pred = df_binary_pred.rename(columns={'Binary': 'y_test'})
        
        # save the threshold val. Need it to get predictions on new data because need to use the same binarization threshold
        np.save(os.path.join(output_path, "binarization_threshold.npy"), threshold_val)
        
        # save only relevant columns
        df_binary_pred[['ROLLINGDB_ID', 'y_test', 'y_pred', 'y_pred_label']].to_csv(os.path.join(output_path, "test_predictions.csv"), index=False)
            
        # add the binary metrics, like sens, spec accuracy, F1, etc. using the compute_binary_metrics function
        summary_df = compute_binary_metrics(df_binary_pred['y_test'], df_binary_pred['y_pred_label'], binary_thresh, binarize=False)
                
        # add AUC
        summary_df['AUC'] = sklearn.metrics.roc_auc_score(df_binary_pred['y_test'], df_binary_pred['y_pred'])
                        
    else:
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

    summary_df.to_csv(os.path.join(output_path, "full_model_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
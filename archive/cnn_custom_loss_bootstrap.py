import sys, argparse, glob, os, yaml, sparse, tracemalloc, pickle
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
tf.config.run_functions_eagerly(True)
import scipy.stats as st

# utils files are in the utils_files directory
sys.path.append("utils")
from data_utils import *
from analysis_utils import *
from model_utils import *
from dataloader import MtbGeneDataset

import warnings
warnings.filterwarnings("ignore")


# starting the memory monitoring
tracemalloc.start()

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

parser.add_argument('--bootstrap', default=5, dest='num_bootstrap', type=int, help='Number of bootstrap replicates')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
train_model = cmd_line_args.train_model
patience_epochs = cmd_line_args.patience
num_bootstrap = cmd_line_args.num_bootstrap

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

if drug == 'PZA':
    patience_epochs = 150

num_loci = len(tier1_loci + tier2_loci)
df_phenos = pd.read_csv(phenotype_file)
df_train_val = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)
df_train = df_phenos.query("category == 'train_set'").reset_index(drop=True)
df_val = df_phenos.query("category == 'validation_set'").reset_index(drop=True)
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

if not os.path.isdir(output_path):
    os.makedirs(output_path)

bootstrap_output_path = os.path.join(output_path, "bootstrapping")

if not os.path.isdir(bootstrap_output_path):
    os.makedirs(bootstrap_output_path)

print(f"Saving results to {bootstrap_output_path}")

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
    data_subset="validation_set",
    include_lineage=include_lineage,
    include_peptide_lengths=include_peptide_lengths,
    include_amino_acid_properties=include_amino_acid_properties, 
    bounded_loss=bounded_loss,
    shuffle_batches=False, # don't need to shuffle validation data because order doesn't matter,
)

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

longest_locus = val_generator.longest_locus
longest_protein = val_generator.longest_protein
num_proteins = val_generator.num_proteins

# both features are combined into the same vector, so if at least one of them is True, there is a vector
if include_lineage or include_peptide_lengths:
    additional_data_len = val_generator[0][0][-3].shape[1]
else:
    additional_data_len = 0

results = []
history_df = []

if train_model:
    
    for rep in range(num_bootstrap):
    
        print(f"\nTraining bootstrap replicate {rep+1}/{num_bootstrap}")
        
        # sample indices with replacement
        train_idx = np.random.choice(np.arange(0, len(df_train)), size=len(df_train), replace=True)
        # df_bootstrap = get_stratified_bootstrap_sample(df_train, drug, col="Binary")
        
        bs_train_generator = MtbGeneDataset(
                                            drug,
                                            df_train_val, 
                                            os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'), 
                                            os.path.join(seq_data_path, 'pkl_AA_train_val.npy'),
                                            seq_data_path=seq_data_path,
                                            binary=binary,
                                            cc=binary_thresh,
                                            tier1_loci=tier1_loci,
                                            tier2_loci=tier2_loci,
                                            data_subset='train_set',
                                            data_idx=train_idx,
                                            include_lineage=include_lineage, 
                                            include_peptide_lengths=include_peptide_lengths, 
                                            include_amino_acid_properties=include_amino_acid_properties, 
                                            bounded_loss=bounded_loss, 
                                            shuffle_batches=True
        )
    
        # initialize the model using the regularization strength determined above:
        if include_amino_acid_properties:
            model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
        else:
            model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)
        
        # run the function to train a single model. Use 100 epochs patience for actual model training
        val_loss = train_single_CNN(model, loss_type, N_epochs, bs_train_generator, val_generator, len(df_train), len(df_val), save_model_fName=os.path.join(bootstrap_output_path, f"model_{rep}.h5"), save_history_df=False, patience_epochs=patience_epochs, return_min_loss=True)
        
        # load in the weights to get predictions and summary statistics
        model.load_weights(os.path.join(bootstrap_output_path, f"model_{rep}.h5"))
    
        # add validation loss for this replicate to the history dataframe
        history_df.append(pd.DataFrame({f"rep_{rep}": val_loss}))
        
        # get model predictions on the test dataset
        y_pred = model.predict(x=test_generator,
                               workers=4,
                               use_multiprocessing=True,
        )
    
        summary_df = create_summary_df(df_test, 
                                       y_pred, 
                                       drug, 
                                       binary_thresh, 
                                       num_loci, 
                                       model_name="CNN", 
                                       binarize=True, 
                                       save_fName=os.path.join(bootstrap_output_path, f"predictions_{rep}.csv")
                                      )
    
        summary_df["CV"] = rep + 1
             
        results.append(summary_df)

        # save intermediate bootstrap statistics in case it times out before all replicates are done
        pd.concat(results, axis=0).to_csv(os.path.join(bootstrap_output_path, "results.csv"), index=False)
        pd.concat(history_df, axis=1).to_csv(os.path.join(bootstrap_output_path, "histories.csv"), index=False)
        
        del model
        del val_loss
        # del df_bootstrap
        del train_idx
        del bs_train_generator
        K.clear_session()
    
    # save aggregate bootstrap statistics
    pd.concat(results, axis=0).to_csv(os.path.join(bootstrap_output_path, "results.csv"), index=False)
    pd.concat(history_df, axis=1).to_csv(os.path.join(bootstrap_output_path, "histories.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
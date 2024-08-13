import sys, argparse, glob, os, yaml, sparse, tracemalloc, pickle
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
tf.config.run_functions_eagerly(True)

# utils files are in the utils_files directory
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

parser.add_argument('--patience', default=100, type=int, help='Number of patience epochs for model training')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
patience_epochs = cmd_line_args.patience

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
output_path = f"/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/{drug}"

loss_type = "L1"
binary = False
bounded_loss = True
N_epochs = 10000

df_phenos = pd.read_csv(phenotype_file)
df_train = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)
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
    
# update output path for the saliency folder. Save the permutation models in a new subdirectory
saliency_dir = os.path.join(output_path, "saliency", "permutation_test")
print(f"Saving results to {saliency_dir}")
del output_path
    
if not os.path.isdir(saliency_dir):
    os.makedirs(os.path.join(saliency_dir)) 

test_generator = MtbGeneDataset(
    drug,
    df_test,
    os.path.join(seq_data_path, 'pkl_sparse_test.npz'),
    os.path.join(seq_data_path, 'pkl_AA_test.npy'),
    seq_data_path=seq_data_path, # use this for the test dataset if AF_thresh is not 0.75
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
        
num_reps = 10

for rep in range(num_reps):
    
    print(f"\nTraining permutation {rep+1}/{num_reps} with {loss_type} loss")
    
    # for each replicate, randomly shuffle the MICs, which is stochastic. Use entire training set, but because the shuffling is different, we can train 10 times like this
    train_generator = MtbGeneDataset(
        drug,
        df_train, 
        os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'), 
        os.path.join(seq_data_path, 'pkl_AA_train_val.npy'),
        seq_data_path=seq_data_path,
        binary=binary,
        cc=binary_thresh,
        tier1_loci=tier1_loci,
        tier2_loci=tier2_loci,
        data_subset="train_set", 
        shuffle_phenos=True, # shuffle phenotypes for the permutation test
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

    # just need to train and save the permuted models here. In the saliency scripts, we will use the trained models to compute saliency scores
    train_single_CNN(model, 
                     loss_type, 
                     N_epochs, 
                     train_generator, 
                     test_generator, 
                     len(df_train), 
                     len(df_test), 
                     save_model_fName=os.path.join(saliency_dir, f"permutation_{rep}.h5"), 
                     save_history_fName=None, 
                     patience_epochs=patience_epochs, 
                     return_min_loss=False
                    )              

    K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
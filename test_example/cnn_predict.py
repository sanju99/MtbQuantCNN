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

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--augment', dest='augment', action='store_true', help='If True, use the {drug}_augment directory')

parser.add_argument('--binary', dest='binary', action='store_true', help='If True, use the {drug}_binary directory and train a binary model')

parser.add_argument('-d', dest='results_dir', default="/n/data1/hms/dbmi/farhat/Sanjana/CNN_results", help='Output directory to save output files to')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
AF_thresh = cmd_line_args.AF_thresh
augment = cmd_line_args.augment
binary = cmd_line_args.binary
output_path = cmd_line_args.results_dir

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
phenotype_file = os.path.join(output_path, "df_phenos.csv")
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]

# because we're just doing prediction, so don't keep track of the MIC bounds
bounded_loss = False

if augment:

    # the last folder in the directory name is "fastas", and we need to append "augment" to the second to last level
    genotype_input_directory = os.path.join(os.path.dirname(genotype_input_directory) + "_augment", os.path.basename(genotype_input_directory))

    # same thing for the phenotypes file
    phenotype_file = os.path.join(os.path.dirname(phenotype_file) + "_augment", os.path.basename(phenotype_file))

    
if binary:

    # the last folder in the directory name is "fastas", and we need to append "augment" to the second to last level
    genotype_input_directory = os.path.join(os.path.dirname(genotype_input_directory) + "_binary", os.path.basename(genotype_input_directory))

    # same thing for the phenotypes file
    phenotype_file = os.path.join(os.path.dirname(phenotype_file) + "_binary", os.path.basename(phenotype_file))

    
df_phenos = pd.read_csv(phenotype_file)
print(df_phenos['Binary'].value_counts())
    
if not os.path.isdir(output_path):
    os.makedirs(output_path)

print(f"Saving results to {output_path}")

# input files that need to be present
if not os.path.isfile(os.path.join(output_path, "pkl_sparse_test.npz")):
    
    print(f"Making nucleotide one-hot encodings files using FASTA files in {genotype_input_directory}...\n")

    # make for all loci
    make_nucleotide_matrices(drug, 
                             kwargs["tier1_loci"] + kwargs['tier2_loci'],
                             output_path,
                             df_phenos,
                             genotype_input_directory,
                             split_groups=True
                            )


if include_amino_acid_properties:

    genes_lst = get_genes_lst(kwargs["tier1_loci"] + kwargs['tier2_loci'])
    print(f"Genes list: {','.join(genes_lst)}")

    # make the amino acid property files if they don't exist
    if not os.path.isfile(os.path.join(output_path, "pkl_AA_test.npy")):

        # need to make the full pickle file of all sequences to translate, then get the amino acid properties
        if not os.path.isfile(os.path.join(output_path, "seqDict.pkl")):

            print(f"{len(df_phenos['ROLLINGDB_ID'].values)} isolates")
            print(f"genotype input directory: {genotype_input_directory}")
            
            all_loci_seq = create_all_loci_matrices(kwargs['tier1_loci'] + kwargs['tier2_loci'], 
                                                    genotype_input_directory, 
                                                    df_phenos['ROLLINGDB_ID'].values
                                                   )
            pickle.dump(all_loci_seq, open(os.path.join(output_path, "seqDict.pkl"), "wb"))

        # make protein FASTA files for all loci, both tiers
        create_AA_alns(drug, kwargs["tier1_loci"] + kwargs["tier2_loci"], genotype_input_directory, os.path.join(output_path, "seqDict.pkl"))
        
        make_AA_property_matrices(drug,
                                  genes_lst,
                                  output_path, 
                                  df_phenos, 
                                  genotype_input_directory,
                                  split_groups=True
                                 )   

test_generator = MtbGeneDataset(
    drug,
    df_phenos,
    os.path.join(output_path, 'pkl_sparse_test.npz'),
    os.path.join(output_path, 'pkl_AA_test.npy'),
    seq_data_path=output_path,
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

if include_amino_acid_properties:
    model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
else:
    model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)

# load in the weights of the best model
model.load_weights(os.path.join(output_path, "best_model.h5"))

y_pred = model.predict(x=test_generator,
                        workers=1,
                        use_multiprocessing=False,
                      )

if not binary:
    
    # predictions dataframe: get indices of validation data in the cv splits
    pred_df = df_phenos[["ROLLINGDB_ID", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound", "Span_CC"]]

    # rename columns to make them easier to read
    pred_df.rename(columns={"ROLLINGDB_ID": "Isolate", 
                            f"{drug}_midpoint": "y_test",
                            f"{drug}_lower_bound": "lower",
                            f"{drug}_upper_bound": "upper"
                           }, 
                   inplace=True
                  )

    # add model predictions, and log-transform the test values
    pred_df["y_pred"] = np.squeeze(y_pred)
    pred_df["y_test_log2"] = np.log2(pred_df["y_test"])
    
    # exponentiate the predictions
    pred_df['y_pred_exp'] = np.exp2(y_pred)
    
    # save
    pred_df.to_csv(os.path.join(output_path, "test_predictions.csv"), index=False) 

else:
    df_binary_pred = df_phenos.copy()
    df_binary_pred['y_pred'] = y_pred

    # determine the threshold that maximizes sensitivity and specificity. This function adds a column y_pred_label, the binarized predictions, to the dataframe
    threshold_val, df_binary_pred = get_threshold_val(df_binary_pred, 'y_pred', 'Binary', spec_thresh=None)
    df_binary_pred = df_binary_pred.rename(columns={'Binary': 'y_test'})

    # save the threshold val. Need it to get predictions on new data because need to use the same binarization threshold
    np.save(os.path.join(output_path, "binarization_threshold.npy"), threshold_val)

    # save only relevant columns
    df_binary_pred[['ROLLINGDB_ID', 'y_test', 'y_pred', 'y_pred_label']].to_csv(os.path.join(output_path, "test_predictions.csv"), index=False) 

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
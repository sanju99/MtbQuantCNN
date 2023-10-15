import sys, glob, os, yaml, sparse, tracemalloc
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

# starting the memory monitoring
tracemalloc.start()

_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
locus_list = kwargs["locus_list"]
filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]
include_peptide_length = kwargs["include_peptide_length"]
N_epochs = 10000
loss_type = "L1"
binary = kwargs["binary"]
bounded_loss = kwargs["bounded_loss"]

num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)

# creat output directories
if binary:
    model_prefix = "binary_"
    save_prefix = "binary"
else:
    model_prefix = ""
    save_prefix = "quant"

seq_data_path = output_path.replace("_lineage", "").replace("_peptide", "")
    
# make peptide lengths dataframe if it has not already been created
if include_peptide_length and not os.path.isfile(os.path.join(seq_data_path, "locus_peptide_lengths.csv")):

    if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
        raise ValueError(f'Please create {os.path.join(seq_data_path, "seqDict.pkl")} using the saliency_utils functions before running this peptide lengths model')
    
    locus_peptide_lengths = make_CDS_length_df(locus_list, genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
    
    # keep index because that's the samples column
    locus_peptide_lengths.to_csv(os.path.join(seq_data_path, "locus_peptide_lengths.csv"))


# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(seq_data_path, 'pkl_sparse_ref.npz'))
longest_locus = X_h37rv.shape[2]
del X_h37rv

val_generator = MtbGeneDataset(
    os.path.join(seq_data_path, 'pkl_sparse_full.npz'),
    phenotype_file,
    drug,
    locus_list,
    fasta_dir=genotype_input_directory,
    train_or_test="original_test_set",
    binary=binary,
    cc=binary_thresh,
    shuffle_phenos=False,
    include_lineage=include_lineage,
    include_peptide_length=include_peptide_length,
    bounded_loss=bounded_loss,
    data_idx=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)          
        
# both features are combined into the same vector, so if at least one of them is True, there is a vector
if include_lineage or include_peptide_length:
    additional_data_len = val_generator[0][0][1].shape[1]
else:
    additional_data_len = 0

# update output path for the saliency folder. Save the permutation models in a new subdirectory
saliency_dir = os.path.join(output_path, "saliency", save_prefix, "permutation_test")
print(f"Saving results to {saliency_dir}")
    
if not os.path.isdir(saliency_dir):
    os.makedirs(os.path.join(saliency_dir))    
    
# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)

# get regularization parameter
losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))

# get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
select_alpha = np.round(losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0], 6)  # not sure why, but some of the alphas are like 0.999999999 instead of 1
print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}\n")

if drug == "PZA":
    patience_epochs = 75
else:
    patience_epochs = 50
        
num_reps = 10

for rep in range(num_reps):
    
    print(f"\nTraining permutation {rep+1}/{num_reps} with {loss_type} loss")
    
    # for each replicate, randomly shuffle the MICs, so get new training data each time. Use entire training set
    train_generator = MtbGeneDataset(
        os.path.join(seq_data_path, 'pkl_sparse_full.npz'),
        phenotype_file,
        drug,
        locus_list,
        fasta_dir=genotype_input_directory,
        train_or_test="original_train_set",
        binary=binary,
        cc=binary_thresh,
        shuffle_phenos=True,
        include_lineage=include_lineage,
        include_peptide_length=include_peptide_length,
        bounded_loss=bounded_loss,
        data_idx=None,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    if bounded_loss:
        
        # initialize the model using the function from cnn_utils and the optimizer
        model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=select_alpha)

        # run the function to train a single model. Use 50 epochs patience because it's not going to change much
        # save the model, but don't need the history array
        _ = train_single_CNN(model, loss_type, N_epochs, train_generator, val_generator, len(df_train), len(df_test), save_model_fName=os.path.join(saliency_dir, f"permutation_{rep}.h5"), save_history_df=False, patience_epochs=patience_epochs, return_min_loss=False)

    else:
        if binary:
            loss_func = tf.keras.losses.BinaryCrossentropy()
            
            # get class weights for the training data only
            df_phenos = pd.read_csv(phenotype_file)
            
            if f"{drug}_midpoint" in df_phenos.columns:
                y_train = (df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values > binary_thresh).astype(int)
            else:
                y_train = df_phenos.query("category=='original_train_set'")["phenotype"].values.astype(int)

            assert len(np.unique(y_train)) == 2
            class_weights = class_weighting_dictionary(y_train)
            del y_train
            
        else:
            loss_func = tf.keras.losses.MeanAbsoluteError()
            class_weights = None
        
        model.compile(loss=loss_func, optimizer=optimizer)
        
        model.fit(x=train_generator, 
                  epochs=N_epochs,
                  use_multiprocessing=True,
                  workers=4,
                  class_weight=class_weights,
                )
        
        model.save(os.path.join(saliency_dir, f"permutation_{rep+1}.h5"))                

    K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
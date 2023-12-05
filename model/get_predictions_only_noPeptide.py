import sys, glob, os, yaml, sparse, tracemalloc
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
tf.config.run_functions_eagerly(True)

# utils files are in the utils directory
sys.path.append("utils")
from data_utils import *
from analysis_utils import *
from model_utils import *
from dataloader import MtbGeneDataset

import warnings
warnings.filterwarnings("ignore")


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

include_peptide_length = False
lowAF = True

if lowAF:
    file_suffix = "_lowAF"

loss_type = "L1"
binary = False
bounded_loss = True
N_epochs = 10000

seq_data_path = output_path.replace("_lineage", "").replace("_peptide", "")

# naming consistency
output_path = output_path.replace("_lineage", "").replace("_peptide", "")

if include_peptide_length:
    output_path += "_peptide"
    
if include_lineage:
    output_path += "_lineage"
    
bootstrap_output_path = os.path.join(output_path, "bootstrapping")
num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)

print(np.sort(df_phenos[f"{drug}_upper_bound"].unique()))

if not os.path.isdir(output_path):
    os.mkdir(output_path)

print(f"Saving results to {output_path}")

# if os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_full.npz")): 
#     print("Input one-hot encodings files exists. Proceeding with modeling \n")    
# else:
#     print("Making input one-hot encodings file...\n")
#     make_geno_pheno_files(seq_data_path, **kwargs)
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(seq_data_path, 'pkl_sparse_ref.npz'))

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
del X_h37rv

# # make peptide lengths dataframe if it has not already been created
# if include_peptide_length and not os.path.isfile(os.path.join(seq_data_path, "locus_peptide_lengths.csv")):

#     if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
#         all_loci_seq = create_all_loci_matrices(config_file)
#         pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))
    
#     locus_peptide_lengths = make_CDS_length_df(locus_list, genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
    
#     # keep index because that's the samples column
#     locus_peptide_lengths.to_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"))
    
losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))

# get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
select_alpha = losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0]
print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")
    
val_generator = MtbGeneDataset(
    os.path.join(seq_data_path, 'pkl_sparse_full.npz'),
    phenotype_file,
    drug,
    locus_list,
    fasta_dir=genotype_input_directory,
    binary=binary,
    cc=binary_thresh,
    train_or_test="original_test_set",
    shuffle_phenos=False,
    include_lineage=include_lineage,
    include_peptide_length=include_peptide_length,
    lowAF=lowAF,
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

# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)

# get total numbers of points for computing the loss
num_train = len(df_phenos.query("category=='original_train_set'"))
num_val = len(df_phenos.query("category=='original_test_set'"))
print(f"Training on {num_train} isolates and validating on {num_val} isolates")
        
# initialize a new model and load the weights of the best model
best_model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=select_alpha)
best_model.load_weights(os.path.join(output_path, "best_model.h5"))

# get final model predictions
y_pred = best_model.predict(
    x=val_generator,
    workers=4,
    use_multiprocessing=True,
)

summary_df = create_summary_df(df_phenos.query("category=='original_test_set'").reset_index(drop=True),
                               y_pred, 
                               drug,
                               binary_thresh, 
                               num_loci, 
                               model_name="CNN", 
                               binarize=True, 
                               save_fName=os.path.join(output_path, f"test_predictions{file_suffix}.csv")
                              )
summary_df["CV"] = 0

if loss_type == "L1":
    print(f"    Final Binned MAE: {summary_df['Binned_MAE'].values[0]}")
else:
    print(f"    Final Binned MSE: {summary_df['Binned_MSE'].values[0]}")
        
summary_df.to_csv(os.path.join(output_path, f"results{file_suffix}.csv"), index=False)
K.clear_session()

num_reps = 10
results = []

for rep in range(num_reps):

    print(f"\n Getting predictions for bootstrap replicate {rep+1}/{num_reps}")
    
    model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=select_alpha)
    model.load_weights(os.path.join(bootstrap_output_path, f"model_{rep}.h5"))
    
    # get model predictions
    y_pred = model.predict(x=val_generator,
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
                                   save_fName=os.path.join(bootstrap_output_path, f"predictions_{rep}{file_suffix}.csv")
                                  )

    summary_df["CV"] = rep + 1
         
    results.append(summary_df)
    del model
    K.clear_session()
            
# save summary statistics from cross-validation
pd.concat(results, axis=0).to_csv(os.path.join(bootstrap_output_path, f"results{file_suffix}.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
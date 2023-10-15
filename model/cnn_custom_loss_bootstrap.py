import sys, glob, os, yaml, sparse, tracemalloc
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
loss_type = "L1"
binary = False
bounded_loss = True
N_epochs = 10000

num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)

seq_data_path = output_path.replace("_lineage", "").replace("_peptide", "")
bootstrap_output_path = os.path.join(output_path, "bootstrapping")

if not os.path.isdir(bootstrap_output_path):
    os.makedirs(bootstrap_output_path)

print(f"Saving results to {bootstrap_output_path}")

if os.path.isfile(os.path.join(seq_data_path, "pkl_sparse_full.npz")): 
    print("Input one-hot encodings file exists. Proceeding with modeling \n")    
else:
    print("Making input one-hot encodings file...\n")
    make_geno_pheno_files(**kwargs)
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(seq_data_path, 'pkl_sparse_ref.npz'))

# shape = 1 x 5 x longest_locus x num_loci
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

# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)

num_reps = 10
results = []
history_df = []

# def get_stratified_bootstrap_sample(df, drug, col="Binary"):

#     df_pos = df.query(f"{col} == 1")
#     df_neg = df.query(f"{col} == 0")

#     df_pos_sampled = df.sample(n=len(df_pos), replace=True)
#     df_neg_sampled = df.sample(n=len(df_neg), replace=True)

#     df_bootstrap = pd.concat([df_pos_sampled, df_neg_sampled], axis=0)

#     # check that the MICs could have come from the same distribution
#     assert st.kstest(df[f"{drug}_midpoint"], df_bootstrap[f"{drug}_midpoint"])[1] > 0.05
#     print(df[col].mean(), df_bootstrap[col].mean())
#     return df_bootstrap

# get regularization parameter
losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))

# get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
select_alpha = np.round(losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0], 6)  # not sure why, but some of the alphas are like 0.999999999 instead of 1
print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}\n")

# run the function to train a single model. Use 100 epochs patience for actual model training or 150 for PZA because the loss decreases slower
if drug == "PZA":
    patience_epochs = 150
else:
    patience_epochs = 100

for rep in range(num_reps):

    print(f"\nTraining bootstrap replicate {rep+1}/{num_reps}")
    
    # reset the patience counter and min_loss for each replicate
    patience_counter = 0
    min_loss = 1e3
    val_loss = []
    
    # sample indices with replacement
    train_idx = np.random.choice(np.arange(0, len(df_train)), size=len(df_train), replace=True)
    # df_bootstrap = get_stratified_bootstrap_sample(df_train, drug, col="Binary")
    
    bs_train_generator = MtbGeneDataset(
                                    os.path.join(seq_data_path, 'pkl_sparse_full.npz'),
                                    phenotype_file,
                                    drug,
                                    locus_list,
                                    fasta_dir=genotype_input_directory,
                                    train_or_test="original_train_set",
                                    binary=binary,
                                    cc=binary_thresh,
                                    shuffle_phenos=False,
                                    include_lineage=include_lineage,
                                    include_peptide_length=include_peptide_length,
                                    bounded_loss=bounded_loss,
                                    data_idx=train_idx,
                                    #data_idx=df_bootstrap.index.values, # because the index was not reset, the indices from df_train are preserved
                                    batch_size=BATCH_SIZE,
                                    shuffle=True
    )

    # initialize the model using the function from cnn_utils and the optimizer
    model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=select_alpha)
    optimizer = Adam(learning_rate = np.exp(-1.0 * 9))
    
    # run the function to train a single model. Use 100 epochs patience for actual model training
    val_loss = train_single_CNN(model, loss_type, N_epochs, bs_train_generator, val_generator, len(df_train), len(df_test), save_model_fName=os.path.join(bootstrap_output_path, f"model_{rep}.h5"), save_history_df=False, patience_epochs=patience_epochs, return_min_loss=False)

    model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=select_alpha)
    model.load_weights(os.path.join(bootstrap_output_path, f"model_{rep}.h5"))

    # add validation loss for this replicate to the history dataframe
    history_df.append(pd.DataFrame({f"rep_{rep}": val_loss}))
    
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
                                   save_fName=None
                                  )

    summary_df["CV"] = rep + 1
         
    results.append(summary_df)
    del model
    del optimizer
    del patience_counter
    del min_loss
    del val_loss
    #del df_bootstrap
    del train_idx
    del bs_train_generator
    K.clear_session()
    
            
# save summary statistics from cross-validation
pd.concat(results, axis=0).to_csv(os.path.join(bootstrap_output_path, "results.csv"), index=False)
pd.concat(history_df, axis=1).to_csv(os.path.join(bootstrap_output_path, "histories.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
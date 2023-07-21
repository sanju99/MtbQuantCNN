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
N_epochs = kwargs["N_epochs"]
patience_epochs = None

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]
loss_type = kwargs["loss_type"]
binary = False
bounded_loss = True

num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)

bootstrap_output_path = os.path.join(output_path, "bootstrapping")

if not os.path.isdir(bootstrap_output_path):
    os.makedirs(bootstrap_output_path)

if os.path.isfile(os.path.join(output_path, "pkl_sparse_train.npz")): 
    print("Input one-hot encodings file exists. Proceeding with modeling \n")    
else:
    print("Making input one-hot encodings file...\n")
    make_geno_pheno_files(**kwargs)
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz'))

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
del X_h37rv

train_generator = MtbGeneDataset(
    os.path.join(output_path, 'pkl_sparse_train.npz'),
    phenotype_file,
    drug,
    locus_list,
    train_or_test="original_train_set",
    binary=binary,
    cc=binary_thresh,
    shuffle_phenos=False,
    include_lineage=include_lineage,
    bounded_loss=bounded_loss,
    data_idx=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)
    
val_generator = MtbGeneDataset(
    os.path.join(output_path, 'pkl_sparse_test.npz'),
    phenotype_file,
    drug,
    locus_list,
    train_or_test="original_test_set",
    binary=binary,
    cc=binary_thresh,
    shuffle_phenos=False,
    include_lineage=include_lineage,
    bounded_loss=bounded_loss,
    data_idx=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if include_lineage:
    num_lineages = val_generator[0][0][1].shape[1]
else:
    num_lineages = 0


# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)

num_reps = 10
results = []
history_df = []

for rep in range(num_reps):
    
    val_loss = []
    print(f"\nTraining replicate {rep+1}/{num_reps} with an {loss_type} loss for {N_epochs} epochs")

    # initialize the model using the function from cnn_utils and the optimizer
    model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    optimizer = Adam(learning_rate = np.exp(-1.0 * 9))
    
    @tf.function
    def train_step(x, y):
        '''
        This is the training step for a single batch. Iterating over batches and epochs is done separately. Redefine and recompile this function for every bootstrapped model. 
        '''

        # the bounds are the last 2 elements of the x list
        lower_bounds, upper_bounds = x[-2:]

        with tf.GradientTape() as tape:

            # Make predictions using the model
            y_hat = model(x, training=True)

            # Calculate the loss using the two bounds tensors. custom_bounded_mae is imported from cnn_utils
            loss = boundedLoss_CNN(lower_bounds, upper_bounds, loss_type)(y, y_hat)

        # Calculate the gradients
        gradients = tape.gradient(loss, model.trainable_weights)

        # run the optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # return loss
        return loss.numpy()
    
    
    @tf.function
    def val_step(x, y):

        # the bounds are the last 2 elements of the x list
        lower_bounds, upper_bounds = x[-2:]

        y_hat = model(x, training=False)

        # return loss
        return boundedLoss_CNN(lower_bounds, upper_bounds, loss_type)(y, y_hat).numpy()
    

    # train the model
    for epoch in range(N_epochs):
        
        # only storing validation losses in this script
        val_epoch_loss = []

        # training loop: don't keep track of the train losses because we just want to train the model here
        for (x_batch_train, y_batch_train) in train_generator:

            _ = train_step(x_batch_train, y_batch_train) 
            
        # validation loop for a single epoch
        for (x_batch_val, y_batch_val) in val_generator:

            # compute bounded error
            val_epoch_loss.append(val_step(x_batch_val, y_batch_val))

        val_loss.append(np.sum(val_epoch_loss) / len(df_test))

        if (epoch % 10 == 0) or (epoch == (N_epochs-1)):
            print(f"Epoch {epoch+1} validation loss: {val_loss[-1]}")
          
    model.save(os.path.join(bootstrap_output_path, f"model_{rep}.h5"))
    model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
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
    del val_loss
    
            
# save summary statistics from cross-validation
pd.concat(results, axis=0).to_csv(os.path.join(bootstrap_output_path, "replicates_results.csv"), index=False)
pd.concat(history_df, axis=1).to_csv(os.path.join(bootstrap_output_path, "history_replicates.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
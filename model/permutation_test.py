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
patience_epochs = kwargs["patience_epochs"]
N_epochs = 1000

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]
loss_type = kwargs["loss_type"]
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
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path.replace("_lineage", ""), 'pkl_sparse_ref.npz'))
longest_locus = X_h37rv.shape[2]
del X_h37rv

val_generator = MtbGeneDataset(
    os.path.join(output_path.replace("_lineage", ""), 'pkl_sparse_test.npz'),
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

# update output path for the saliency folder. Save the permutation models in a new subdirectory
saliency_dir = os.path.join(output_path, "saliency", save_prefix, "permutation_test")
print(f"Saving results to {saliency_dir}")
    
if not os.path.isdir(saliency_dir):
    os.makedirs(os.path.join(saliency_dir))    
    
# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)
        
num_reps = 10
for rep in range(num_reps):
    
    patience_counter = 0
    min_loss = 1e3
    val_loss = []
    
    if patience_epochs is None:
        print(f"\nTraining replicate {rep+1}/{num_reps} with {loss_type} loss for {N_epochs} epochs")
    else:
        print(f"\nTraining replicate {rep+1}/{num_reps} with {loss_type} loss and a delay of {patience_epochs} epochs")
    
    # for each replicate, randomly shuffle the MICs, so get new training data each time. Use entire training set
    train_generator = MtbGeneDataset(
        os.path.join(output_path.replace("_lineage", ""), 'pkl_sparse_train.npz'),
        phenotype_file,
        drug,
        locus_list,
        train_or_test="original_train_set",
        binary=binary,
        cc=binary_thresh,
        shuffle_phenos=True,
        include_lineage=include_lineage,
        bounded_loss=bounded_loss,
        data_idx=None,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    if include_lineage:
        num_lineages = train_generator[0][0][1].shape[1]
    else:
        num_lineages = 0
    
    # initialize the model using the function from cnn_utils and the optimizer
    model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    optimizer = Adam(learning_rate = np.exp(-1.0 * 9))
    
    if bounded_loss:
        
        @tf.function
        def train_step(x, y):
            '''
            This is the training step for a single batch. Iterating over batches and epochs is done separately. Redefine and recompile this function for every permuted model. 
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

            if patience_epochs is not None:

                # if loss decreases by at least 1%
                if float((min_loss - val_loss[-1]) / min_loss) >= 0.01:

                    print(f"Epoch {epoch+1}: Validation loss improved from {min_loss} to {val_loss[-1]}")

                    # update min loss, then zero out the patience counter. Save the model only if the loss decreases so the the model in the patience window doesn't save
                    model.save(os.path.join(saliency_dir, f"permutation_{rep+1}.h5"))
                    min_loss = val_loss[-1]
                    patience_counter = 0

                else:
                    patience_counter += 1

                if patience_counter == patience_epochs:
                    break

            # train the model for the specified number of epochs
            else:
                if N_epochs < 10:
                    print(f"Epoch {epoch} validation loss: {val_loss[-1]}")
                else:
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch} validation loss: {val_loss[-1]}")

        if patience_epochs is None:
            model.save(os.path.join(saliency_dir, f"permutation_{rep+1}.h5"))


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
            loss_func = tf.keras.losses.MeanSquaredError()
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
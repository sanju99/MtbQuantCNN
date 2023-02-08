import sys, glob, os, yaml, sparse, tracemalloc
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K

# code to go up one level in the directory tree if needed
# sys.path.append(os.path.dirname(os.getcwd()))
from data_utils import *
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
N_epochs = kwargs["N_epochs"]
N_epochs = 1
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

if not os.path.isdir(output_path):
    os.mkdir(output_path)

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
    num_lineages = train_generator[0][0][1].shape[1]
else:
    num_lineages = 0


@tf.function
def train_step(x, y):
    '''
    This is the training step for a single batch. Iterating over batches and epochs is done separately
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
    return loss
    
    
@tf.function
def val_step(x, y):
        
    # the bounds are the last 2 elements of the x list
    lower_bounds, upper_bounds = x[-2:]
    
    y_hat = model(x, training=False)
    
    # return loss
    return boundedLoss_CNN(lower_bounds, upper_bounds, loss_type)(y, y_hat)


# initialize the model using the function from cnn_utils and the optimizer
model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
optimizer = Adam(learning_rate = np.exp(-1.0 * 9))

# manual implementation of model callbacks
patience_counter = 0
min_loss = np.inf

# initialize lists to store losses
train_loss = []
train_error = []
val_loss = []
val_error = []

results = pd.DataFrame(columns=["train_loss", "train_error", "val_loss", "val_error"])

if patience_epochs is None:
    print(f"Training the model with an {loss_type} loss for {N_epochs} epochs")
else:
    print(f"Using early stopping with an {loss_type} loss and a delay of {patience_epochs} epochs")


for epoch in range(N_epochs):
        
    # list to keep track of the losses for each batch
    train_epoch_loss = []
    train_epoch_error = []
    val_epoch_loss = []
    val_epoch_error = []
    
    # training loop
    for train_idx, (x_batch_train, y_batch_train) in enumerate(train_generator):
                
        # compute bounded error
        train_epoch_loss.append(train_step(x_batch_train, y_batch_train).numpy())
        
        # compute absolute error
        y_hat_train = model(x_batch_train, training=False)
        
        if loss_type == "L1":
            train_epoch_error.append(np.mean(np.abs(y_hat_train - y_batch_train.flatten())))
        elif loss_type == "L2":
            train_epoch_error.append(np.mean((y_hat_train - y_batch_train.flatten())**2))
        
    # store losses for the epoch -- mean of all the batches
    train_loss.append(np.mean(train_epoch_loss))    
    train_error.append(np.mean(train_epoch_error))

    # validation loop
    for _, (x_batch_val, y_batch_val) in enumerate(val_generator):
                        
        # compute bounded error
        val_epoch_loss.append(val_step(x_batch_val, y_batch_val).numpy())

        # compute absolute error
        y_hat_val = model(x_batch_val, training=False)
        
        if loss_type == "L1":
            val_epoch_error.append(np.mean(np.abs(y_hat_val - y_batch_val.flatten())))
        elif loss_type == "L2":
            val_epoch_error.append(np.mean((y_hat_val - y_batch_val.flatten())**2))
        
    val_loss.append(np.mean(val_epoch_loss))
    val_error.append(np.mean(val_epoch_error))
    
    results.loc[epoch, :] = [train_loss[-1], train_error[-1], val_loss[-1], val_error[-1]]
    
    if patience_epochs is not None:
        if val_loss[-1] < min_loss:
            print(f"Epoch {epoch+1}: Validation loss improved from {min_loss} to {val_loss[-1]}")

            # update min loss, then zero out the patience counter
            min_loss = val_loss[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter == patience_epochs:
            break
    
    # train the model for the specified number of epochs
    else:
        continue
         
print(results)
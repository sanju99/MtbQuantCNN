import sys, glob, os, yaml, sparse, tracemalloc
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from cnn_utils import *
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
patience_epochs = kwargs["patience_epochs"]

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]

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
print(f"Longest locus: {longest_locus}")


train_generator = MtbGeneDataset(
    os.path.join(output_path, 'pkl_sparse_train.npz'),
    phenotype_file,
    drug,
    locus_list,
    train_or_test="original_train_set",
    binary=False,
    cc=binary_thresh,
    include_lineage=include_lineage,
    bounded_loss=True,
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
    binary=False,
    cc=binary_thresh,
    include_lineage=include_lineage,
    bounded_loss=True,
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
        loss = custom_bounded_mae(lower_bounds, upper_bounds)(y, y_hat)

    # Calculate the gradients
    gradients = tape.gradient(loss, model.trainable_weights)

    # run the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return loss
    
    
@tf.function
def val_step(x, y):
        
    # the bounds are the last 2 elements of the x list
    lower_bounds, upper_bounds = x[-2:]
    
    y_hat = model(x, training=False)
    
    # return the loss
    return custom_bounded_mae(lower_bounds, upper_bounds)(y, y_hat)


# initialize the model using the function from cnn_utils
model = custom_loss_quant_CNN(longest_locus, num_loci, num_lineages, filter_size=filter_size)

# manual implementation of model callbacks
patience_counter = 0
min_loss = np.inf

# initialize lists to store losses
train_loss = []
val_loss = []
mae = []

# initialize an optimizer
optimizer=Adam(learning_rate = np.exp(-1.0 * 9))

results = pd.DataFrame(columns=["train_loss", "val_loss", "val_mae"])

for epoch in range(N_epochs):
        
    # list to keep track of the losses for each batch
    train_epoch_loss = []
    val_epoch_loss = []
    val_epoch_mae = []
    
    # training loop
    for train_idx, (x_batch_train, y_batch_train) in enumerate(train_generator):
        
        # get the mean loss of the batch
        train_epoch_loss.append(train_step(x_batch_train, y_batch_train).numpy())
        
    # store losses for the epoch -- mean of all the batches
    train_loss.append(np.mean(train_epoch_loss))    

    # validation loop
    for _, (x_batch_val, y_batch_val) in enumerate(val_generator):
        
        y_hat = model(x_batch_val, training=False).numpy().flatten()
        
        # compute bounded MAE
        val_epoch_loss.append(val_step(x_batch_val, y_batch_val).numpy())

        # compute absolute MAE
        val_epoch_mae.append(np.mean(np.abs(y_hat - y_batch_val)))
        
    val_loss.append(np.mean(val_epoch_loss))
    mae.append(np.mean(val_epoch_mae))
    
    results.loc[epoch, :] = [train_loss[-1], val_loss[-1], mae[-1]]
    
    if val_loss[-1] < min_loss:
        model.save(os.path.join(output_path, "best_model.h5"))
        print(f"Epoch {epoch+1}: Validation loss improved from {min_loss} to {val_loss[-1]}")
        
        # update min loss, then zero out the patience counter
        min_loss = val_loss[-1]
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter == patience_epochs:
        break
    

results.to_csv(os.path.join(output_path, "history.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
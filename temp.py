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
binary = kwargs["binary"]
include_lineage = kwargs["include_lineage"]
bounded_loss = kwargs["bounded_loss"]

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
    binary=binary,
    cc=binary_thresh,
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
    
    




def custom_loss_model(lineages):
    
    cnn_input = tf.keras.Input(shape=(5, longest_locus, num_loci), name='seq_input')
    mlp_input = tf.keras.Input(shape=(num_lineages, ), name='lineage_input')

    # first perform convolutions and max pooling as in the original model. 
    x = layers.Conv2D(64, (5, filter_size), data_format='channels_last', activation='relu', input_shape=(5, longest_locus, num_loci), name='conv1')(cnn_input)
    x = layers.Conv2D(64, (1,12), activation='relu', name='conv2')(x)

    conv_block_1 = layers.MaxPooling2D((1,3), name='maxPooling1')(x)

    y = layers.Conv2D(32, (1,3), activation='relu', name='conv3')(conv_block_1)
    y = layers.Conv2D(32, (1,3), activation='relu', name='conv4')(y)

    conv_block_2 = layers.MaxPooling2D((1,3), name='maxPooling2')(y)

    # flattened output of convolutional block. Concatenate this with the lineages, then pass into dense layers
    cnn_output = layers.Flatten(name='flatten')(conv_block_2)
    dense_inputs = layers.concatenate([cnn_output, mlp_input], axis=1, name='concatenate')

    dense = layers.Dense(256, activation='relu', name='dense1')(dense_inputs)
    dense = layers.Dense(256, activation='relu', name='dense2')(dense)
    output = layers.Dense(1, activation=None, name='output')(dense)

    lower_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='lower_bounds')
    upper_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='upper_bounds')

    # sequence one-hot encoding, lineages, lower_bounds, upper_bounds
    model = keras.Model(inputs=[cnn_input, mlp_input, lower_bounds, upper_bounds], outputs=output)
    return model


def train_step(x, y):
    '''
    This is the training step for a single batch. Iterating over batches and epochs is done separately
    '''
    
    # the bounds are the last 2 elements of the x list
    lower_bounds, upper_bounds = x[-2:]
    
    with tf.GradientTape() as tape:

        # Make predictions using the model
        y_hat = model(x, training=True)

        # Calculate the loss using the two bounds tensors
        loss = custom_bounded_mae(lower_bounds, upper_bounds)(y, y_hat)

    # Calculate the gradients
    gradients = tape.gradient(loss, model.trainable_weights)

    # run the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return loss
    
    
def val_step(x, y):
        
    # the bounds are the last 2 elements of the x list
    lower_bounds, upper_bounds = x[-2:]
    
    y_hat = model(x, training=False)
    
    # return the loss
    return custom_bounded_mae(lower_bounds, upper_bounds)(y, y_hat)


model = get_model()
N_epochs = 1000
patience_counter = 0
patience_thresh = 20
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
    
    # print(f"Finished epoch {epoch+1} out of {N_epochs}: train_loss = {train_loss[-1]}, val_loss = {val_loss[-1]}, mae = {mae[-1]}")
    results.loc[epoch, :] = [train_loss[-1], val_loss[-1], mae[-1]]
    
    if val_loss[-1] < min_loss:
        model.save(os.path.join(output_path, "best_model.h5"))
        print(f"Epoch {epoch+1}: Validation loss improved from {min_loss} to {val_loss[-1]}. Saving model")
        
        # update min loss, then zero out the patience counter
        min_loss = val_loss[-1]
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter == patience_thresh:
        break
    

results.to_csv(os.path.join(output_path, "history.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
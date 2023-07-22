import sys, glob, os, yaml, sparse, tracemalloc
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
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
patience_epochs = kwargs["patience_epochs"]

if patience_epochs is not None:
    N_epochs = 1000
else:
    N_epochs = kwargs["N_epochs"]

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
    num_lineages = val_generator[0][0][1].shape[1]
else:
    num_lineages = 0

    
# get total numbers of points for computing the loss
num_train = len(df_phenos.query("category=='original_train_set'"))
num_val = len(df_phenos.query("category=='original_test_set'"))
print(f"Training on {num_train} isolates and validating on {num_val} isolates")


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
    
    # return loss and error. quantLoss_CNN returns a numpy object from a tensor
    return loss.numpy(), quantLoss_CNN(y, y_hat, loss_type)
    
    
@tf.function
def val_step(x, y):
        
    # the bounds are the last 2 elements of the x list
    lower_bounds, upper_bounds = x[-2:]
    
    # y_hat = model.predict(x)
    y_hat = model(x, training=False)
    
    # return loss and error. quantLoss_CNN returns a numpy object from a tensor
    return boundedLoss_CNN(lower_bounds, upper_bounds, loss_type)(y, y_hat).numpy(), quantLoss_CNN(y, y_hat, loss_type)


# initialize the model using the function from cnn_utils and the optimizer
model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
optimizer = Adam(learning_rate = np.exp(-1.0 * 9))

# manual implementation of model callbacks
patience_counter = 0
min_loss = 1e3

# initialize lists to store losses
train_loss = []
train_error = []
val_loss = []
val_error = []

history = pd.DataFrame(columns=["loss", "error", "val_loss", "val_error"])

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
    for x_batch_train, y_batch_train in train_generator:
                
        # compute loss and error. These are sums over the points in the batch
        loss, error = train_step(x_batch_train, y_batch_train)
        
        train_epoch_loss.append(loss)
        train_epoch_error.append(error)
        
    # store losses for the epoch -- mean of all the batches
    train_loss.append(np.sum(train_epoch_loss) / num_train)   
    train_error.append(np.sum(train_epoch_error) / num_train)

    # validation loop -- iterate through all batches
    for x_batch_val, y_batch_val in val_generator:
                        
        # compute loss and error
        loss, error = val_step(x_batch_val, y_batch_val)
        
        val_epoch_loss.append(loss)
        val_epoch_error.append(error)
      
    # store the mean loss of the batch
    val_loss.append(np.sum(val_epoch_loss) / num_val)
    val_error.append(np.sum(val_epoch_error) / num_val)

    history.loc[epoch, :] = [train_loss[-1], train_error[-1], val_loss[-1], val_error[-1]]    
    
    if patience_epochs is not None:
        #if val_loss[-1] < min_loss:

        # if loss decreases by at least 1%
        if float((min_loss - val_loss[-1]) / min_loss) >= 0.01:
        
            print(f"Epoch {epoch+1}: Validation loss improved from {min_loss} to {val_loss[-1]}")
            
            # save the model because it is better than the previous iteration
            model.save(os.path.join(output_path, "best_model.h5"))

            # update min loss, then zero out the patience counter
            min_loss = val_loss[-1]
            patience_counter = 0
            
        else:
            patience_counter += 1

        if patience_counter == patience_epochs:
            break
    
    # train the model for the specified number of epochs
    else:
        print(f"Epoch {epoch} validation loss: {val_loss[-1]}")
        continue
        
    
# save the history dataframe to see the additional epochs during the patience period. DON'T SAVE THE MODEL because we want the model at the early stop point
history.to_csv(os.path.join(output_path, "history.csv"), index=False)

# only save the model if not using early stopping because it doesn't get saved as you go in the above loop
if patience_epochs is None:
    model.save(os.path.join(output_path, "best_model.h5"))

# initialize a new model and load the weights of the best model
best_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
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
                               save_fName=os.path.join(output_path, "test_predictions.csv")
                              )
summary_df["CV"] = 0
summary_df.to_csv(os.path.join(output_path, "cnn_results.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
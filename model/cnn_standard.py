import sys, glob, os, yaml, sparse, tracemalloc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# utils files are in the utils_files directory
sys.path.append("utils_files")
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
patience_epochs = kwargs["patience_epochs"]

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
binary = kwargs["binary"]
loss_type = kwargs["loss_type"]
include_lineage = kwargs["include_lineage"]
bounded_loss = False

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

print(f"Including {num_lineages} lineages in this model")
model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
optimizer = Adam(learning_rate = np.exp(-1.0 * 9))

# get class weights for binary training to balance weights
if binary:
    prefix = "binary_"
    
    # get class weights for the training data only
    y_train = (df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values > binary_thresh).astype(int)

    assert len(np.unique(y_train)) == 2
    class_weights = class_weighting_dictionary(y_train)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer)
            
else:
    prefix = ""
    class_weights = None
    
    if loss_type == "L1":
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=optimizer)
    elif loss_type == "L2":
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)
    else:
        raise ValueError(f"{loss_type} is an invalid loss type for quantitative CNNs")
            

# include early stopping and get the model checkpoint at the best epoch
if patience_epochs is not None:
    print(f"Using early stopping with an {loss_type} loss and a delay of {patience_epochs} epochs")
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience_epochs, verbose=1)
    mc = ModelCheckpoint(os.path.join(output_path, f'{prefix}best_model.h5'), monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    model_callbacks = [es, mc]
else:
    print(f"Training the model with an {loss_type} loss for {N_epochs} epochs")
    model_callbacks = []
    
    
# don't specify batch size when using data generators
history = model.fit(
                    x=train_generator, 
                    epochs=N_epochs,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks=model_callbacks,
                    class_weight=class_weights,
                   )

# save history dataframe, predictions vs. test values dataframe, and the model
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(output_path, f"{prefix}history.csv"), index=False)

# manually save the model if not using the callback
if patience_epochs is None:
    model.save(os.path.join(output_path, f"{prefix}best_model.h5"))

# initialize a new model and load the weights of the best model
best_model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
best_model.load_weights(os.path.join(output_path, f"{prefix}best_model.h5"))

# get model predictions
y_pred = best_model.predict(
    x=val_generator,
    workers=4,
    use_multiprocessing=True,
)

# get test values and IDs from the dataset class
ids = np.array([])
y_test = np.array([])

for i, _ in enumerate(val_generator):
    
    val_batch = val_generator.__getTestData__(i)
    
    ids = np.concatenate([ids, val_batch[0]])
    y_test = np.concatenate([y_test, val_batch[1]])
    
    
pred_df = pd.DataFrame({"Isolate": ids, "y_pred": np.squeeze(y_pred), "y_test": y_test})
    
if binary:
    # optimize the classification threshold using sensitivity and specificity and add the class labels 
    pred_df = get_threshold_val(pred_df, "y_pred", "y_test")        

pred_df.to_csv(os.path.join(output_path, f"{prefix}test_predictions.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
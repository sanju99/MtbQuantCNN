import sys, glob, os, yaml, sparse
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
import tracemalloc


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
lineage_file = kwargs["lineage_file"]
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

# keep only isolates that are in the phenotypes file. Drop any lineage columns that are all 0 (none represented)
lineage_mat = pd.read_csv(lineage_file, index_col=[0]).loc[df_phenos["ROLLINGDB_ID"]]
lineage_mat = lineage_mat.replace(0, np.nan).dropna(axis=1, how="all").replace(np.nan, 0).astype(int)

# check that every lineage has at least 1 isolate representing it
assert lineage_mat.sum(axis=0).min() > 0

# check that every isolate has only a single lineage listed
assert len(np.unique(lineage_mat.sum(axis=1))) == 1
assert np.unique(lineage_mat.sum(axis=1))[0] == 1

train_generator = MtbGeneDataset(
    os.path.join(output_path, 'pkl_sparse_train.npz'),
    phenotype_file,
    lineage_mat,
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
    lineage_mat,
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
model = conv_nn(longest_locus, num_loci, num_lineages, binary, bounded_loss, filter_size=filter_size, preSoftmax=False)
print(f"{model.count_params()} parameters in the model")


# get class weights for binary training to balance weights
if binary:
    prefix = "binary_"
    
    # get class weights for the training data only
    if drug+"_midpoint" in df_phenos.columns:
        y_train = (df_phenos.query("category=='original_train_set'")[drug+"_midpoint"].values > binary_thresh).astype(int)
    else:
        y_train = df_phenos.query("category=='original_train_set'")["phenotype"].values.astype(int)

    assert len(np.unique(y_train)) == 2
    class_weights = class_weighting_dictionary(y_train)
            
else:
    prefix = ""
    class_weights = None
    

# include early stopping and get the model checkpoint at the best epoch
if patience_epochs is not None:
    print("Using early stopping...")
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience_epochs, verbose=1)
    mc = ModelCheckpoint(os.path.join(output_path, f'{prefix}best_model.h5'), monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    model_callbacks = [es, mc]
else:
    print(f"Training model for {N_epochs} epochs...")
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

# load in the saved best model
best_model = load_model(os.path.join(output_path, f"{prefix}best_model.h5"), custom_objects={'bounded_mae': bounded_mae})

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
else:
    # add bounds for the prediction
    pred_df = pred_df.merge(df_phenos[["ROLLINGDB_ID", f"{drug}_lower_bound", f"{drug}_upper_bound"]], left_on="Isolate", right_on="ROLLINGDB_ID")
    del pred_df["ROLLINGDB_ID"]
    pred_df[["y_pred_exp", "y_test_exp"]] = np.exp(pred_df[["y_pred", "y_test"]])
    assert sum(pred_df["y_test_exp"] < pred_df[f"{drug}_lower_bound"]) == 0
    assert sum(pred_df["y_test_exp"] > pred_df[f"{drug}_upper_bound"]) == 0         

pred_df.to_csv(os.path.join(output_path, f"{prefix}test_predictions.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
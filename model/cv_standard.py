import sys, glob, os, yaml, sparse, tracemalloc
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam

# cnn_utils is one level up in the directory tree
sys.path.append(os.path.dirname(os.getcwd()))
from cnn_utils import *


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
binary = kwargs["binary"]
binary_thresh = kwargs["binary_thresh"]

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
include_lineage = kwargs["include_lineage"]
bounded_loss = False

    
num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)

if not os.path.isdir(output_path):
    os.mkdir(output_path)

if os.path.isfile(os.path.join(output_path, "pkl_sparse_train.npz")): 
    print("Input one-hot encodings file exists. Proceeding with modeling")    
else:
    print("Making input one-hot encodings file...")
    make_geno_pheno_files(**kwargs)
    
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz'))

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
del X_h37rv

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

# dataframe to store the results from multiple models
if binary:
    prefix = "binary_"
else:
    prefix = ""

# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)

bootstrap_reps = 1
results = []
history_df = pd.DataFrame(columns=[f"rep_{i+1}" for i in range(bootstrap_reps)])

for rep in range(bootstrap_reps):

    print(f"Working on replicate {rep+1}/{bootstrap_reps}")
    val_loss = []
    
    # sample indices with replacement
    train_idx = np.random.choice(np.arange(0, len(df_train)), size=len(df_train), replace=True)
    
    cv_train_generator = MtbGeneDataset(
                                        os.path.join(output_path, 'pkl_sparse_train.npz'),
                                        phenotype_file,
                                        drug,
                                        locus_list,
                                        train_or_test="original_train_set",
                                        binary=binary,
                                        cc=binary_thresh,
                                        include_lineage=include_lineage,
                                        bounded_loss=bounded_loss,
                                        data_idx=train_idx,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True
    )
    
    if include_lineage:
        num_lineages = cv_train_generator[0][0][1].shape[1]
    else:
        num_lineages = 0

    # initialize the model using the function from cnn_utils and the optimizer
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
        
    history = model.fit(x=cv_train_generator, 
                        epochs=N_epochs,
                        validation_data=val_generator,
                        use_multiprocessing=True,
                        workers=4,
                        class_weight=class_weights,
    )
    
    # add validation loss for this replicate to the history dataframe
    history_df[f"fold_{fold+1}"] = pd.DataFrame(history.history)["val_loss"]
    model.save(os.path.join(output_path, f"{prefix}cv_model_{fold+1}.h5"))
    
    # get model predictions
    y_pred = model.predict(x=val_generator,
                           workers=4,
                           use_multiprocessing=True,
    )

    ids, y_test = list(zip(*df_test[["ROLLINGDB_ID", f"{drug}_midpoint"]].values))

    if binary:        
        pred_df = pd.DataFrame({"Isolate": ids, "y_pred": np.squeeze(y_pred), "y_test": (np.array(y_test) > binary_thresh).astype(int)})
        
        # adds a column called y_pred_label based on the threshold determined for classification of the sigmoid outputs
        pred_df = get_threshold_val(pred_df, "y_pred", "y_test")
                
        # compute binary metrics: sens, spec, auc, auc_pr, acc, balanced_acc
        binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred_label"], binary_thresh, binarize=False)
    else:
        pred_df = pd.DataFrame({"Isolate": ids, "y_pred": np.squeeze(y_pred), "y_test": np.log(y_test)})
        
        # compute quantitative metrics
        mae = np.mean(np.abs(pred_df["y_test"] - pred_df["y_pred"]))
        mse = np.mean((pred_df["y_test"] - pred_df["y_pred"])**2)
        pearson = st.pearsonr(pred_df["y_test"], pred_df["y_pred"])[0]

        summary_df = pd.DataFrame({"Drug": drug,
                                   "Model": "CNN",
                                   "Num_Loci": num_loci,
                                   "MAE": mae,
                                   "MSE": mse,
                                   "Pearson": pearson,
                                  }, index=[0])

        # compute binary metrics: sens, spec, auc, auc_pr, acc, balanced_acc
        binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred"], binary_thresh, binarize=True)
        binary_metrics_df = pd.concat([summary_df, binary_metrics_df], axis=1)
        
    results.append(binary_metrics_df)
                        
# save summary statistics from cross-validation
pd.concat(results).to_csv(os.path.join(output_path, f"{prefix}val_results.csv"), index=False)
history_df.to_csv(os.path.join(output_path, f"{prefix}history_cv.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
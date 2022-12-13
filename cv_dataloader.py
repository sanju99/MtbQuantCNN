import sys
import glob
import os
import yaml
import sparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import scipy.stats as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
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
binary = kwargs["binary"]
binary_thresh = kwargs["binary_thresh"]

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
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
    num_lineages = val_generator[0][0][1].shape[1]
else:
    num_lineages = 0

# dataframe to store the results from multiple models
if binary:
    prefix = "binary_"
else:
    prefix = ""

# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)

bootstrap_reps = 10
results = []

for fold in range(bootstrap_reps):
# for fold in range(5):
    # assert sum(pd.isnull(df_train[f"CV{fold}_train"])) == 0
    
    # the isolates assigned to the training set for the current fold
    # train_idx = df_train.loc[df_train[f"CV{fold}_train"] == 1].index.values
    
    # cv_train_generator = MtbGeneDataset(
    #             os.path.join(output_path, 'pkl_sparse_train.npz'),
    #             phenotype_file,
    #             drug,
    #             locus_list,
    #             train_or_test="original_train_set",
    #             binary=binary,
    #             cc=binary_thresh,
    #             include_lineage=include_lineage,
    #             data_idx=train_idx,
    #             batch_size=BATCH_SIZE,
    #             shuffle=True
    # )
    
    # if include_lineage:
    #     print("Including lineage in this model")
    #     model = conv_nn_with_lineage(longest_locus, num_loci, num_snps, binary, filter_size)
    # else:
    #     model = conv_nn(longest_locus, num_loci, binary, filter_size)
    # print(f"{model.count_params()} parameters in the model")
    
    print(f"Working on fold {fold+1}/{bootstrap_reps}")
    
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

    print(f"Including {num_lineages} lineages in this model")
    model = conv_nn(longest_locus, num_loci, num_lineages, binary, bounded_loss, filter_size=filter_size, preSoftmax=False)
    print(f"{model.count_params()} parameters in the model")

    if binary:
        # get class weights for the training data only
        y_train = (df_train[drug+"_midpoint"].values[train_idx] > binary_thresh).astype(int)
        class_weights = class_weighting_dictionary(y_train)
    else:
        class_weights = None
        
    history = model.fit(x=cv_train_generator, 
                        epochs=N_epochs,
                        validation_data=val_generator,
                        use_multiprocessing=True,
                        workers=4,
                        class_weight=class_weights,
    )
    
    # save model. Make sure to include custom_objects={'bounded_mae': bounded_mae} if using load_model later
    model.save(os.path.join(output_path, f"{prefix}model_cv_split_{fold+1}.h5"))
    
    # save history dataframe. Consider not saving this in the future
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_path, f"{prefix}history_cv_split_{fold+1}.csv"), index=False)
    
    # get model predictions
    y_pred = model.predict(x=val_generator,
                           workers=4,
                           use_multiprocessing=True,
    )

    # get test values and IDs from the dataset class
    ids = np.array([])
    y_val = np.array([])

    for i, _ in enumerate(val_generator):

        val_batch = val_generator.__getTestData__(i)

        ids = np.concatenate([ids, val_batch[0]])
        y_val = np.concatenate([y_val, val_batch[1]])

    pred_df = pd.DataFrame({"Isolate": ids, "y_pred": np.squeeze(y_pred), "y_test": y_val})        
        
    if binary:
        pred_df = get_threshold_val(pred_df, "y_pred", "y_test")
        y_val_binary = pred_df["y_test"].values
        y_pred_binary = pred_df["y_pred_label"].values
                
        # compute binary metrics: sens, spec, auc, auc_pr, acc, balanced_acc
        binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred_label"], binary_thresh, binarize=False)
    else:
        y_val_binary = (pred_df["y_test"].values > np.log(binary_thresh)).astype(int)
        y_pred_binary = (pred_df["y_pred"].values > np.log(binary_thresh)).astype(int)
        
        # compute the proportion of MICs predicted wtihin 1 bin of the true bin
        pred_df = pred_df.merge(df_phenos[["ROLLINGDB_ID", f"{drug}_lower_bound", f"{drug}_upper_bound"]], left_on="Isolate", right_on="ROLLINGDB_ID")
        del pred_df["ROLLINGDB_ID"]
        pred_df[["y_pred_exp", "y_test_exp"]] = np.exp(pred_df[["y_pred", "y_test"]])
        assert sum(pred_df["y_test_exp"] < pred_df[f"{drug}_lower_bound"]) == 0
        assert sum(pred_df["y_test_exp"] > pred_df[f"{drug}_upper_bound"]) == 0    
        within_1bin = len(pred_df.loc[(pred_df["y_pred_exp"] >= pred_df[f"{drug}_lower_bound"] / 2) & (pred_df["y_pred_exp"] <= pred_df[f"{drug}_upper_bound"] * 2)]) / len(pred_df)
        
        # compute quantitative metrics
        binned_mae = bounded_mae_standalone(pred_df.y_test, pred_df.y_pred)
        mae = np.mean(np.abs(pred_df.y_test - pred_df.y_pred))
        rmse = np.sqrt(np.mean((pred_df.y_test - pred_df.y_pred)**2))
        pearson = st.pearsonr(pred_df.y_test, pred_df.y_pred)[0]
                
        # compute binary metrics: sens, spec, auc, auc_pr, acc, balanced_acc
        binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred"], binary_thresh, binarize=True)
        binary_metrics_df[["Binned_MAE", "MAE", "RMSE", "Pearson", "Within_1Bin"]] = [binned_mae, mae, rmse, pearson, within_1bin]
        
    results.append(binary_metrics_df)
                
results = pd.concat(results)
results[["Model", "Drug"]] = ["CNN", drug]
        
# save summary statistics from cross-validation
results.to_csv(os.path.join(output_path, f"{prefix}val_results.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
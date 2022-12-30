import sys, glob, os, yaml, sparse, tracemalloc
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
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
patience_epochs = kwargs["patience_epochs"]

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
    print("Input one-hot encodings file exists. Proceeding with modeling")    
else:
    print("Making input one-hot encodings file...")
    make_geno_pheno_files(**kwargs)
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz'))

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
del X_h37rv

cv = 5
cv_splits = StratifiedKFold(n_splits=cv)
results = []
history_df = pd.DataFrame(columns=[f"fold_{i+1}" for i in range(cv)])

# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)

# stratify by binary phenotype and primary lineage
df_train["stratify_col"] = df_train["Binary"].astype(str) + "_" + df_train["Primary_Lineage"].astype(str)

for fold, (train_idx, val_idx) in enumerate(cv_splits.split(df_train[f"{drug}_midpoint"], df_train["stratify_col"])):

    print(f"Working on fold {fold+1}/{cv}")
    val_loss = []
    
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
    
    cv_val_generator = MtbGeneDataset(
                                        os.path.join(output_path, 'pkl_sparse_train.npz'),
                                        phenotype_file,
                                        drug,
                                        locus_list,
                                        train_or_test="original_train_set",
                                        binary=binary,
                                        cc=binary_thresh,
                                        include_lineage=include_lineage,
                                        bounded_loss=bounded_loss,
                                        data_idx=val_idx,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False
    )

    if include_lineage:
        num_lineages = cv_train_generator[0][0][1].shape[1]
    else:
        num_lineages = 0
    
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
        return loss
    
    
    @tf.function
    def val_step(x, y):

        # the bounds are the last 2 elements of the x list
        lower_bounds, upper_bounds = x[-2:]

        y_hat = model(x, training=False)

        # return loss
        return boundedLoss_CNN(lower_bounds, upper_bounds, loss_type)(y, y_hat)
    

    # train the bootstrapped model
    for epoch in range(N_epochs):

        # training loop: don't keep track of the train losses because we just want to train the model here
        for train_idx, (x_batch_train, y_batch_train) in enumerate(cv_train_generator):

            _ = train_step(x_batch_train, y_batch_train) 
        
        # validation loop
        val_epoch_loss = []
        for _, (x_batch_val, y_batch_val) in enumerate(cv_val_generator):

            # compute bounded error
            val_epoch_loss.append(val_step(x_batch_val, y_batch_val).numpy())

        val_loss.append(np.mean(val_epoch_loss))
          
    # add validation loss for this fold to the history dataframe
    history_df[f"fold_{fold+1}"] = val_loss
    
    # get model predictions
    y_pred = model.predict(x=cv_val_generator,
                           workers=4,
                           use_multiprocessing=True,
                          )
        
    ids, y_test, lower, upper = list(zip(*df_train.loc[val_idx, :][["ROLLINGDB_ID", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound"]].values))

    pred_df = pd.DataFrame({"Isolate": ids, "y_pred": np.squeeze(y_pred), "y_test": np.log(y_test), "lower": np.array(lower), "upper": np.array(upper)})     

    # compute quantitative metrics
    binned_mae, binned_mse, within_1bin = boundedLoss_predict(pred_df, "y_pred", "y_test", "lower", "upper")
    mae = np.mean(np.abs(pred_df.y_test - pred_df.y_pred))
    mse = np.mean((pred_df.y_test - pred_df.y_pred)**2)
    pearson = st.pearsonr(pred_df.y_test, pred_df.y_pred)[0]
    
    summary_df = pd.DataFrame({"Drug": drug,
                               "Model": "CNN",
                               "Num_Loci": num_loci,
                               "Binned_MAE": binned_mae,
                               "Binned_MSE": binned_mse,
                               "MAE": mae,
                               "MSE": mse,
                               "Within_1bin": within_1bin,
                               "Pearson": pearson,
                              }, index=[0])

    # compute binary metrics: sens, spec, auc, auc_pr, acc, balanced_acc
    binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred"], binary_thresh, binarize=True)
    summary_df = pd.concat([summary_df, binary_metrics_df], axis=1)
        
    results.append(summary_df)
    del model
    del optimizer
            
# save summary statistics from cross-validation
pd.concat(results).to_csv(os.path.join(output_path, "val_results.csv"), index=False)
history_df.to_csv(os.path.join(output_path, "history_cv.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
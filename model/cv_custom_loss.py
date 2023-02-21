import sys, glob, os, yaml, sparse, tracemalloc
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam

# utils files are in the utils_files directory
sys.path.append("utils")
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

# get test values and IDs from the validation dataset to compare with the predicted values later
ids = np.array([])
y_test = np.array([])
lower_bounds = np.array([])
upper_bounds = np.array([])

for i, _ in enumerate(val_generator):

    val_batch = val_generator.__getTestData__(i)    
    ids = np.concatenate([ids, val_batch[0]])
    y_test = np.concatenate([y_test, val_batch[1]])

    bounds_batch = val_generator.__getBounds__(i)
    lower_bounds = np.concatenate([lower_bounds, bounds_batch[0]])
    upper_bounds = np.concatenate([upper_bounds, bounds_batch[1]])
        
        
if include_lineage:
    num_lineages = val_generator[0][0][1].shape[1]
else:
    num_lineages = 0


# need the train dataframe indices for slicing it. Reset index so that it's the index within the values, not in the overall dataframe
df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)
df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)

bootstrap_reps = 10
results = []
history_df = pd.DataFrame(columns=[f"rep_{i+1}" for i in range(bootstrap_reps)])

for rep in range(bootstrap_reps):

    print(f"Training replicate {rep+1}/{bootstrap_reps} for {N_epochs} epochs")
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
        for (x_batch_train, y_batch_train) in cv_train_generator:

            _ = train_step(x_batch_train, y_batch_train) 
        
        # validation loop
        val_epoch_loss = []
        for _, (x_batch_val, y_batch_val) in enumerate(val_generator):

            # compute bounded error
            val_epoch_loss.append(val_step(x_batch_val, y_batch_val).numpy())

        val_loss.append(np.mean(val_epoch_loss))
          
    # add validation loss for this replicate to the history dataframe
    history_df[f"rep_{rep+1}"] = val_loss
    
    # get model predictions
    y_pred = model.predict(x=val_generator,
                           workers=4,
                           use_multiprocessing=True,
    )

    ids, y_test, lower, upper = list(zip(*df_test[["ROLLINGDB_ID", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound"]].values))

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
history_df.to_csv(os.path.join(output_path, "history_replicates.csv"), index=False)
K.clear_session()

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Ridge, RidgeCV
import sklearn.metrics
    
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

from data_utils import *


def quantLoss_CNN(y_true, y_pred, loss_type):
    '''
    This function returns MAE or MSE (unbounded) for a model. This is just a metric to keep track of, but IT IS NOT USED AS THE MODEL LOSS FUNCTION
    '''
    
    # ensure same types of everything
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    # compute the errors first using the log-MICs, based on the desired loss type
    if loss_type == "L1":
        errors = K.abs(y_true - y_pred)
    elif loss_type == "L2":
        errors = K.square(y_true - y_pred)
    else:
        raise RuntimeError(f"{loss_type} is not a valid loss function type")
        
    # return sum because in the training loop it will be divided by the total number of points in each batch
    return K.sum(errors).numpy()




def boundedLoss_CNN(lower_bounds, upper_bounds, loss_type):
    '''
    The bounds are in exponentiated form because some lower bounds are 0. So when computing the loss, y_pred must be exponentiated

    This function computes error relative to the lower and upper bounds for each MIC range -- lower if the predicted MIC is below the range and upper if the predicted MIC is above the range.

    This turns the problem into an ordinal regression

    y_test and y_pred are log-transformed. lower_bounds and upper_bounds are NOT, but they will be log-transformed in the wrapper function after adding a tiny amount to lower_bounds to make it non-zero
    '''

    # lower_bounds /= 2
    # upper_bounds *= 2

    # add a tiny amount so they can be log-transformed. 1e-10 is well below the smallest MIC measured for any drug
    lower_bounds[lower_bounds==0] += 1e-10
    
    # take log2. There is only natural log in tensorflow backend, so use the change of base formula
    lower_bounds = tf.squeeze(K.log(lower_bounds) / K.log(K.constant(2, shape=len(lower_bounds), dtype=tf.float64)))
    upper_bounds = tf.squeeze(K.log(upper_bounds) / K.log(K.constant(2, shape=len(upper_bounds), dtype=tf.float64)))

    def boundedLoss_CNN_helper(y_true, y_pred):

        # ensure same types of everything -- also remove extra dimension of 1
        y_pred = tf.squeeze(tf.cast(y_pred, tf.float64))
        
        # this returns the lower bound, upper bound, or value itself
        bound_to_compute_error = K.clip(y_pred, lower_bounds, upper_bounds)

        # compute the errors first using the log-MICs, based on the desired loss type
        if loss_type == "L1":
            errors = tf.squeeze(K.abs(bound_to_compute_error - y_pred))
        elif loss_type == "L2":
            errors = tf.squeeze(K.square(bound_to_compute_error - y_pred))
        else:
            raise RuntimeError(f"{loss_type} is not a valid loss function type")

        ########## FOR MOST ISOLATES, THIS NEXT STEP IS NOT NEEDED. BUT CLEANER TO DO IT FOR ALL ##########
        # when K.clip is run on arrays with infinities, the non-infinity value is returned
        # so i.e. if y_pred = 1, lower = 0.5, and upper = inf, K.clip returns lower, and the function will compute an error for that sample
        # this is not correct though, this sample should have compute_error = 0. Therefore we need to do this step for all cases
        # assign 1 to predicted points that are less than the lower bound or greater than the upper bound. 
        # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
        outside_bounds_mask = tf.cast(K.less_equal(y_pred, lower_bounds) | K.greater(y_pred, upper_bounds), tf.float64)

        # multiply so that the points predicted in their bin are multiplied by 0 so they have 0 error
        masked_errors = outside_bounds_mask * errors

        # return the sum of the errors of only points that are predicted outside of their bin
        # return sum because in the training loop it will be divided by the total number of points in each batch
        return K.sum(masked_errors)
            
    return boundedLoss_CNN_helper      




def conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0):
    
    cnn_input = tf.keras.Input(shape=(5, longest_locus, num_loci), name='seq_input')
    
    # first perform convolutions and max pooling as in the original model. 
    x = layers.Conv2D(64, (5, filter_size), data_format='channels_last', activation='relu', input_shape=(5, longest_locus, num_loci), name='conv1')(cnn_input)
    x = layers.Conv2D(64, (1, filter_size), activation='relu', name='conv2')(x)

    conv_block_1 = layers.MaxPooling2D((1,3), name='maxPooling1')(x)

    y = layers.Conv2D(32, (1,3), activation='relu', name='conv3')(conv_block_1)
    y = layers.Conv2D(32, (1,3), activation='relu', name='conv4')(y)

    conv_block_2 = layers.MaxPooling2D((1,3), name='maxPooling2')(y)
        
    if additional_data_len > 0:
        print(f"{additional_data_len} features in the MLP block")
        cnn_output = layers.Flatten(name='flatten')(conv_block_2)
        mlp_input = tf.keras.Input(shape=(additional_data_len, ), name='mlp_input')
        dense_inputs = layers.concatenate([cnn_output, mlp_input], axis=1, name='concatenate')
    else:
        dense_inputs = layers.Flatten(name='flatten')(conv_block_2)

    # if you put kernel_regularizer='l2', the default strength is 0.01
    if reg_strength != 0:
        print(f"    Using L2 regularization with {reg_strength} strength")
        dense = layers.Dense(256, activation='relu', name='dense1', kernel_regularizer=regularizers.L2(reg_strength))(dense_inputs)
        dense = layers.Dense(256, activation='relu', name='dense2', kernel_regularizer=regularizers.L2(reg_strength))(dense)
    else:
        dense = layers.Dense(256, activation='relu', name='dense1')(dense_inputs)
        dense = layers.Dense(256, activation='relu', name='dense2')(dense)
        
    if binary:
        output = layers.Dense(1, activation='sigmoid', name='output')(dense)
    else:
        output = layers.Dense(1, activation=None, name='output')(dense)

    inputs_lst = [cnn_input]
    
    if additional_data_len > 0:
        inputs_lst += [mlp_input]
            
    # add bounds to the inputs list if True
    if bounded_loss:
        lower_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='lower_bounds')
        upper_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='upper_bounds')

        inputs_lst += [lower_bounds]
        inputs_lst += [upper_bounds]

    if len(inputs_lst) == 1:
        inputs_lst = inputs_lst[0]

    return tf.keras.Model(inputs=inputs_lst, outputs=output)




def multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_genes, additional_data_len, bounded_loss, filter_size, reg_strength=0):
    
    nt_cnn_input = tf.keras.Input(shape=(5, longest_locus, num_loci), name='nt_seq_input')
    aa_cnn_input = tf.keras.Input(shape=(3, longest_protein, num_genes), name='aa_biophys_input')

    ######################### nucleotide convolutions and max pooling #########################
    nt_cnn_output = layers.Conv2D(64, (5, filter_size), data_format='channels_last', activation='relu', input_shape=(5, longest_locus, num_loci), name='nt_conv1')(nt_cnn_input)
    nt_cnn_output = layers.Conv2D(64, (1, filter_size), activation='relu', name='nt_conv2')(nt_cnn_output)
    nt_cnn_output = layers.MaxPooling2D((1,3), name='nt_maxPooling1')(nt_cnn_output)

    nt_cnn_output = layers.Conv2D(32, (1,3), activation='relu', name='nt_conv3')(nt_cnn_output)
    nt_cnn_output = layers.Conv2D(32, (1,3), activation='relu', name='nt_conv4')(nt_cnn_output)
    nt_cnn_output = layers.MaxPooling2D((1,3), name='nt_maxPooling2')(nt_cnn_output)

    ######################### amino acid convolutions and max pooling #########################
    aa_cnn_output = layers.Conv2D(64, (3, filter_size), data_format='channels_last', activation='relu', input_shape=(3, longest_protein, num_genes), name='aa_conv1')(aa_cnn_input)
    aa_cnn_output = layers.Conv2D(64, (1, filter_size), activation='relu', name='aa_conv2')(aa_cnn_output)
    aa_cnn_output = layers.MaxPooling2D((1,3), name='aa_maxPooling1')(aa_cnn_output)

    aa_cnn_output = layers.Conv2D(32, (1,3), activation='relu', name='aa_conv3')(aa_cnn_output)
    aa_cnn_output = layers.Conv2D(32, (1,3), activation='relu', name='aa_conv4')(aa_cnn_output)
    aa_cnn_output = layers.MaxPooling2D((1,3), name='aa_maxPooling2')(aa_cnn_output)

    # flatten both outputs, then concatenate them
    dense_inputs = layers.concatenate([layers.Flatten(name='nt_flatten')(nt_cnn_output), layers.Flatten(name='aa_flatten')(aa_cnn_output)], axis=1, name='concatenate_cnn_inputs')

    if additional_data_len > 0:
        print(f"{additional_data_len} features in the MLP block")                                        
        mlp_input = tf.keras.Input(shape=(additional_data_len, ), name='mlp_input')

        # combine data for MLP only (no convolving) with the dense_ouputs created from both the NT and AA convolutional outputs
        dense_inputs = layers.concatenate([dense_inputs, mlp_input], axis=1, name='concatenate_cnn_mlp_inputs')        

    # if you put kernel_regularizer='l2', the default strength is 0.01
    if reg_strength != 0:
        print(f"    Using L2 regularization with {reg_strength} strength")
        dense = layers.Dense(256, activation='relu', name='dense1', kernel_regularizer=regularizers.L2(reg_strength))(dense_inputs)
        dense = layers.Dense(256, activation='relu', name='dense2', kernel_regularizer=regularizers.L2(reg_strength))(dense)
    else:
        dense = layers.Dense(256, activation='relu', name='dense1')(dense_inputs)
        dense = layers.Dense(256, activation='relu', name='dense2')(dense)
        
    if binary:
        output = layers.Dense(1, activation='sigmoid', name='output')(dense)
    else:
        output = layers.Dense(1, activation=None, name='output')(dense)

    # create list of all inputs
    inputs_lst = [nt_cnn_input, aa_cnn_input]
    
    if additional_data_len > 0:
        inputs_lst += [mlp_input]
            
    # add bounds to the inputs list if True
    if bounded_loss:
        lower_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='lower_bounds')
        upper_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='upper_bounds')

        inputs_lst += [lower_bounds]
        inputs_lst += [upper_bounds]

    return tf.keras.Model(inputs=inputs_lst, outputs=output)




@tf.function
def train_step(model, optimizer, loss_type, x, y):
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
def val_step(model, loss_type, x, y):
        
    # the bounds are the last 2 elements of the x list
    lower_bounds, upper_bounds = x[-2:]
    
    # y_hat = model.predict(x)
    y_hat = model(x, training=False)
    
    # return loss and complete error. quantLoss_CNN returns a numpy object from a tensor
    return boundedLoss_CNN(lower_bounds, upper_bounds, loss_type)(y, y_hat).numpy(), quantLoss_CNN(y, y_hat, loss_type)
    



def train_single_CNN(model, loss_type, N_epochs, train_generator, val_generator, num_train, num_val, save_model_fName, save_history_fName=None, patience_epochs=None, return_min_loss=False):

    output_path = os.path.dirname(save_model_fName)
    
    optimizer = Adam(learning_rate = np.exp(-1.0 * 9))
    
    if patience_epochs is None:
        print(f"    Training the model with an {loss_type} loss for {N_epochs} epochs")
    else:
        print(f"    Using early stopping with an {loss_type} loss and a delay of {patience_epochs} epochs")
    
    # manual implementation of model callbacks
    patience_counter = 0
    min_loss = 1e3
    
    # initialize lists to store losses
    train_loss = []
    train_error = []
    val_loss = []
    val_error = []
    
    history = pd.DataFrame(columns=["loss", "error", "val_loss", "val_error"])

    for epoch in range(N_epochs):
                   
        # list to keep track of the losses for each batch
        train_epoch_loss = []
        train_epoch_error = []
        val_epoch_loss = []
        val_epoch_error = []
        
        # training loop
        for x_batch_train, y_batch_train in train_generator:
                    
            # compute loss and error. These are sums over the points in the batch
            loss, error = train_step(model, optimizer, loss_type, x_batch_train, y_batch_train)

            if pd.isnull(loss) or pd.isnull(error):
                raise ValueError("Losses are NA. Please check!!!")
            
            train_epoch_loss.append(loss)
            train_epoch_error.append(error)
            
        # store losses for the epoch -- mean of all the batches
        train_loss.append(np.sum(train_epoch_loss) / num_train)   
        train_error.append(np.sum(train_epoch_error) / num_train)
    
        # validation loop -- iterate through all batches
        for x_batch_val, y_batch_val in val_generator:
                            
            # compute loss and error
            loss, error = val_step(model, loss_type, x_batch_val, y_batch_val)
            
            val_epoch_loss.append(loss)
            val_epoch_error.append(error)
          
        # store the mean loss of the batch
        val_loss.append(np.sum(val_epoch_loss) / num_val)
        val_error.append(np.sum(val_epoch_error) / num_val)
    
        history.loc[epoch, :] = [train_loss[-1], train_error[-1], val_loss[-1], val_error[-1]]    
        
        if patience_epochs is not None:
            # if loss decreases by at least 1%
            if float((min_loss - val_loss[-1]) / min_loss) >= 0.01:
            
                print(f"Epoch {epoch+1}: Validation loss improved from {min_loss} to {val_loss[-1]}")
                
                # save the model because it is better than the previous iteration
                model.save(save_model_fName)
    
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

    # only save the model if not using early stopping because it doesn't get saved as you go in the above loop
    if patience_epochs is None:
        model.save(save_model_fName)

    # clear all model variables
    K.clear_session()

    # save the history dataframe to see the additional epochs during the patience period. DON'T SAVE THE MODEL because we want the model at the early stop point
    if save_history_fName is not None:
        history.to_csv(save_history_fName, index=False)
    else:
        return history["val_loss"].values
    
    # this is for cross-validation of the regularization parameter: return the losses to keep track which regularization strength is best
    if return_min_loss:
        # if using early stopping, then the loss of the trained model is the min_loss (because it got updated)
        if patience_epochs is not None:
            return min_loss
        # if not using early stopping, then the model loss is the last value in val_loss (because you trained the model for a specified number of epochs and are taking the last loss)
        else:
            return val_loss[-1]



def boundedLoss_Reg(y_pred, y_true, lower_bounds, upper_bounds, loss_type="L1"):
    '''
    This function is used to select the regularization strength of a Regression model. It is the Regression analog
    of boundedLoss_CNN
    
    y_test and y_pred are log2-transformed. lower_bounds and upper_bounds are NOT
    loss_type is L1 or L2, specifying whether to return the MAE or MSE
    '''

    # lower_bounds /= 2
    # upper_bounds *= 2

    # add a tiny amount so they can be log-transformed, then log-transform
    lower_bounds[lower_bounds==0] += 1e-10
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)

    # determine whether to compute error from the bounds or if the error is 0. 
    bound_to_compute_error = np.clip(y_pred, lower_bounds, upper_bounds)

    # compute the errors first using the log-MICs, based on the desired loss type
    # if you want to compute error relative to the midpoint, replace bound_to_compute_error with y_true
    if loss_type == "L1":
        errors = np.abs(bound_to_compute_error - y_pred)
    elif loss_type == "L2":
        errors = np.exp2(bound_to_compute_error - y_pred)
    else:
        raise RuntimeError(f"{loss_type} is not a valid loss function type")

    # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
    outside_bounds_mask = (np.less_equal(y_pred, lower_bounds) | np.greater(y_pred, upper_bounds)).astype(int)

    # multiply so that the points predicted in their bin are multiplied by 0 so they have 0 error
    masked_errors = outside_bounds_mask * errors

    # because we used np.clip above, the value in the errors array for points predicted within the MIC bin will be 0, so just take the simple mean
    return np.mean(masked_errors)




def create_summary_df(df_test, y_pred, drug, binary_thresh, num_loci, model_name, binarize=True, save_fName=None):
    
    # predictions dataframe: get indices of validation data in the cv splits
    pred_df = df_test[["ROLLINGDB_ID", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound", "Span_CC"]]

    # rename columns to make them easier to read
    pred_df.rename(columns={"ROLLINGDB_ID": "Isolate", 
                            f"{drug}_midpoint": "y_test",
                            f"{drug}_lower_bound": "lower",
                            f"{drug}_upper_bound": "upper"
                           }, 
                   inplace=True
                  )

    # add model predictions, and log-transform the test values
    pred_df["y_pred"] = np.squeeze(y_pred)
    pred_df["y_test"] = np.log2(pred_df["y_test"])
    
    if save_fName is not None:
        pred_df.to_csv(save_fName, index=False)

    # exclude isolates whose MICs span the CC from the final error calculation. BUT they are saved in the dataframe above to see later
    pred_df = pred_df.query("Span_CC==0").reset_index(drop=True)

    binned_mae, binned_mse, within_doubling = boundedLoss_predict(pred_df, y_pred_col="y_pred", lower_bounds_col="lower", upper_bounds_col="upper")

    summary_df = pd.DataFrame({"Drug": drug,
                               "Model": model_name,
                               "Num_Loci": num_loci,
                               "Binned_MAE": binned_mae,
                               "Binned_MSE": binned_mse,
                               "MAE": np.mean(np.abs(pred_df["y_pred"]-pred_df["y_test"])),
                               "MSE": np.mean(np.square(pred_df["y_pred"]-pred_df["y_test"])),
                               "Within_doubling": within_doubling,
                               "Spearman": st.spearmanr(pred_df["y_pred"], pred_df["y_test"])[0],
                               "Spearman_pval": st.spearmanr(pred_df["y_pred"], pred_df["y_test"])[1],
                              }, index=[0])

    # compute binary metrics using the upper bound
    binary_metrics_df = compute_binary_metrics(pred_df["upper"], pred_df["y_pred"], binary_thresh, binarize=binarize)
    summary_df = pd.concat([summary_df, binary_metrics_df], axis=1)
    return summary_df





def get_train_test_val_lineages(df_train, df_test, df_val=None, lineage_fName="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv"):
    
    lineages = pd.read_csv(lineage_fName, index_col=[0])
    lineages.columns = [f"lineageSNP_{col}" for col in lineages.columns]
    assert len(np.unique(lineages.values)) == 2

    train_lineages = lineages.loc[df_train["ROLLINGDB_ID"].values]
    assert sum(train_lineages.index.values != df_train["ROLLINGDB_ID"].values) == 0

    test_lineages = lineages.loc[df_test["ROLLINGDB_ID"].values]
    assert sum(test_lineages.index.values != df_test["ROLLINGDB_ID"].values) == 0

    # include support for no validation data (i.e. Pyrazinamide)
    if df_val is not None:
        val_lineages = lineages.loc[df_val["ROLLINGDB_ID"].values]
        assert sum(val_lineages.index.values != df_val["ROLLINGDB_ID"].values) == 0
    else:
        val_lineages = None
        
    return train_lineages, test_lineages, val_lineages
    



def prepare_model_inputs(X, model_type, include_lineage, feature_names=None, lineages_matrix=None):
    
    if model_type == "CNN":
        
        if include_lineage:
            model_inputs = [X, lineages_matrix.values, np.zeros(len(X)), np.zeros(len(X))]
        else:
            model_inputs = [X, np.zeros(len(X)), np.zeros(len(X))]
        
    elif model_type == "Regression":
  
        if include_lineage:
            
            # # first combine the dataframes to preserve the features, then get only the features specified in the argument
            # assert sum(X.index.values != lineages_matrix.index.values) == 0
            model_inputs = X.merge(lineages_matrix, left_index=True, right_index=True, how="inner")
            model_inputs = model_inputs[feature_names].values
        else:
            # keep only the features specified in the argument
            model_inputs = X[feature_names].values
        
    else:
        raise ValueError(f"{model_type} is not a valid model type!")
        
    return model_inputs

    


def get_inputs_for_regression(config_file):

    kwargs = yaml.safe_load(open(config_file, "r"))

    df_phenos = pd.read_csv(kwargs["phenotype_file"])
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    drug = kwargs["drug"]
    results_dir = kwargs["output_path"]
    ridge_dir = os.path.join(results_dir, "ridge")
    fasta_dir = kwargs["genotype_input_directory"]
    include_lineage = kwargs["include_lineage"]

    # make dataframes of coordinates
    gene_coords, _ = get_gene_coords(locus_list, fasta_dir)
    h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)    

    # this is for samples that don't have data from the MIC-ML consortium, so there is no validation dataset
    if os.path.isfile(os.path.join(data_dir, "validation_data_for_model.csv")):
        val_data_present = True
    else:
        val_data_present = False

    if val_data_present:

        df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))

        # get the pickle file made for the CNN input
        if not os.path.isfile(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")):
            
            val_matrix = get_new_aln_for_CNN(df_val,
                                            locus_list,
                                            fasta_dir
                                           )
            sparse.save_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz"), sparse.COO(val_matrix))
        else:
            val_matrix = sparse.load_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")).todense()
        
        val_samples = val_matrix.shape[0]  
        one_hot_encodings = val_matrix.shape[1]
        longest_locus = val_matrix.shape[2]
        num_loci = val_matrix.shape[3]
        assert one_hot_encodings == 5

        ref_matrix = sparse.load_npz(f"{results_dir.replace('_lineage', '')}/pkl_sparse_ref.npz").todense()
        
        if os.path.isfile(os.path.join(ridge_dir.replace("_lineage", ""), "val_seq_matrix.pkl")):
            X_val = pd.read_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "val_seq_matrix.pkl"))
    
        else:
            print(f"Creating validation data pickle file")
            
            X_val = []
            
            for locus in locus_list:

                # don't need the reference matrix here
                single_locus_matrix, _ = get_single_locus_Reg_input(locus, locus_list, df_phenos, val_matrix, ref_matrix, h37Rv_coords)
                X_val.append(single_locus_matrix)
        
            X_val = pd.concat(X_val, axis=1)
            X_val.index = df_val["ROLLINGDB_ID"].values
            X_val.to_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "val_seq_matrix.pkl"))

    else:
        df_val = None
        X_val = None

    # read in the pickle file of all the sequence features. This should be of length 5 x ALL nucleotides across all loci
    # this is before anything has been dropped due to redundancy or not being present in the samples
    X_train_test = pd.read_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "full_seq_matrix.pkl"))

    df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)    
    df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)    

    X_train = X_train_test.loc[df_train.ROLLINGDB_ID.values]
    X_test = X_train_test.loc[df_test.ROLLINGDB_ID.values]

    # X_train, X_test, and X_val should all be dataframes read in from pickle files, so the indices are ROLLLINGDB_ID and the columns are features
    return X_train, X_test, X_val, df_train, df_test, df_val



def get_new_aln_for_CNN(df,
                        locus_list,
                        fasta_dir
                       ):
    
    # argument = directory that contains the fasta file
    df_genos = make_genotype_df(locus_list, fasta_dir)
    df_genos.index = [name.split(".")[0] for name in df_genos.index.values]
        
    # the additional new strains to predict MICs for
    df_genos = df_genos.loc[df["ROLLINGDB_ID"].values]
    
    assert len(df_genos) == len(df)

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[locus + "_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))
        
    return create_X(df_genos)


                           

def get_inputs_for_CNN(config_file, keep_idx=None):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    drug = kwargs["drug"]
    locus_list = kwargs["locus_list"]
    results_dir = kwargs["output_path"]
    fasta_dir = kwargs["genotype_input_directory"]
    include_lineage = kwargs["include_lineage"]
    df_phenos = pd.read_csv(kwargs["phenotype_file"])

    binary_thresh = kwargs["binary_thresh"]
    binary = kwargs["binary"]

    # this is for samples that don't have data from the MIC-ML consortium, so there is no validation dataset
    if os.path.isfile(os.path.join(data_dir, "validation_data_for_model.csv")):
        val_data_present = True

        df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))

        if not os.path.isfile(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")):
            
            X_val = get_new_aln_for_CNN(df_val,
                                        locus_list,
                                        fasta_dir
                                       )
            sparse.save_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz"), sparse.COO(X_val))
            
        else:
            X_val = sparse.load_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")).todense()

    else:
        val_data_present = False
        df_val = None
        X_val = None
        
    df_train = df_phenos.query("category=='original_train_set'")
    df_test = df_phenos.query("category=='original_test_set'")    

    X_train_test = sparse.load_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_full.npz")).todense()
    X_train = X_train_test[df_train.index.values]
    X_test = X_train_test[df_test.index.values]

    # these are in the same order as df_train, df_test, and df_val, which are in the same order as X_train, X_test, and X_val
    train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)

    X_train = prepare_model_inputs(X_train, "CNN", include_lineage, feature_names=None, lineages_matrix=train_lineages)
    X_test = prepare_model_inputs(X_test, "CNN", include_lineage, feature_names=None, lineages_matrix=test_lineages)
    
    if val_data_present:
        if keep_idx is not None:
            X_val = X_val[keep_idx, :]

            # lineages matrices have samples as the index for merging
            if include_lineage:
                val_lineages = val_lineages.iloc[keep_idx, :]

            df_val = df_val.iloc[keep_idx, :]
            
        X_val = prepare_model_inputs(X_val, "CNN", include_lineage, feature_names=None, lineages_matrix=val_lineages)

    # X_train, X_test, and X_val should all be numpy arrays (so no indices or columns)
    return X_train, X_test, X_val, df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val




def get_threshold_val(pred_df, pred_col, test_col, spec_thresh=None):
    
    y_prob = pred_df[pred_col].values
    y_test = pred_df[test_col].values
    
    # Test thresholds from 0 to 1, in 0.01 increments
    thresholds = np.linspace(0, 1, 101)
    results_df = pd.DataFrame(columns=["thresh", "sens_spec", "sens", "spec"])
    
    for i, thresh in enumerate(thresholds):

        y_pred = (y_prob > thresh).astype(int)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
        
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        
        results_df.loc[i, :] = [thresh, sens + spec, sens, spec]
        
    # get index of highest sum(s) of sens and spec.
    if spec_thresh is None:
        select_thresh = results_df.sort_values("sens_spec", ascending=False)["thresh"].values[0]
    # if there is a threshold on specificity, then choose the threshold that maximizes sensitivity while having a specificity above the threshold
    else:
        if results_df["spec"].max() >= spec_thresh:
            select_thresh = results_df.query("spec >= @spec_thresh").sort_values("sens", ascending=False)["thresh"].values[0]
        # if there are no cases when the specificity reaches the threshold, take the highest sensitivity given that the specificity is maximized
        else:
            max_spec = results_df["spec"].max()
            select_thresh = results_df.query("spec >= @max_spec").sort_values("sens", ascending=False)["thresh"].values[0]

    print(f"Binarization threshold: {select_thresh}")
    
    # add the labels using the selected threshold
    pred_df["y_pred_label"] = (pred_df[pred_col] > select_thresh).astype(int)    
    return select_thresh, pred_df




def compute_binary_metrics(y_true, y_pred, binary_thresh, binarize=False):
        
    # binarize using the critical concentration
    # see if the upper bound is greater than the critical concentration. If so, resistant. If the upper bound is equal to the CC, then it is susceptible because it dies at the CC.
    if binarize:
        y_true_binary = (y_true > binary_thresh).astype(int)
        y_pred_binary = (y_pred > np.log2(binary_thresh)).astype(int)
    else:
        y_true_binary = np.copy(y_true)
        y_pred_binary = np.copy(y_pred)
        
    assert len(np.unique(y_true_binary)) <= 2
    assert len(np.unique(y_pred_binary)) <= 2
    
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_binary, y_pred_binary).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)
    precision = tp / (tp+fp)
    acc = sklearn.metrics.accuracy_score(y_true_binary, y_pred_binary)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true_binary, y_pred_binary)
    F1 = sklearn.metrics.f1_score(y_true_binary, y_pred_binary)

    return pd.DataFrame({"Sensitivity": sens,
                         "Specificity": spec,
                         "Precision": precision,
                         "Accuracy": acc,
                         "Balanced_Acc": balanced_acc,
                         "F1": F1,
                        }, index=[0]
                       )


    
def compute_proportion_within_1bin(df, y_pred_col, y_true_col, lower_bounds_col, upper_bounds_col, binary_thresh):
    
    df = df.reset_index(drop=True)
    
    # list of all lower and upper bounds from the table
    MIC_vals = list(np.sort(np.unique(np.concatenate([df["lower"].values, df["upper"].values]))))
    max_val = np.max(MIC_vals)
    
    for i, row in df.iterrows():

        pred_MIC, actual_MIC = np.exp2(row[y_pred_col]), np.round(np.exp2(row[y_true_col]), 2)
        lower, upper = row[lower_bounds_col], row[upper_bounds_col]

        if not lower <= actual_MIC:
            print("lower problem", lower, actual_MIC)
            
        if not actual_MIC <= upper:
            print("upper problem", actual_MIC, upper)

        lower_idx = MIC_vals.index(lower)
        upper_idx = MIC_vals.index(upper)

        if lower > 0:
            lower_adj = MIC_vals[lower_idx - 1]
        else:
            lower_adj = 0

        if upper < np.max(df[upper_bounds_col].values):
            upper_adj = MIC_vals[upper_idx + 1]
        else:
            upper_adj = np.max(df[upper_bounds_col].values)

        assert lower_adj < upper_adj
        
        if lower_adj > 0:
            assert lower_adj < lower
        else:
            assert lower_adj <= lower
        
        if upper_adj < max_val:
            assert upper_adj > upper
        else:
            assert upper_adj >= upper
            
        df.loc[i, ["lower_adj", "upper_adj"]] = [lower_adj, upper_adj]

        if pred_MIC >= lower_adj and pred_MIC <= upper_adj:
            df.loc[i, "within_1bin"] = 1
        else:
            df.loc[i, "within_1bin"] = 0

    assert np.nan not in df["within_1bin"].unique()
    df["within_1bin"] = df["within_1bin"].astype(int)

    return df




def boundedLoss_predict(pred_df, y_pred_col="y_pred", lower_bounds_col="lower", upper_bounds_col="upper"):
    '''
    y_true and y_pred are log-MICs. lower_bounds and upper_bounds are exponentiated. 
    
    This function returns bounded MAE, MSE, and the proportion of points measured within 1 MIC doubling (1 log2 unit)
    ''' 
    
    del_cols = [f"{y_pred_col}_exp", "within_doubling", "within_1bin", "compute_error", f"{lower_bounds_col}_rounded", f"{upper_bounds_col}_rounded"]

    for col in del_cols:
        if col in pred_df.columns:
            del pred_df[col]

    # first add essential agreement (proportion within 1 doubling dilution)
    # not always helpful because some "doubling" dilutions are not exact, i.e. 0.3, 0.6, 0.125, 0.5. But the number is here if needed
    pred_df[f"{y_pred_col}_exp"] = np.round(np.exp2(pred_df[y_pred_col]).astype(float), 2)
    
    pred_df.loc[(pred_df[lower_bounds_col] / 2 <= pred_df[f"{y_pred_col}_exp"]) & 
                (pred_df[upper_bounds_col] * 2 >= pred_df[f"{y_pred_col}_exp"])
                , "within_doubling"] = 1

    pred_df.loc[(pred_df[f"{y_pred_col}_exp"] == 0.06) & (pred_df[lower_bounds_col] == 0.12), "within_doubling"] = 1
    pred_df["within_doubling"] = pred_df["within_doubling"].fillna(0).astype(int)
        
    # make copies to avoid changing the original dataframe
    lower_bounds = np.copy(pred_df[lower_bounds_col].values) #pred_df[lower_bounds_col].values / 2
    upper_bounds = np.copy(pred_df[upper_bounds_col].values) #pred_df[upper_bounds_col].values * 2
    
    lower_bounds[lower_bounds==0] += 1e-10
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)

    # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
    pred_df["compute_error"] = ((pred_df[y_pred_col].values <= lower_bounds) | (pred_df[y_pred_col].values > upper_bounds)).astype(int)

    # compute the error relative to the bounds, NOT RELATIVE TO THE MIDPOINT (y_test) of each isolate
    # np.clip returns one of the values from lower_bounds or upper_bounds, whichever is closest to the prediction, if the value is outside the bounds
    # if the test values are within the bounds, the values themselves are returned
    bound_to_compute_error = np.clip(pred_df[y_pred_col].values, lower_bounds, upper_bounds)
    mae = np.mean((np.abs(bound_to_compute_error - pred_df[y_pred_col])))
    mse = np.mean((np.square(bound_to_compute_error - pred_df[y_pred_col])))

    return mae, mse, pred_df["within_doubling"].mean()
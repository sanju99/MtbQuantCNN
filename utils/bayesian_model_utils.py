import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
import tensorflow_probability as tfp
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Ridge, RidgeCV
import sklearn.metrics
    
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")



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




def single_bayesian_2D_conv_layer(num_filters, kernel_size, layer_name):

    # 2D convolution only
    assert len(kernel_size) == 2

    # the posterior is the product of multiple normal distributions. 
    # Because we assume that the weights have independent distributions (because otherwise there is an intractable number of parameters due to different weights covarying with each other), the product of their distributions is the simple product
    # 
    return tfp.layers.Convolution2DReparameterization(num_filters, 
                                                      kernel_size, 
                                                      data_format='channels_last', 
                                                      activation='relu', 
                                                       # kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(), # defualt posterior.
                                                       # kernel_prior_fn=tfp.layers.default_multivariate_normal_fn(), # default prior
                                                      name=layer_name
                                                     )



def single_bayesian_dense_layer(num_nodes, activation, layer_name):
    '''
    There are multiple types of dense layers available in tensorflow-probability. 
    '''

    return tfp.layers.DenseVariational(num_nodes, 
                                       activation=activation, 
                                       # kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(), # defualt posterior.
                                       # kernel_prior_fn=tfp.layers.default_multivariate_normal_fn(), # default prior
                                       name=layer_name
                                      )



def bayesian_multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_genes, additional_data_len, bounded_loss, filter_size, reg_strength=0):

    # # Define the posterior distribution (mean and stddev parameters)
    # def posterior_func(kernel_size, bias_size, dtype=None):
    #     n = kernel_size + bias_size
    #     posterior_model = tf.keras.Sequential([
    #         layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(n)),
    #         tfp.layers.MultivariateNormalTriL(n)
    #     ])
    #     return posterior_model

    nt_cnn_input = tf.keras.Input(shape=(5, longest_locus, num_loci), name='nt_seq_input')
    aa_cnn_input = tf.keras.Input(shape=(3, longest_protein, num_genes), name='aa_biophys_input')

    ######################### Bayesian nucleotide convolutions and max pooling #########################

    # two layers of 2D convolutions
    nt_cnn_output = single_bayesian_2D_conv_layer(64, (5,filter_size), 'nt_conv1')(nt_cnn_input)
    # nt_cnn_output = single_bayesian_2D_conv_layer(64, (1,filter_size), name='nt_conv2')(nt_cnn_output)

    # # one max pool layer
    # nt_cnn_output = layers.MaxPooling2D((1,3), name='nt_maxPooling1')(nt_cnn_output)

    # # two more layers of 2D convolutions
    # nt_cnn_output = single_bayesian_2D_conv_layer(32, (1,3), 'nt_conv3')(nt_cnn_output)
    # nt_cnn_output = single_bayesian_2D_conv_layer(32, (1,3), 'nt_conv4')(nt_cnn_output)

    # # one more max pool layer -- not Bayesian because there are no weight distributions to learn here, we're just taking the maximum of the distribution in the previous layer
    # nt_cnn_output = layers.MaxPooling2D((1,3), name='nt_maxPooling2')(nt_cnn_output)

    ######################### Bayesian amino acid convolutions and max pooling #########################
    
    # two layers of 2D convolutions
    aa_cnn_output = single_bayesian_2D_conv_layer(64, (3,filter_size), 'aa_conv1')(aa_cnn_input)
    # aa_cnn_output = single_bayesian_2D_conv_layer(64, (1,filter_size), 'aa_conv2')(aa_cnn_input)

    # # one max pool layer
    # aa_cnn_output = layers.MaxPooling2D((1,3), name='aa_maxPooling1')(aa_cnn_output)

    # # two more layers of 2D convolutions
    # aa_cnn_output = single_bayesian_2D_conv_layer(32, (1,3), 'aa_conv3')(aa_cnn_output)
    # aa_cnn_output = single_bayesian_2D_conv_layer(32, (1,3), 'aa_conv4')(aa_cnn_output)

    # # one more max pool layer
    # aa_cnn_output = layers.MaxPooling2D((1,3), name='aa_maxPooling2')(aa_cnn_output)

    # Flatten both outputs, then concatenate them
    dense_inputs = layers.concatenate([layers.Flatten(name='nt_flatten')(nt_cnn_output),
                                       layers.Flatten(name='aa_flatten')(aa_cnn_output)], axis=1, name='concatenate_cnn_inputs')

    if additional_data_len > 0:
        
        print(f"{additional_data_len} features in the MLP block")
        mlp_input = tf.keras.Input(shape=(additional_data_len,), name='mlp_input')

        # Combine data for MLP only (no convolving) with the dense outputs from both the NT and AA convolutional outputs
        dense_inputs = layers.concatenate([dense_inputs, mlp_input], axis=1, name='concatenate_cnn_mlp_inputs')

    # two fully connected (dense) layers with 256 nodes each
    dense = single_bayesian_dense_layer(256, 'relu', 'dense1')(dense_inputs)
    dense = single_bayesian_dense_layer(256, 'relu', 'dense2')(dense)

    # output layer with 1 node because this is a single outcome model, no activation (linear activation)
    output = single_bayesian_dense_layer(1, None, 'output')(dense)

    # Create list of all inputs
    inputs_lst = [nt_cnn_input, aa_cnn_input]
    
    if additional_data_len > 0:
        inputs_lst += [mlp_input]
            
    # Add bounds to the inputs list if True
    if bounded_loss:
        lower_bounds = tf.keras.Input(shape=(1,), dtype=tf.float64, name='lower_bounds')
        upper_bounds = tf.keras.Input(shape=(1,), dtype=tf.float64, name='upper_bounds')

        inputs_lst += [lower_bounds]
        inputs_lst += [upper_bounds]

    return tf.keras.Model(inputs=inputs_lst, outputs=output)
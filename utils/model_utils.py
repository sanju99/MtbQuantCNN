import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.linear_model import Ridge, RidgeCV
import sklearn.metrics    
    


def quantLoss_CNN(y_true, y_pred, loss_type):
    '''
    This function returns MAE or MSE (unbounded) for a model.
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




# def boundedLoss_CNN(lower_bounds, upper_bounds, loss_type):
#     '''
#     The bounds are in exponentiated form because some lower bounds are 0. So when computing the loss, y_pred must be exponentiated
#     '''

#     # add a tiny amount so they can be log-transformed
#     lower_bounds[lower_bounds==0] += 1e-6
    
#     # take log2. There is only ln in tensorflow backend, so use the change of base formula
#     lower_bounds = tf.squeeze(K.log(lower_bounds) / K.log(K.constant(2, shape=len(lower_bounds), dtype=tf.float64)))
#     upper_bounds = tf.squeeze(K.log(upper_bounds) / K.log(K.constant(2, shape=len(upper_bounds), dtype=tf.float64)))
    
#     def boundedLoss_CNN_helper(y_true, y_pred):
#         '''
#         y_test and y_pred are log-transformed. lower_bounds and upper_bounds are NOT
#         '''
#         # ensure same types of everything
#         y_true = tf.squeeze(tf.cast(y_true, tf.float64))
#         y_pred = tf.squeeze(tf.cast(y_pred, tf.float64))

#         # this returns the lower bound, upper bound, or value itself
#         # if the predicted value is less than the lower bound, return lower
#         # if prediction > upper bound, return upper
#         # if lower <= prediction <= upper, return the value
#         bound_to_compute_error = K.clip(y_pred, lower_bounds, upper_bounds)

#         # compute the errors first using the log-MICs, based on the desired loss type
#         if loss_type == "L1":
#             errors = tf.squeeze(K.abs(bound_to_compute_error - y_pred))
#         elif loss_type == "L2":
#             errors = tf.squeeze(K.square(bound_to_compute_error - y_pred))
#         else:
#             raise RuntimeError(f"{loss_type} is not a valid loss function type")
        
#         # assign 1 to predicted points that are less than the lower bound or greater than the upper bound. 
#         outside_bounds_mask = tf.cast(K.less(y_pred, lower_bounds) | K.greater(y_pred, upper_bounds), tf.float64)

#         # multiply so that the points predicted in their bin are multiplied by 0 so they have 0 error
#         masked_errors = outside_bounds_mask * errors

#         # return the sum of the errors of only points that are predicted outside of their bin
#         # return sum because when iterating through batches it will be divided by the total number of points in each batch
#         return K.sum(masked_errors)
    
#     return boundedLoss_CNN_helper



def boundedLoss_CNN(lower_bounds, upper_bounds, loss_type):
    '''
    The bounds are in exponentiated form because some lower bounds are 0. So when computing the loss, y_pred must be exponentiated
    '''

    def boundedLoss_CNN_helper(y_true, y_pred):
        '''
        y_test and y_pred are log-transformed. lower_bounds and upper_bounds are NOT
        '''

        # ensure same types of everything
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        # exponentiate the predictions to get actual MICs
        y_pred_MIC = tf.squeeze(K.pow(2, y_pred))

        # compute the errors first using the log-MICs, based on the desired loss type
        if loss_type == "L1":
            errors = tf.squeeze(K.abs(y_true - y_pred))
        elif loss_type == "L2":
            errors = tf.squeeze(K.square(y_true - y_pred))
        else:
            raise RuntimeError(f"{loss_type} is not a valid loss function type")
        
        # assign 1 to predicted points that are less than the lower bound or greater than the upper bound. 
        outside_bounds_mask = tf.cast(K.less(y_pred_MIC, lower_bounds) | K.greater(y_pred_MIC, upper_bounds), tf.float64)

        # multiply so that the points predicted in their bin are multiplied by 0 so they have 0 error
        masked_errors = outside_bounds_mask * errors

        # return the sum of the errors of only points that are predicted outside of their bin
        # return sum because in the training loop it will be divided by the total number of points in each batch
        return K.sum(masked_errors)
    
    return boundedLoss_CNN_helper  
        
        



def conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size):
    
    cnn_input = tf.keras.Input(shape=(5, longest_locus, num_loci), name='seq_input')
    
    # first perform convolutions and max pooling as in the original model. 
    x = layers.Conv2D(64, (5, filter_size), data_format='channels_last', activation='relu', input_shape=(5, longest_locus, num_loci), name='conv1')(cnn_input)
    x = layers.Conv2D(64, (1,12), activation='relu', name='conv2')(x)

    conv_block_1 = layers.MaxPooling2D((1,3), name='maxPooling1')(x)

    y = layers.Conv2D(32, (1,3), activation='relu', name='conv3')(conv_block_1)
    y = layers.Conv2D(32, (1,3), activation='relu', name='conv4')(y)

    conv_block_2 = layers.MaxPooling2D((1,3), name='maxPooling2')(y)

    # flattened output of convolutional block. Concatenate this with the lineages, then pass into dense layers
    if num_lineages > 0:
        cnn_output = layers.Flatten(name='flatten')(conv_block_2)
        mlp_input = tf.keras.Input(shape=(num_lineages, ), name='lineage_input')
        dense_inputs = layers.concatenate([cnn_output, mlp_input], axis=1, name='concatenate')
    else:
        dense_inputs = layers.Flatten(name='flatten')(conv_block_2)

    # change regularization strength
    # kernel_regularizer=regularizers.L2(0.01)
    
    # dense = layers.Dense(256, activation='relu', name='dense1', kernel_regularizer='l2')(dense_inputs)
    # dense = layers.Dense(256, activation='relu', name='dense2', kernel_regularizer='l2')(dense)
    dense = layers.Dense(256, activation='relu', name='dense1')(dense_inputs)
    dense = layers.Dense(256, activation='relu', name='dense2')(dense)
    
    if binary:
        print("Fitting binary model")
        output = layers.Dense(1, activation='sigmoid', name='output')(dense)
    else:
        print("Fitting quantitative model")
        output = layers.Dense(1, activation=None, name='output')(dense)

    if num_lineages > 0:
        inputs_lst = [cnn_input, mlp_input]
    else:
        inputs_lst = [cnn_input]
    
    # add bounds to the inputs list if True
    if bounded_loss:
        lower_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='lower_bounds')
        upper_bounds = tf.keras.Input(shape=(1, ), dtype=tf.float64, name='upper_bounds')

        inputs_lst.append(lower_bounds)
        inputs_lst.append(upper_bounds)

    if len(inputs_lst) == 1:
        inputs_lst = inputs_lst[0]
        
    return tf.keras.Model(inputs=inputs_lst, outputs=output)



    
# class CustomRidgeCV(RidgeCV):
                
#     def fit(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None, *args, **kwargs):
        
#         self.loss_type = loss_type
        
#         # processing of bounds. first add a tiny amount so they can be log-transformed
#         lower_bounds[lower_bounds==0] += 1e-6
        
#         # take log2. There is only ln in tensorflow backend, so use the change of base formula
#         self.lower_bounds = np.log2(lower_bounds)
#         self.upper_bounds = np.log2(upper_bounds)
        
#         super().fit(X, y, *args, **kwargs)
        
#     def score(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None):
        
#         self.loss_type = loss_type
#         self.lower_bounds = lower_bounds
#         self.upper_bounds = upper_bounds
        
#         def boundedLoss_Reg(y_pred, y_true):

#             '''
#             y_test and y_pred are log2-transformed. lower_bounds and upper_bounds are NOT
#             loss_type is L1 or L2, specifying whether to return the MAE or MSE
#             reg_param is the strength of regularization to apply -- multiply the sum of the squares of sample_weights by this term
#             '''

#             bound_to_compute_error = np.clip(y_pred, lower_bounds, upper_bounds)

#             # compute the errors first using the log-MICs, based on the desired loss type
#             if self.loss_type == "L1":
#                 errors = np.abs(bound_to_compute_error - y_pred)
#             elif self.loss_type == "L2":
#                 errors = np.exp2(bound_to_compute_error - y_pred)
#             else:
#                 raise RuntimeError(f"{self.loss_type} is not a valid loss function type")

#             # compute error using only the points that are predicted outside of their bin. Sum the errors, then divide by the number of points
#             binned_error = np.sum(errors[((y_pred < self.lower_bounds) | (y_pred > self.upper_bounds))]) / len(y_pred)
#             return binned_error + self.alpha * np.sum(np.square(self.coef_))
        
#         y_pred = self.predict(X)
#         return -boundedLoss_Reg(y_pred, y)



class CustomRidgeCV(RidgeCV):
                
    def fit(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None, *args, **kwargs):
        
        self.loss_type = loss_type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        super().fit(X, y, *args, **kwargs)
        
    def score(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None):
        
        self.loss_type = loss_type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        def boundedLoss_Reg(y_pred, y_true):

            '''
            y_test and y_pred are log2-transformed. lower_bounds and upper_bounds are NOT
            loss_type is L1 or L2, specifying whether to return the MAE or MSE
            reg_param is the strength of regularization to apply -- multiply the sum of the squares of sample_weights by this term
            '''

            # get predictions, then exponentiate to get actual MICs
            y_pred_MIC = np.exp2(y_pred)

            # compute the errors first using the log-MICs, based on the desired loss type
            if self.loss_type == "L1":
                errors = np.abs(y_true - y_pred)
            elif self.loss_type == "L2":
                errors = (y_true - y_pred)**2
            else:
                raise RuntimeError(f"{self.loss_type} is not a valid loss function type")

            # compute error using only the points that are predicted outside of their bin. Sum the errors, then divide by the number of points
            binned_error = np.sum(errors[((y_pred_MIC < self.lower_bounds) | (y_pred_MIC > self.upper_bounds))]) / len(y_pred_MIC)
            return binned_error + self.alpha * np.sum(np.square(self.coef_))
        
        y_pred = self.predict(X)
        return -boundedLoss_Reg(y_pred, y)
    
    
    
class CustomRidge(Ridge):
                
    def fit(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None, *args, **kwargs):
        
        self.loss_type = loss_type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        super().fit(X, y, *args, **kwargs)
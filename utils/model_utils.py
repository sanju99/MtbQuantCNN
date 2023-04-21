import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import Sequence
import scipy.stats as st
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, balanced_accuracy_score
    
    
    
def get_threshold_val(pred_df, pred_col, test_col):
    
    y_pred = pred_df[pred_col].values
    y_test = pred_df[test_col].values
    
    # Compute number resistant and sensitive
    num_samples = len(pred_df)
    num_resistant = np.sum(y_test).astype(int)
    num_sensitive = num_samples - num_resistant

    # Test thresholds from 0 to 1, in 0.01 increments
    thresholds = np.linspace(0, 1, 101)
    
    fpr_ = []
    tpr_ = []

    for thresh in thresholds:
        
        # binarize using the threshold, then compute true and false positives
        pred_df["y_pred_label"] = (pred_df[pred_col] > thresh).astype(int)
        
        tp = len(pred_df.loc[(pred_df["y_pred_label"] == 1) & (pred_df[test_col] == 1)])
        fp = len(pred_df.loc[(pred_df["y_pred_label"] == 1) & (pred_df[test_col] == 0)])

        # Compute FPR and TPR. FPR = FP / N. TPR = TP / P
        fpr_.append(fp / num_sensitive)
        tpr_.append(tp / num_resistant)

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    sens_spec_sum = (1 - fpr_) + tpr_

    # get index of highest sum(s) of sens and spec. Arbitrarily take the first threshold when there are multiple
    best_sens_spec_sum_idx = np.where(sens_spec_sum == np.max(sens_spec_sum))[0][0]
    select_thresh = thresholds[best_sens_spec_sum_idx]
    print(f"Binarization threshold: {select_thresh}")

    # add the labels using the selected threshold
    pred_df["y_pred_label"] = (pred_df[pred_col] > select_thresh).astype(int)    
    return pred_df



def compute_binary_metrics(y_val, y_pred, binary_thresh, binarize=False):
        
    # binarize using the critical concentration
    if binarize:
        y_val_binary = (y_val > np.log2(binary_thresh)).astype(int)
        y_pred_binary = (y_pred > np.log2(binary_thresh)).astype(int)
    else:
        y_val_binary = np.copy(y_val)
        y_pred_binary = np.copy(y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_val_binary, y_pred_binary).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)
    auc = roc_auc_score(y_val_binary, y_pred_binary)
    auc_pr = average_precision_score(y_val_binary, y_pred_binary, pos_label=1)
    acc = accuracy_score(y_val_binary, y_pred_binary)
    balanced_acc = balanced_accuracy_score(y_val_binary, y_pred_binary)
        
    return pd.DataFrame({"Sensitivity": sens,
                         "Specificity": spec,
                         "AUC": auc,
                         "AUC_PR": auc_pr,
                         "Accuracy": acc,
                         "Balanced_Acc": balanced_acc
                        }, index=[0]
                       )



def class_weighting_dictionary(y):
    '''
    Returns a dictionary of weights for the binary CNN to weight the loss and metrics functions by. 
    '''
    
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                               classes=np.unique(y),
                                               y=y
                                            )
    return dict(zip(np.unique(y), weights))



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




def boundedLoss_CNN(lower_bounds, upper_bounds, loss_type):
    '''
    The bounds are in exponentiated form because some lower bounds are 0. So when computing the loss, y_pred must be exponentiated
    '''

    # add a tiny amount so they can be log-transformed
    lower_bounds[lower_bounds==0] += 1e-6
    
    # take log2. There is only ln in tensorflow backend, so use the change of base formula
    lower_bounds = tf.squeeze(K.log(lower_bounds) / K.log(K.constant(2, shape=len(lower_bounds), dtype=tf.float64)))
    upper_bounds = tf.squeeze(K.log(upper_bounds) / K.log(K.constant(2, shape=len(upper_bounds), dtype=tf.float64)))
    
    def boundedLoss_CNN_helper(y_true, y_pred):
        '''
        y_test and y_pred are log-transformed. lower_bounds and upper_bounds are NOT
        '''
        # ensure same types of everything
        y_true = tf.squeeze(tf.cast(y_true, tf.float64))
        y_pred = tf.squeeze(tf.cast(y_pred, tf.float64))

        # this returns the lower bound, upper bound, or value itself
        # if the predicted value is less than the lower bound, return lower
        # if prediction > upper bound, return upper
        # if lower <= prediction <= upper, return the value
        bound_to_compute_error = K.clip(y_pred, lower_bounds, upper_bounds)

        # compute the errors first using the log-MICs, based on the desired loss type
        if loss_type == "L1":
            errors = tf.squeeze(K.abs(bound_to_compute_error - y_pred))
        elif loss_type == "L2":
            errors = tf.squeeze(K.square(bound_to_compute_error - y_pred))
        else:
            raise RuntimeError(f"{loss_type} is not a valid loss function type")
        
        # assign 1 to predicted points that are less than the lower bound or greater than the upper bound. 
        outside_bounds_mask = tf.cast(K.less(y_pred, lower_bounds) | K.greater(y_pred, upper_bounds), tf.float64)

        # multiply so that the points predicted in their bin are multiplied by 0 so they have 0 error
        masked_errors = outside_bounds_mask * errors

        # return the sum of the errors of only points that are predicted outside of their bin
        # return sum because when iterating through batches it will be divided by the total number of points in each batch
        return K.sum(masked_errors)
    
    return boundedLoss_CNN_helper



def compute_proportion_within_1bin(df, y_pred_col, y_true_col, lower_bounds_col, upper_bounds_col, binary_thresh):
    
    df = df.reset_index(drop=True)
    
    # list of all lower and upper bounds from the table
    MIC_vals = list(np.sort(np.unique(np.concatenate([df["lower"].values, df["upper"].values]))))
    
    for i, row in df.iterrows():

        pred_MIC, actual_MIC = np.exp2(row[y_pred_col]), np.round(np.exp2(row[y_true_col]), 2)
        lower, upper = row[lower_bounds_col], row[upper_bounds_col]

        assert lower <= actual_MIC
        assert actual_MIC <= upper

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
        assert lower_adj < lower
        assert upper_adj > upper

        if pred_MIC >= lower_adj and pred_MIC <= upper_adj:
            df.loc[i, "within_1bin"] = 1
        else:
            df.loc[i, "within_1bin"] = 0

    assert np.nan not in df["within_1bin"].unique()
    df["within_1bin"] = df["within_1bin"].astype(int)
    
    log_binary_thresh = np.log2(binary_thresh)
    df.loc[((df[y_pred_col] < log_binary_thresh) & (df[y_true_col] < log_binary_thresh)) | ((df[y_pred_col] > log_binary_thresh) & (df[y_true_col] > log_binary_thresh)), "binary"] = 1
    df["binary"] = df["binary"].fillna(0).astype(int)
    
    return df



def boundedLoss_predict(pred_df, binary_thresh, y_pred_col="y_pred", y_true_col="y_test", lower_bounds_col="lower", upper_bounds_col="upper"):
    '''
    y_true and y_pred are log-MICs. lower_bounds and upper_bounds are exponentiated. 
    
    This function returns bounded MAE, MSE, and the proportion of points measured within 1 MIC doubling (1 log2 unit)
    ''' 
    
    # make copies to avoid changing the original dataframe
    lower_bounds = np.copy(pred_df[lower_bounds_col].values)
    upper_bounds = np.copy(pred_df[upper_bounds_col].values)
    
    lower_bounds[lower_bounds==0] += 1e-6
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)
    
    bound_to_compute_error = np.clip(pred_df[y_pred_col].values, lower_bounds, upper_bounds)

    pred_df["compute_error"] = ((pred_df[y_pred_col] < lower_bounds) | (pred_df[y_pred_col] > upper_bounds)).astype(int)
    mae = np.mean(pred_df["compute_error"] * (np.abs(bound_to_compute_error - pred_df[y_pred_col])))
    mse = np.mean(pred_df["compute_error"] * (np.square(bound_to_compute_error - pred_df[y_pred_col])))
    
    pred_df = compute_proportion_within_1bin(pred_df, y_pred_col, y_true_col, lower_bounds_col, upper_bounds_col, binary_thresh)
    
    # return error and proportion within 1 doubline of the measured MIC
    return mae, mse, pred_df["within_1bin"].mean()

            
        
        
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



    
class CustomRidgeCV(RidgeCV):
                
    def fit(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None, *args, **kwargs):
        
        self.loss_type = loss_type
        
        # processing of bounds. first add a tiny amount so they can be log-transformed
        lower_bounds[lower_bounds==0] += 1e-6
        
        # take log2. There is only ln in tensorflow backend, so use the change of base formula
        self.lower_bounds = np.log2(lower_bounds)
        self.upper_bounds = np.log2(upper_bounds)
        
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

            bound_to_compute_error = np.clip(y_pred, lower_bounds, upper_bounds)

            # compute the errors first using the log-MICs, based on the desired loss type
            if self.loss_type == "L1":
                errors = np.abs(bound_to_compute_error - y_pred)
            elif self.loss_type == "L2":
                errors = np.exp2(bound_to_compute_error - y_pred)
            else:
                raise RuntimeError(f"{self.loss_type} is not a valid loss function type")

            # compute error using only the points that are predicted outside of their bin. Sum the errors, then divide by the number of points
            binned_error = np.sum(errors[((y_pred < self.lower_bounds) | (y_pred > self.upper_bounds))]) / len(y_pred)
            return binned_error + self.alpha * np.sum(np.square(self.coef_))
        
        y_pred = self.predict(X)
        return -boundedLoss_Reg(y_pred, y)
    
    
    
    
class CustomRidge(Ridge):
                
    def fit(self, X, y, loss_type=None, lower_bounds=None, upper_bounds=None, *args, **kwargs):
        
        self.loss_type = loss_type
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        super().fit(X, y, *args, **kwargs)
        
        
        
        
def create_summary_df(df_test, y_pred, drug, binary_thresh, num_loci, model_name, binarize=True, save_fName=None):
    
    # predictions dataframe: get indices of validation data in the cv splits
    pred_df = df_test[["ROLLINGDB_ID", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound"]]

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

    binned_mae, binned_mse, within_1bin = boundedLoss_predict(pred_df, binary_thresh)
    
    if save_fName is not None:
        pred_df.to_csv(save_fName, index=False)

    summary_df = pd.DataFrame({"Drug": drug,
                               "Model": model_name,
                               "Num_Loci": num_loci,
                               "Binned_MAE": binned_mae,
                               "Binned_MSE": binned_mse,
                               "MAE": np.mean(np.abs(pred_df["y_pred"]-pred_df["y_test"])),
                               "MSE": np.mean(np.square(pred_df["y_pred"]-pred_df["y_test"])),
                               "Within_1Bin": within_1bin,
                               "Spearman": st.spearmanr(pred_df["y_pred"], pred_df["y_test"])[0],
                               "Pearson": st.pearsonr(pred_df["y_pred"], pred_df["y_test"])[0],
                              }, index=[0])

    binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred"], binary_thresh, binarize=binarize)
    summary_df = pd.concat([summary_df, binary_metrics_df], axis=1)
    return summary_df
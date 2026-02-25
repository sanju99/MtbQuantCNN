import numpy as np
import pandas as pd
import glob, os, subprocess, vcf, shutil, sparse, yaml, sys, argparse, pickle, warnings, tracemalloc
import scipy.stats as st
warnings.filterwarnings("ignore")
import tensorflow as tf

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from model_utils import *

import xgboost as xgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")

def check_gpu_availability():
    if len(tf.config.list_physical_devices('GPU')) > 0:
        return True
    return False

# Check for GPU availability
gpu_available = check_gpu_availability()

print(f"GPU available: {gpu_available}")

# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

# boolean argument for including lineage SNPs, default value False. If you include the flag, it is considered True
parser.add_argument('--lineage', action='store_true', help='Flag to add lineage SNPs to model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--tier2', action='store_true', help='Flag to add tier 2 loci to the model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--amino-acid', dest='amino_acid', action='store_true', help='Flag to add amino acid biophysical properties to the model')

parser.add_argument('--custom_loss', dest='custom_loss', action='store_true', help='Use the custom loss function')

parser.add_argument('--train', dest='train_model', action='store_true', help='If true, train a model on the full data.')

parser.add_argument('--CV', dest='perform_cross_validation', action='store_true', help="If true, perform 5-fold cross-validation.")

parser.add_argument('--epochs', default=10000, type=int, help='Maximum number of epochs to train the model')

parser.add_argument('--patience', default=200, type=int, help='Number of patience epochs for model training')

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--binary', dest='binary', action='store_true', help='If specified, train a binary model, not an MIC model (default)')

parser.add_argument('--loss', dest='loss_type', default="L1", type=str, help='Must be one of L1 or L2 to specify the model loss (mean absolute error or mean squared error)')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
custom_loss = cmd_line_args.custom_loss
train_model = cmd_line_args.train_model
perform_cross_validation = cmd_line_args.perform_cross_validation
N_epochs = cmd_line_args.epochs
patience_epochs = cmd_line_args.patience
AF_thresh = cmd_line_args.AF_thresh
binary = cmd_line_args.binary
loss_type = cmd_line_args.loss_type

# use the non-75% AF thresh for the test data generator if specified
if AF_thresh > 1:
    AF_thresh /= 100

kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
tier1_loci = kwargs["tier1_loci"]

if include_tier2:
    tier2_loci = kwargs["tier2_loci"]
else:
    tier2_loci = []

locus_list = tier1_loci + tier2_loci
num_loci = len(locus_list)
filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
output_path = f"/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/{drug}"

df_phenos = pd.read_csv(phenotype_file)
df_train_val = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)
df_test = df_phenos.query("category == 'test_set'").reset_index(drop=True)
del df_phenos

seq_data_path = output_path

if include_lineage:
    output_path += "_lineage"

if include_tier2:
    output_path += "_tier2"

if include_amino_acid_properties:
    output_path += "_amino_acid"

xgboost_dir = os.path.join(output_path, "xgboost")

if custom_loss:
    xgboost_dir = os.path.join(xgboost_dir, "custom_loss")

cv_dir = os.path.join(xgboost_dir, "cross_validation")
print(f"Saving results to {xgboost_dir}")
os.makedirs(cv_dir, exist_ok=True)

if AF_thresh != 0.75:
    test_seq_data_path = os.path.join(seq_data_path, f"AF_thresh_{int(AF_thresh*100)}")

    if not os.path.isdir(test_seq_data_path):
        os.makedirs(test_seq_data_path)
    
    test_predictions_fName = os.path.join(xgboost_dir, f"test_predictions_AF_thresh_{int(AF_thresh*100)}.csv")
    test_results_fName = os.path.join(xgboost_dir, f"results_AF_thresh_{int(AF_thresh*100)}.csv")
else:
    test_seq_data_path = seq_data_path
    test_predictions_fName = os.path.join(xgboost_dir, "test_predictions.csv")
    test_results_fName = os.path.join(xgboost_dir, "results.csv")

if loss_type == 'L1':
    loss_func = 'reg:absoluteerror'
    eval_metric = 'mae'
elif loss_type == 'L2':
    loss_func = 'reg:squared_error'
    eval_metric = 'mse'
else:
    raise ValueError(f"{loss_type} is not a valid loss type")


# get the input matrices using the helper function.
X_train, X_test = get_all_regression_inputs(df_train_val, df_test, seq_data_path, test_seq_data_path, locus_list, include_lineage=include_lineage, include_amino_acid_properties=include_amino_acid_properties)

lower_bounds_train, upper_bounds_train = df_train_val[[f"{drug}_lower_bound", f"{drug}_upper_bound"]].T.values
lower_bounds_test, upper_bounds_test = df_test[[f"{drug}_lower_bound", f"{drug}_upper_bound"]].T.values

y_train = np.log2(df_train_val[f"{drug}_midpoint"])
y_test = np.log2(df_test[f"{drug}_midpoint"])



def bounded_loss_xgb(lower_bounds, upper_bounds, loss_type):
    '''
    Helper function to pass in additional arguments
    
    This function is used to select the regularization strength of a Regression model. It is the Regression analog of boundedLoss_CNN
    
    y_test and y_pred are log2-transformed. lower_bounds and upper_bounds are NOT
    loss_type is L1 or L2, specifying whether to return the MAE or MSE
    '''

    # some lower bounds are 0, so log-transform them, then replace inf and -inf with 0. Will still work because predicted log-MIC (and also MIC) can not be below 0
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)

    lower_bounds[lower_bounds==np.inf] = 0
    lower_bounds[lower_bounds==-np.inf] = 0

    
    def bounded_loss_xgb_helper(y_pred, dtrain):

        '''
        Arguments must be y_pred and dtrain, in that order. dtrain is not needed here because we use the lower and upper bounds
        
        For compatibility with XGBoost, the loss function must return the gradient and the Hessian.
    
        Hessian matrix is the matrix of second order partial derivatives: 
    
            MAE ~ |y_true - y_hat| --> MAE' = -1 --> MAE'' = 0
            MSE ~ (y_true - y_hat)^2 --> MSE' = -2(y_true - y_hat) --> MSE'' = 2

            residual = R = y_true - y_hat

            gradient MAE = 
        '''

        # determine whether to compute error from the bounds or if the error is 0. 
        bound_to_compute_error = np.clip(y_pred, lower_bounds, upper_bounds)

        # residuals are the simple difference
        residuals = bound_to_compute_error - y_pred

        # print(y_pred[:10])
        # print(lower_bounds[:10])
        # print(upper_bounds[:10])
        # print(residuals[:10])

        # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
        outside_bounds_mask = (np.less_equal(y_pred, lower_bounds) | np.greater(y_pred, upper_bounds)).astype(int)

        # print(f"Outside bounds: {len(outside_bounds_mask)}, {np.mean(outside_bounds_mask)}")

        # multiply so that the points predicted in their bin are multiplied by 0 so they have 0 residual
        masked_residuals = outside_bounds_mask * residuals
        
        # Compute Hessian (constant for MAE)
        if loss_type == 'L1':
            
            # second derivative is 0. But the Hessian can't be 0 due to numerical instability issues, so make it 1
            hessian = 0 * np.ones_like(masked_residuals)

            # gradient is +1 if y_true > y_pred and -1 if y_true < y_pred, so this will take care of that
            gradient = np.sign(masked_residuals)
        
        elif loss_type == 'L2':

            # second derivative is 2
            hessian = 2 * np.ones_like(masked_residuals)

            # gradient is 2 (y_true - y_predd)
            gradient = 2 * masked_residuals

        # gradient is the sign of the residuals
        return gradient, hessian

    return bounded_loss_xgb_helper
    


def train_single_xgboost_model(X, y, lower_bounds, upper_bounds, params, X_val=None, y_val=None, lower_bounds_val=None, upper_bounds_val=None, train_idx=None, val_idx=None, save_model_fName=None, verbose=False):
    
    if X_val is None and (train_idx is None or val_idx is None):
        raise ValueError(f"Both X_val and train_id/val_idx must not be None!")

    if X_val is None:
        # need to split the passed in X_train using train_idx and val_idx
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        lower_bounds_train = lower_bounds[train_idx]
        upper_bounds_train = upper_bounds[train_idx]
        
        X_val = X[val_idx, :]
        y_val = y[val_idx]
        lower_bounds_val = lower_bounds[val_idx]
        upper_bounds_val = upper_bounds[val_idx]
    # if not, then the correct matrices have been passed in already
    else:
        X_train = X
        y_train = y
        lower_bounds_train = lower_bounds
        upper_bounds_train = upper_bounds
        
    # length checks to make sure the correct arguments have been passed in
    assert X_train.shape[0] == len(y_train) == len(lower_bounds_train) == len(upper_bounds_train)
    assert X_val.shape[0] == len(y_val) == len(lower_bounds_val) == len(upper_bounds_val)
    
    # initialize model
    model = xgb.XGBRegressor(**params,
                             n_estimators=N_epochs,
                             early_stopping_rounds=patience_epochs,
                             objective=loss_func,
    )
    
    model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=eval_metric,
                verbose=verbose
            )
    
    # save model. The sklearn API 
    if save_model_fName is not None:
        model.save_model(save_model_fName)
        
    results = model.evals_result()
    val_loss_array = results['validation_0']['mae']  # validation loss

    best_iter = model.best_iteration
    best_val_loss = model.best_score
        
    return best_iter, best_val_loss, val_loss_array

    
    
def train_single_xgboost_model_custom_loss(X, y, lower_bounds, upper_bounds, params, X_val=None, y_val=None, lower_bounds_val=None, upper_bounds_val=None, train_idx=None, val_idx=None, save_model_fName=None, verbose=False):
    
    if X_val is None and (train_idx is None or val_idx is None):
        raise ValueError(f"Both X_val and train_id/val_idx must not be None!")

    if X_val is None:
        # need to split the passed in X_train using train_idx and val_idx
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        lower_bounds_train = lower_bounds[train_idx]
        upper_bounds_train = upper_bounds[train_idx]
        
        X_val = X[val_idx, :]
        y_val = y[val_idx]
        lower_bounds_val = lower_bounds[val_idx]
        upper_bounds_val = upper_bounds[val_idx]
    # if not, then the correct matrices have been passed in already
    else:
        X_train = X
        y_train = y
        lower_bounds_train = lower_bounds
        upper_bounds_train = upper_bounds
        
    # length checks to make sure the correct arguments have been passed in
    assert X_train.shape[0] == len(y_train) == len(lower_bounds_train) == len(upper_bounds_train)
    assert X_val.shape[0] == len(y_val) == len(lower_bounds_val) == len(upper_bounds_val)
    
    # Initialize an empty model for each model
    model = None
    min_loss = 1e3
    patience_counter = 0

    train_loss_array = []
    val_loss_array = []
    
    # initialize model
    model = xgb.XGBRegressor(**params,
                             objective=bounded_loss_xgb(lower_bounds_train, upper_bounds_train, loss_type) # this returns an instance of the bounded_loss_xgb function
                            )

    # step_size = 10
    step_size = 1
    
    for n in np.arange(1, N_epochs+1, step_size):
    # for n in np.arange(10, N_epochs+1, step_size):

        model.fit(X_train, y_train)

        # y_hat = model.predict(dval)
        train_loss = boundedLoss_Reg(model.predict(X_train),
                                     y_train, 
                                     lower_bounds_train, 
                                     upper_bounds_train, 
                                     loss_type=loss_type
                                    )

        val_loss = boundedLoss_Reg(model.predict(X_val), 
                                   y_val,
                                   lower_bounds_val,
                                   upper_bounds_val,
                                   loss_type=loss_type
                                  )

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

        # if loss decreases by at least 1%
        if float((min_loss - val_loss) / min_loss) >= 0.01:

            if verbose:
                print(f"    {n} estimators: Validation loss improved from {min_loss} to {val_loss}")

            # zero out the patience counter
            patience_counter = 0

            # update the minimum loss
            min_loss = val_loss

            if save_model_fName is not None:
                model.save_model(save_model_fName)

        else:
            # print(f"{n} estimators: training loss = {train_loss}, validation loss = {val_loss}")

            # increment the patience counter
            patience_counter += step_size

            if patience_counter == patience_epochs:
                break

    # return best_iter, best_val_loss, and val_loss_array
    return n - patience_epochs, min_loss, val_loss_array
    
    

if not os.path.isfile(f"{xgboost_dir}/best_params.csv"):
    
    param_grid = {
        # 'learning_rate': [np.exp(-1.0 * 9)],
        # 'learning_rate': [0.1, 0.01, 0.001],
        'reg_lambda': np.logspace(-3, 3, 7),
        # 'max_depth': np.arange(3, 10+1),
        # 'reg_lambda': np.logspace(1, 3, 3),
        'max_depth': np.arange(10, 10+1),
    }

    if gpu_available:
        print("Using GPU for training.")
        param_grid['tree_method'] = ['gpu_hist']  # Use GPU for training
        param_grid['gpu_id'] = [0]  # Specify GPU id, use 0 because we only use 1 GPU node at a time
    else:
        print("Using CPU for training.")
        param_grid['tree_method'] = ['hist']  # Use CPU for training

    num_cv_splits = 5

    # use the same splits for all parameter sets by setting shuffle = False, then can use a paired test to compare them
    kfold_splits = StratifiedKFold(n_splits=num_cv_splits, shuffle=False)

    losses_df = []

    for params in ParameterGrid(param_grid):

        print(f"Evaluating params: {params}")

        # stratify by the binary resistance phenotype only. You can pass in a dummy variable for X, which is np.zeros(len(df_train_val)) here
        for split, (train_idx, val_idx) in enumerate(kfold_splits.split(np.zeros(len(df_train_val)), df_train_val["Binary"])): 

            if custom_loss:
                model_train_func = train_single_xgboost_model_custom_loss
            else:
                model_train_func = train_single_xgboost_model
                
            best_iter, best_val_loss, val_loss_array = model_train_func(X_train, 
                                                                        y_train,
                                                                        lower_bounds_train,
                                                                        upper_bounds_train,
                                                                        params,
                                                                        X_val=None,
                                                                        y_val=None,
                                                                        lower_bounds_val=None,
                                                                        upper_bounds_val=None,
                                                                        train_idx=train_idx,
                                                                        val_idx=val_idx,
                                                                        verbose=False
                                                                       )
            
            
            
            print(f"    Split {split+1}: Model stopped at iteration {best_iter} with validation loss {best_val_loss}")
            # print(f"    Params: {model.get_params()}")

            losses_df.append(pd.DataFrame({"CV": split,
                                           "max_depth": params['max_depth'], 
                                           'reg_lambda': params['reg_lambda'],
                                           'stop_iter': best_iter,
                                           "val_loss": best_val_loss
                                          }, 
                                          index=[0]))
            
        # save intermittently
        pd.concat(losses_df).to_csv(f"{xgboost_dir}/losses_df.csv", index=False)

    pd.concat(losses_df).to_csv(f"{xgboost_dir}/losses_df.csv", index=False)

    # read in the losses dataframe
    losses_df = pd.read_csv(f"{xgboost_dir}/losses_df.csv")

    best_params = pd.DataFrame(losses_df.groupby(['max_depth', 'reg_lambda'])['val_loss'].mean()).reset_index().sort_values("val_loss", ascending=False).iloc[[0], :]

    # save best params
    best_params.to_csv(f"{xgboost_dir}/best_params.csv", index=False)


# read in best hyperparameters
best_params = pd.read_csv(f"{xgboost_dir}/best_params.csv")
assert len(best_params) == 1
del best_params['val_loss']
assert best_params.shape[1] == 2

# convert to dictionary to pass into XGBRegressor
best_params = dict(best_params.iloc[0, ])
best_params['reg_lambda'] = float(best_params['reg_lambda'])
best_params['max_depth'] = int(best_params['max_depth'])
print(best_params)


if perform_cross_validation:

    num_cv_splits = 5
    cv_model_results = []

    kfold_splits = StratifiedKFold(n_splits=num_cv_splits, shuffle=True)
    
    # best_params = {'learning_rate': 0.1, 'reg_lambda': 1, 'early_stopping_rounds': patience_epochs, 'max_depth': 5}

    print(f"Performing cross-validation with best parameters: {best_params}\n")

    # stratify by the binary resistance phenotype only. You can pass in a dummy variable for X, which is np.zeros(len(df_train_val)) here
    for split, (train_idx, val_idx) in enumerate(kfold_splits.split(np.zeros(len(df_train_val)), df_train_val["Binary"])): 

        print(f"\nTraining {split+1}/{num_cv_splits} cross-validation splits")

        print(f"    CV train R: {df_train_val.iloc[train_idx]['Binary'].mean()}")
        print(f"    CV val R: {df_train_val.iloc[val_idx]['Binary'].mean()}")

        best_iter, best_val_loss, val_loss_array = train_single_xgboost_model(X_train, 
                                                                                y_train,
                                                                                lower_bounds_train,
                                                                                upper_bounds_train,
                                                                                best_params,
                                                                                X_val=None,
                                                                                y_val=None,
                                                                                lower_bounds_val=None,
                                                                                upper_bounds_val=None,
                                                                                train_idx=train_idx,
                                                                                val_idx=val_idx,
                                                                                save_model_fName=f"{cv_dir}/model_{split}.json",
                                                                                verbose=False
                                                                               )
        
        history_df = pd.DataFrame({'val_loss': val_loss_array})
        history_df.to_csv(f"{cv_dir}/history_{split}.csv", index=False)

        print(f"    Split {split+1}: Model stopped at iteration {best_iter} with validation loss {best_val_loss}")

        # get final predictions on the test set
        model = xgb.XGBRegressor()
        model.load_model(f"{cv_dir}/model_{split}.json")

        y_pred = model.predict(X_test)

        summary_df = create_summary_df(df_test,
                                       y_pred, 
                                       drug,
                                       binary_thresh, 
                                       num_loci, 
                                       model_name="XGBoost", 
                                       binarize=True, 
                                       save_fName=os.path.join(cv_dir, f"test_predictions_{split}.csv"),
                                      )
        summary_df["CV"] = split
        cv_model_results.append(summary_df)

        if loss_type == "L1":
            print(f"    Final Binned MAE: {summary_df['Binned_MAE'].values[0]}\n")
        else:
            print(f"    Final Binned MSE: {summary_df['Binned_MSE'].values[0]}\n")


    # save
    pd.concat(cv_model_results).to_csv(f"{xgboost_dir}/results.csv", index=False)


# ############################### FINAL MODEL ###############################

# if train_model:

#     best_params = {'learning_rate': 0.1, 'reg_lambda': 1, 'early_stopping_rounds': patience_epochs, 'max_depth': 5}
    
#     print(f"Training model with best parameters: {best_params}\n")
    
#     val_loss_array = train_single_xgboost_model(X_train,
#                                                                             y_train,
#                                                                             lower_bounds_train,
#                                                                             upper_bounds_train,
#                                                                             best_params,
#                                                                             X_val=X_test,
#                                                                             y_val=y_test,
#                                                                             lower_bounds_val=lower_bounds_test,
#                                                                             upper_bounds_val=upper_bounds_test,
#                                                                             train_idx=None,
#                                                                             val_idx=None,
#                                                                             save_model_fName=f"{xgboost_dir}/best_model.json",
#                                                                             verbose=True
#                                                                            )

#     # history_df = pd.DataFrame({'train_loss': train_loss_array, 'val_loss': val_loss_array})
#     history_df = pd.DataFrame({'val_loss': val_loss_array})

#     history_df.to_csv(f"{xgboost_dir}/history.csv", index=False)
    
    
# # get model predictions
# trained_model = xgb.XGBRegressor()

# trained_model.load_model(f"{xgboost_dir}/best_model.json")

# y_pred = trained_model.predict(X_test)

# # predictions will be saved for isolates with Span_CC = 1, but the summary df will not include them in the computation
# summary_df = create_summary_df(df_test,
#                                y_pred, 
#                                drug,
#                                binary_thresh, 
#                                num_loci, 
#                                model_name="XGBoost",
#                                binarize=True, 
#                                save_fName=os.path.join(xgboost_dir, "test_predictions.csv"),
#                               )    

# summary_df.to_csv(os.path.join(xgboost_dir, "full_model_results.csv"), index=False)
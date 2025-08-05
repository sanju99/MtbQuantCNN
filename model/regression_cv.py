import numpy as np
import pandas as pd
import glob, os, subprocess, vcf, shutil, sparse, yaml, sys, argparse, pickle, warnings, tracemalloc
import scipy.stats as st
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from model_utils import *

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")
results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"


# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

# boolean argument for including lineage SNPs, default value False. If you include the flag, it is considered True
parser.add_argument('--lineage', action='store_true', help='Flag to add lineage SNPs to model')

# boolean argument for including peptide lengths, default value False. If you include the flag, it is considered True
parser.add_argument('--peptide-lengths', dest='peptide_lengths', action='store_true', help='Flag to add peptide lengths to model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--tier2', action='store_true', help='Flag to add tier 2 loci to the model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--amino-acid', dest='amino_acid', action='store_true', help='Flag to add amino acid biophysical properties to the model')

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--augment', dest='augment', action='store_true', help='If True, use the {drug}_augment directory')

parser.add_argument('--binary', dest='binary', action='store_true', help='If True, use the {drug}_binary directory and train a binary model')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
AF_thresh = cmd_line_args.AF_thresh
augment = cmd_line_args.augment
binary = cmd_line_args.binary

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
filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
phenotype_file = kwargs["phenotype_file"]
binary_thresh = kwargs["binary_thresh"]

output_path = f"{results_dir}/{drug}"

if augment:
    output_path += "_augment"

    # same thing for the phenotypes file
    phenotype_file = os.path.join(os.path.dirname(phenotype_file) + "_augment", os.path.basename(phenotype_file))
    
if binary:
    output_path += "_binary"

    # same thing for the phenotypes file
    phenotype_file = os.path.join(os.path.dirname(phenotype_file) + "_binary", os.path.basename(phenotype_file))
    
loss_type = "L1"

num_loci = len(tier1_loci + tier2_loci)
df_phenos = pd.read_csv(phenotype_file)
df_train_val = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)
df_test = df_phenos.query("category == 'test_set'").reset_index(drop=True)

seq_data_path = output_path

if include_peptide_lengths:
    output_path += "_peptide"
    
if include_lineage:
    output_path += "_lineage"

if include_tier2:
    output_path += "_tier2"

if include_amino_acid_properties:
    output_path += "_amino_acid"

if AF_thresh != 0.75:
    # separate input and output paths for different AF
    test_seq_data_path = f"{seq_data_path}_AF{int(AF_thresh*100)}"
    output_path = f"{output_path}_AF{int(AF_thresh*100)}"

else:
    test_seq_data_path = seq_data_path

ridge_dir = os.path.join(output_path, "ridge")
del output_path

# but the output file names will be the same
results_fName = os.path.join(ridge_dir, "results.csv")

if not os.path.isdir(os.path.join(ridge_dir, "cross_validation")):
    os.makedirs(os.path.join(ridge_dir, "cross_validation"))

print(f"Saving results to {ridge_dir}")

# get the input matrices using the helper function.
X_train, X_test = get_all_regression_inputs(df_train_val, df_test, seq_data_path, test_seq_data_path, locus_list, include_lineage=include_lineage, include_amino_acid_properties=include_amino_acid_properties)

print(f"{X_train.shape[1]} features in the regression model")

lower_bounds_train, upper_bounds_train = df_train_val[[f"{drug}_lower_bound", f"{drug}_upper_bound"]].T.values
lower_bounds_test, upper_bounds_test = df_test[[f"{drug}_lower_bound", f"{drug}_upper_bound"]].T.values

if binary:
    y_train = df_train_val['Binary'].values
    y_test = df_test['Binary'].values
else:
    y_train = np.log2(df_train_val[f"{drug}_midpoint"])
    y_test = np.log2(df_test[f"{drug}_midpoint"])
    print(f"    Minimizing {loss_type} loss")


################################ 


reg_param_lst = np.logspace(-5, 5, 11)

num_cv_splits = 5

kfold_splits = StratifiedKFold(n_splits=num_cv_splits, shuffle=True)

results = []

# stratify by the binary resistance phenotype only. You can pass in a dummy variable for X, which is np.zeros(len(df_train_val)) here
for split, (train_idx, val_idx) in enumerate(kfold_splits.split(np.zeros(len(df_train_val)), df_train_val["Binary"])): 

    print(f"\nTraining {split+1}/{num_cv_splits} cross-validation splits")
    
    print(f"    CV train R: {df_train_val.iloc[train_idx]['Binary'].mean()}")
    print(f"    CV val R: {df_train_val.iloc[val_idx]['Binary'].mean()}")

    X_cv_train = X_train[train_idx, :]
    y_cv_train = y_train[train_idx]

    losses_df = pd.DataFrame(columns=["alpha", "val_loss"])

    for alpha in reg_param_lst:
        
        if binary:
            model = LogisticRegression(penalty='l2', C=1/alpha, class_weight='balanced')
        else:
            model = Ridge(alpha=alpha)
        
        model.fit(X_cv_train, y_cv_train)
    
        # predict() for logreg will already binarize then predictions at 0.5. So use predict_proba(), which returns the class probabilities for all classes. So get the last one, which is class 1
        if binary:
            y_hat = model.predict_proba(X_train[val_idx])[:, -1]
            losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "val_loss": sklearn.metrics.log_loss(y_train[val_idx], y_hat)}, index=[0])])
        else:
            y_hat = np.squeeze(model.predict(X_train[val_idx]))
            losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "val_loss": boundedLoss_Reg(y_hat, y_train[val_idx], lower_bounds_train[val_idx], upper_bounds_train[val_idx], loss_type=loss_type)}, index=[0])])

    # different regularization parameter for each split
    select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
    print(f"    Regularization parameter: {select_alpha}, minimum validation loss: {losses_df.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")

    del losses_df
    del alpha
    del model

    # fit a new model using the selected alpha (selected using the validation set), then get metrics on the test set
    if binary:
        model = LogisticRegression(penalty='l2', C=1/select_alpha, class_weight='balanced')
    else:
        model = Ridge(alpha=select_alpha)
        
    model.fit(X_cv_train, y_cv_train)

    # save the model in case it's needed for later (i.e. TRUST predictions or something)
    pickle.dump(model, open(os.path.join(ridge_dir, 'cross_validation', f"model_{split}.sav"), "wb"))

    # also save the predictions on the test set
    if binary:
        y_pred = model.predict_proba(X_test)[:, -1]
        
        df_binary_pred = df_test.copy()
        df_binary_pred['y_pred'] = y_pred

        # determine the threshold that maximizes sensitivity and specificity. This function adds a column y_pred_label, the binarized predictions, to the dataframe
        df_binary_pred = get_threshold_val(df_binary_pred, 'y_pred', 'Binary', spec_thresh=None).rename(columns={'Binary': 'y_test'})

        # save only relevant columns
        df_binary_pred[['ROLLINGDB_ID', 'y_test', 'y_pred', 'y_pred_label']].to_csv(os.path.join(ridge_dir, "cross_validation", f"test_predictions_{split}.csv"), index=False)

        # add the binary metrics, like sens, spec accuracy, F1, etc. using the compute_binary_metrics function
        summary_df = compute_binary_metrics(df_binary_pred['y_test'], df_binary_pred['y_pred_label'], binary_thresh, binarize=False)

        # add AUC
        summary_df['AUC'] = sklearn.metrics.roc_auc_score(df_binary_pred['y_test'], df_binary_pred['y_pred'])
    else:
        y_pred = np.squeeze(model.predict(X_test))
        
        summary_df = create_summary_df(df_test, 
                                       y_pred, 
                                       drug, 
                                       binary_thresh, 
                                       num_loci, 
                                       model_name="Reg",
                                       binarize=True, 
                                       save_fName=os.path.join(ridge_dir, 'cross_validation', f"test_predictions_{split}.csv")
                                      )
    summary_df["CV"] = split
    results.append(summary_df)
    del y_pred

pd.concat(results).to_csv(results_fName, index=False)

print("Fitting full model")

losses_df = pd.DataFrame(columns=["alpha", "val_loss"])

for alpha in reg_param_lst:
    
    if binary:
        model = LogisticRegression(penalty='l2', C=1/alpha, class_weight='balanced')
    else:
        model = Ridge(alpha=alpha)
    
    model.fit(X_train, y_train)

    # get predictions on the test set, then compute binned error or binary cross entropy
    if binary:
        y_hat = model.predict_proba(X_test)[:, -1]
        losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "val_loss": sklearn.metrics.log_loss(y_test, y_hat)}, index=[0])])
    else:
        y_hat = np.squeeze(model.predict(X_test))
        losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "val_loss": boundedLoss_Reg(y_hat, 
                                                                                                y_test, 
                                                                                                lower_bounds_test, 
                                                                                                upper_bounds_test, 
                                                                                                loss_type=loss_type
                                                                                               )
                                                   }, index=[0])])


# different regularization parameter for each split
select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
print(f"    Regularization parameter: {select_alpha}, minimum test loss: {losses_df.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")

del losses_df
del alpha
del model

# fit a new model using the selected alpha (selected using the validation set), then get metrics on the test set
if binary:
    model = LogisticRegression(penalty='l2', C=1/select_alpha, class_weight='balanced')
else:
    model = Ridge(alpha=select_alpha)
    
model.fit(X_train, y_train)

# save the model in case it's needed for later (i.e. TRUST predictions or something)
pickle.dump(model, open(os.path.join(ridge_dir, "best_model.sav"), "wb"))

# also save the predictions on the test set
if binary:
    y_pred = model.predict_proba(X_test)[:, -1]
    
    df_binary_pred = df_test.copy()
    df_binary_pred['y_pred'] = y_pred

    # determine the threshold that maximizes sensitivity and specificity. This function adds a column y_pred_label, the binarized predictions, to the dataframe
    df_binary_pred = get_threshold_val(df_binary_pred, 'y_pred', 'Binary', spec_thresh=None).rename(columns={'Binary': 'y_test'})

    # save only relevant columns
    df_binary_pred[['ROLLINGDB_ID', 'y_test', 'y_pred', 'y_pred_label']].to_csv(os.path.join(ridge_dir, "test_predictions.csv"), index=False)

    # add the binary metrics, like sens, spec accuracy, F1, etc. using the compute_binary_metrics function
    summary_df = compute_binary_metrics(df_binary_pred['y_test'], df_binary_pred['y_pred_label'], binary_thresh, binarize=False)

    # add AUC
    summary_df['AUC'] = sklearn.metrics.roc_auc_score(df_binary_pred['y_test'], df_binary_pred['y_pred'])

else:
    y_pred = np.squeeze(model.predict(X_test))
    
    summary_df = create_summary_df(df_test, 
                               y_pred, 
                               drug, 
                               binary_thresh, 
                               num_loci, 
                               model_name="Reg",
                               binarize=True, 
                               save_fName=os.path.join(ridge_dir, "test_predictions.csv")
                              )

summary_df.to_csv(os.path.join(ridge_dir, "full_model_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
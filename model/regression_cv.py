import numpy as np
import pandas as pd
import glob, os, subprocess, vcf, shutil, sparse, yaml, sys, argparse, pickle, warnings, tracemalloc
import scipy.stats as st
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from model_utils import *
from analysis_utils import *

from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")


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

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
AF_thresh = cmd_line_args.AF_thresh

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
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]

if 'output_path' in kwargs.keys():
    output_path = kwargs["output_path"]
else:
    output_path = f"/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/{drug}"
    
loss_type = "L1"
binary = False

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

    genotype_input_directory = f"{genotype_input_directory.replace('fastas', 'AF_thresh_25/fastas')}"

    output_path = f"{output_path}_AF{int(AF_thresh*100)}"

else:
    test_seq_data_path = seq_data_path

ridge_dir = os.path.join(output_path, "ridge")

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

y_train = np.log2(df_train_val[f"{drug}_midpoint"])
y_test = np.log2(df_test[f"{drug}_midpoint"])


################################ 


reg_param_lst = np.logspace(-5, 5, 11)
    
print(f"    Minimizing {loss_type} loss")

num_cv_splits = 5

# don't need to shuffle because samples within a batch are shuffled by the dataloader
kfold_splits = StratifiedKFold(n_splits=num_cv_splits, shuffle=False)

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
        
        model = Ridge(alpha=alpha)
        model.fit(X_cv_train, y_cv_train)
    
        # get predictions on the CV validation set, then compute binned error. Use the same functional form as for the CNN
        y_hat = np.squeeze(model.predict(X_train[val_idx]))

        losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "val_loss": boundedLoss_Reg(y_hat, y_train[val_idx], lower_bounds_train[val_idx], upper_bounds_train[val_idx], loss_type=loss_type)}, index=[0])])

    # different regularization parameter for each split
    select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
    print(f"    Regularization parameter: {select_alpha}, minimum validation loss: {losses_df.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")

    del losses_df
    del alpha
    del model

    # fit a new model using the selected alpha (selected using the validation set), then get metrics on the test set
    model = Ridge(alpha=select_alpha)
    model.fit(X_cv_train, y_cv_train)

    # save the model in case it's needed for later (i.e. TRUST predictions or something)
    pickle.dump(model, open(os.path.join(ridge_dir, 'cross_validation', f"model_{split}.sav"), "wb"))

    # get predictions on the test set, then compute binned error. Use the same functional form as for the CNN
    y_pred = np.squeeze(model.predict(X_test))

    # also save the predictions
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

pd.concat(results).to_csv(results_fName, index=False)

print("Fitting full model")

losses_df = pd.DataFrame(columns=["alpha", "val_loss"])

for alpha in reg_param_lst:
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # get predictions on the CV validation set, then compute binned error. Use the same functional form as for the CNN
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
model = Ridge(alpha=select_alpha)
model.fit(X_train, y_train)

# save the model in case it's needed for later (i.e. TRUST predictions or something)
pickle.dump(model, open(os.path.join(ridge_dir, "best_model.sav"), "wb"))

# get predictions on the test set, including the isolates that span the CC
y_pred = np.squeeze(model.predict(X_test))

# save the predictions, but not the metrics
summary_df = create_summary_df(df_test, 
                               y_pred, 
                               drug, 
                               binary_thresh, 
                               num_loci, 
                               model_name="Reg",
                               binarize=True, 
                               save_fName=os.path.join(ridge_dir, "test_predictions.csv")
                              )

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
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

parser.add_argument('--train', dest='train_model', action='store_true', help='If true, train a new model. If false, just create input matrices.')

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--bootstrap', default=5, dest='num_bootstrap', type=int, help='Number of bootstrap replicates')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
train_model = cmd_line_args.train_model
AF_thresh = cmd_line_args.AF_thresh
num_bootstrap = cmd_line_args.num_bootstrap

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

filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
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

ridge_dir = os.path.join(output_path, "ridge")

if AF_thresh != 0.75:
    test_seq_data_path = os.path.join(seq_data_path, f"AF_thresh_{int(AF_thresh*100)}")

    if not os.path.isdir(test_seq_data_path):
        os.makedirs(test_seq_data_path)
    
    test_predictions_fName = os.path.join(ridge_dir, f"test_predictions_AF_thresh_{int(AF_thresh*100)}.csv")
    test_results_fName = os.path.join(ridge_dir, f"results_AF_thresh_{int(AF_thresh*100)}.csv")
else:
    test_seq_data_path = seq_data_path
    test_predictions_fName = os.path.join(ridge_dir, "test_predictions.csv")
    test_results_fName = os.path.join(ridge_dir, "results.csv")

if not os.path.isdir(ridge_dir):
    os.makedirs(ridge_dir)

print(f"Saving results to {ridge_dir}")

# read in matrices of input sequences. These are not in the ridge directory, they are the same as the matrices used by the CNN
X_train_val = sparse.load_npz(f"{seq_data_path}/pkl_sparse_train_val.npz").todense()
X_AA_train_val = np.load(f"{seq_data_path}/pkl_AA_train_val.npy")

# different path for the test data because of the possibility of using different AF threshold
X_AA_test = np.load(f"{test_seq_data_path}/pkl_AA_test.npy")

train_idx = df_train_val.query("category=='train_set'").index.values
val_idx = df_train_val.query("category=='validation_set'").index.values

# nucleotide inputs
X_train = get_single_matrix_regression_input(X_train_val, keep_idx=train_idx, num_keep_channels=num_loci)
X_val = get_single_matrix_regression_input(X_train_val, keep_idx=val_idx, num_keep_channels=num_loci)

# different path for the test data because of the possibility of using different AF threshold
X_test = get_single_matrix_regression_input(sparse.load_npz(f"{test_seq_data_path}/pkl_sparse_test.npz").todense(), keep_idx=None, num_keep_channels=num_loci)

# get the genes to keep (this is needed for both amino acid and gene peptide lengths inputs)
# keep only the specified loci for this model
genes_list = get_genes_lst(tier1_loci + tier2_loci)

# check that feature shapes are the same across the matrices
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

# drop sites with no signal across the train set (biallelic sites)
unique_values = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X_train)

# features with variation, only keep these for computational efficiency. These will be indexes
# it's most important to do this for the nucleotide features because they are the largest
keep_features = np.where(unique_values > 1)[0]
print(f"Keeping {len(keep_features)} nucleotide features in the model")

X_train = X_train[:, keep_features]
X_val = X_val[:, keep_features]
X_test = X_test[:, keep_features]

if include_lineage:

    lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
    assert len(np.unique(lineages)) == 2

    X_train = np.concatenate([X_train, lineages.loc[df_train_val['ROLLINGDB_ID']].values[train_idx, :]], axis=1)
    X_val = np.concatenate([X_val, lineages.loc[df_train_val['ROLLINGDB_ID']].values[val_idx, :]], axis=1)
    X_test = np.concatenate([X_test, lineages.loc[df_test['ROLLINGDB_ID']]], axis=1)

# if include_peptide_lengths:

#     gene_peptide_lengths = pd.read_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"), index_col=[0])

#     # keep only those indicated by the locus list
#     gene_peptide_lengths = gene_peptide_lengths[[f"{gene}_length" for gene in genes_list]]
#     print(gene_peptide_lengths.columns)

#     # combine with the nucleotide matrices
#     X_train = np.concatenate([X_train, gene_peptide_lengths.loc[df_train_val['ROLLINGDB_ID']].values[train_idx, :]], axis=1)
#     X_val = np.concatenate([X_val, gene_peptide_lengths.loc[df_train_val['ROLLINGDB_ID']].values[val_idx, :]], axis=1)
#     X_test = np.concatenate([X_test, gene_peptide_lengths.loc[df_test['ROLLINGDB_ID']]], axis=1)

if include_amino_acid_properties:

    # compute the mean and SD of the training set to scale validation and test data later. Only amino acid features need to be scaled
    # scale across the sample axis (0) and the length of the amino acid sequence (2). Don't scale different biophysical properties together (1), or different genes together (3)
    train_mean_fName = os.path.join(seq_data_path, "AA_train_mean.npy")
    train_std_fName = os.path.join(seq_data_path, "AA_train_std.npy")
    
    if not os.path.isfile(train_mean_fName) or not os.path.isfile(train_std_fName):

        print(f"Computing training dataset mean and standard deviation for the amino acid features and saving to {seq_data_path}")

        X_AA_train = X_AA_train_val[train_idx, :]
        train_mean = X_AA_train.mean(axis=(0, 2))
        train_std = X_AA_train.std(axis=(0, 2))

        np.save(train_mean_fName, train_mean)
        np.save(train_std_fName, train_std)

    else:
        train_mean = np.load(train_mean_fName)
        train_std = np.load(train_std_fName)

    # train_mean and train_std are only 2 dimensions. So need to duplicate the arrays to make the full dataset and protein sequence lengths
    # scale all 3 matrices
    X_AA_train_val = (X_AA_train_val - expand_dims_for_rescaling(train_mean, (0, 2), X_AA_train_val)) / expand_dims_for_rescaling(train_std, (0, 2), X_AA_train_val)
    X_AA_test = (X_AA_test - expand_dims_for_rescaling(train_mean, (0, 2), X_AA_test)) / expand_dims_for_rescaling(train_std, (0, 2), X_AA_test)
    
    train_AA_matrix = get_single_matrix_regression_input(X_AA_train_val, keep_idx=train_idx, num_keep_channels=len(genes_list))
    val_AA_matrix = get_single_matrix_regression_input(X_AA_train_val, keep_idx=val_idx, num_keep_channels=len(genes_list))
    test_AA_matrix = get_single_matrix_regression_input(X_AA_test, keep_idx=None, num_keep_channels=len(genes_list))
    
    X_train = np.concatenate([X_train, train_AA_matrix], axis=1)
    X_val = np.concatenate([X_val, val_AA_matrix], axis=1)
    X_test = np.concatenate([X_test, test_AA_matrix], axis=1)


# check that feature shapes are the same across the matrices
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

# select regularization parameter by training individual models on the training set and selecting the model with the smallest loss on the validation set
# (don't need to do cross-validation because we've set aside separate validation and testing datasets)
if train_model:

    print(f"    Minimizing {loss_type} loss")
    reg_param_lst = np.logspace(-5, 5, 11)
    losses_df = pd.DataFrame(columns=["alpha", "val_loss"])
    
    y_train = np.log2(df_train_val.loc[train_idx, f"{drug}_midpoint"].values)
    
    # need the validation data bounds for computing binned error
    lower_bounds_val, y_val, upper_bounds_val = df_train_val.loc[val_idx, [f"{drug}_lower_bound", f"{drug}_midpoint", f"{drug}_upper_bound"]].T.values
    y_val = np.log2(y_val)
    
    for alpha in reg_param_lst:    
            
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        # get predictions on the test set, then compute binned error. Use the same functional form as for the CNN
        y_hat = np.squeeze(model.predict(X_val))
    
        losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "val_loss": boundedLoss_Reg(y_hat, y_val, lower_bounds_val, upper_bounds_val, loss_type=loss_type)}, index=[0])])
        losses_df.to_csv(os.path.join(ridge_dir, "reg_param_losses.csv"), index=False)
    
    select_alpha = losses_df.sort_values("val_loss", ascending=True)["alpha"].values[0]
    print(f"    Regularization parameter: {select_alpha}, minimum validation loss: {losses_df.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")
    del losses_df
    
    # fit new model on the selected regularization parameter
    model = Ridge(alpha=select_alpha)
    model.fit(X_train, y_train)
    
    # save the model in case it's needed for later (i.e. TRUST predictions or something)
    pickle.dump(model, open(os.path.join(ridge_dir, "model.sav"), "wb"))


# just get predictions if the model has been fit
if os.path.isfile(os.path.join(ridge_dir, "model.sav")):

    model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
    
    y_pred = np.squeeze(model.predict(X_test))
    
    results = create_summary_df(df_test, y_pred, drug, binary_thresh, num_loci, model_name="Reg", binarize=True, save_fName=test_predictions_fName)
    
    results["CV"] = 0
    results.to_csv(test_results_fName, index=False)
    
    del model
    del y_pred
    
    if loss_type == "L1":
        print(f"    Final Binned MAE: {results['Binned_MAE'].values[0]}")
    else:
        print(f"    Final Binned MSE: {results['Binned_MSE'].values[0]}")


# train bootstrap replicates and save model metrics if train argument is specified
if train_model:
    
    for rep in range(num_bootstrap):
    
        print(f"Training bootstrap replicate {rep+1}/{num_bootstrap}")
    
        bs_train_idx = np.random.choice(np.arange(0, len(X_train)), size=len(X_train), replace=True)
    
        # already scaled the full X_train above, so don't do it again
        X_bs_train = X_train[bs_train_idx, :]
        y_bs_train = y_train[bs_train_idx]

        # use the same regularization parameter determined on the full model
        model = Ridge(alpha=select_alpha)
        model.fit(X_bs_train, y_bs_train)
        y_pred = np.squeeze(model.predict(X_test))

        # don't save the individual predictions for each bootstrapped model
        bs_summary_df = create_summary_df(df_test, y_pred, drug, binary_thresh, num_loci, model_name="Reg", binarize=True, save_fName=None)
        bs_summary_df["CV"] = rep + 1
        results = pd.concat([results, bs_summary_df])
    
    results.to_csv(test_results_fName, index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
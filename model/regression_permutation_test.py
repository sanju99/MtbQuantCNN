import numpy as np
import pandas as pd

import sklearn, pickle, os, glob, sys, yaml, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge

sys.path.append("utils")
from analysis_utils import *
from model_utils import *

_, config_file = sys.argv

kwargs = yaml.safe_load((open(config_file)))
drug = kwargs["drug"]
include_lineage = kwargs["include_lineage"]
binary_thresh = kwargs["binary_thresh"]
locus_list = kwargs["locus_list"]
num_loci = len(locus_list)
loss_type = kwargs["loss_type"]
fasta_dir = kwargs["genotype_input_directory"]

out_dir = kwargs["output_path"]
ridge_dir = os.path.join(out_dir, "ridge")
reg_model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
feat_list = pd.read_csv(os.path.join(ridge_dir.replace("_lineage", ""), "model_feature_names.txt"), sep="\t", header=None)[0].values

reg_model = pickle.load(open(os.path.join(ridge_dir, "model.sav"), "rb"))
reg_param = reg_model.alpha
print(f"Regularization param from full model: {reg_param}")

if include_lineage:
    lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
    lineages.columns = [f"lineageSNP_{col}" for col in lineages.columns]
    assert len(np.unique(lineages.values)) == 2
    feat_list = np.concatenate([feat_list, lineages.columns])

coef_df = pd.DataFrame({"feature": feat_list, "coef": np.squeeze(reg_model.coef_)})
coef_df.to_csv(os.path.join(ridge_dir, "coef_df.csv"), index=False)

########## PERFORM PERMUTATION TEST

X_train = pd.read_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "train_seq_matrix.pkl"))

if include_lineage:
    X_train = X_train.merge(lineages, left_index=True, right_index=True).values
else:
    X_train = X_train.values
    
X_train = (X_train - X_train.mean()) / X_train.std()

df_train = pd.read_csv(kwargs['phenotype_file']).query("category=='original_train_set'")
lower_bounds_train, upper_bounds_train = df_train[f"{drug}_lower_bound"].values, df_train[f"{drug}_upper_bound"].values
y_train = np.log2(df_train[f"{drug}_midpoint"].values)

num_reps = 10
coefs_lst = []

for i in range(num_reps):
    
    # reshuffle the outcome values
    np.random.shuffle(y_train)
    
    permute_model = Ridge(alpha=reg_param)
    permute_model.fit(X_train, y_train)

    coefs_lst.append(np.squeeze(permute_model.coef_))

    if i % 500 == 0:
        print(i)

permute_df = pd.DataFrame(coefs_lst)
permute_df.columns = feat_list
assert permute_df.shape[1] == len(coef_df)
permute_df.to_csv(os.path.join(ridge_dir, "permute_df.csv"), index=False)
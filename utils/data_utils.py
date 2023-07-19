import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
import sklearn.metrics
import sklearn.utils
import scipy.stats as st


def get_one_hot(sequence):
    """
	Returns
	-------
	np.ndarray of int
		L (seq len) x 5 one-hot encoded sequence
	"""

    seq_len = len(sequence)
    
    # ignore N characters, the vector is all 0s for that
    seq_in_index = [(idx, BASE_TO_COLUMN.get(base, base)) for idx, base in enumerate(sequence) if base in BASE_TO_COLUMN.keys()]
    idx, pos = list(zip(*seq_in_index))

    # initialize
    one_hot = np.zeros((seq_len, 5))

    # Assign the found positions to 1
    one_hot[idx, pos] = 1

    return one_hot


def sequence_dictionary(filename):
    """
	Creates a dataframe that contains the sequence of each locus for each isolate
	Note that this function splits the identifier names in the fasta file on '/'
	and takes the last entry

	Parameters
	----------
	filename: str
		path to directory containing genotype data (one fasta file containing
		sequences for all isolates at a particular locus)

	Returns
	-------
	pd.DataFrame with one column, indexed by strain name
		column name will be the beginning string of the file name
	"""
    seq_dict = SeqIO.to_dict(
        SeqIO.parse(filename, "fasta"),
        key_function=lambda x: x.id.replace(".vcf", "").replace("_freebayes", "").split("/")[-1].split(".cut")[0])

    # create a dictionary of identifier: sequence
    for identifier, sequence in seq_dict.items():
        seq_dict[identifier] = str(sequence.seq)

    df = pd.DataFrame.from_dict(seq_dict, orient='index')
    gene_name = os.path.basename(filename).split(".")[0]
    df.columns = [gene_name]

    return df


def make_genotype_df(locus_list, genotype_input_directory):
    """
	Parameters
	----------
	genotype_input_directory: str
		path to directory containing fasta files of genotype inputs

	Returns
	-------
	pd.DataFrame:
		indexed by isolate name, one column per locus
	"""
    # Make a df that combines all genotype data
    dfs_list = []
    
    fastas = [os.path.join(genotype_input_directory, locus + ".fasta") for locus in locus_list]
    print(f"{len(fastas)} loci!")

    for df_file in fastas:
        print("reading fasta file", df_file)
        _df = sequence_dictionary(df_file)
        
        # if the column is intergenic, it will throw an error if there are multiple intergenics
        # if that's the case, rename the column using the file name
        if 'intergenic' in _df.columns:
            _df.columns = [os.path.basename(df_file).split("_")[1]]
        
        dfs_list.append(_df)
    
    if len(dfs_list) > 1:
        df_genos = dfs_list[0].join(dfs_list[1:], how='outer')
        return df_genos
    else:
        return dfs_list[0]
    
    
    
def create_X(df_geno_pheno):
    """
	Create an input X matrix, with output dimensions:
		n_strains x 5 (one-hot) x longest locus length x no. of loci
	"""

    def _get_shapes(df_geno_pheno):
        """
		Finds the length of each gene in the input dataframe
		Parameters
		----------
		df_geno_pheno: pd.Dataframe

		Returns
		-------
		dict of str: int
			length of coordinates in each column
		"""
        shapes = {}
        for column in df_geno_pheno.columns:
            if "one_hot" in column:
                shapes[column] = df_geno_pheno.loc[df_geno_pheno.index[0], column].shape[0]

        return shapes

    shapes = _get_shapes(df_geno_pheno)

    # Length of longest gene locus
    n_genes = len(shapes)
    L_longest = max(list(shapes.values()))
    print(f"found {n_genes} genes and longest gene {L_longest}")

    # Number of strains in model
    n_strains = df_geno_pheno.shape[0]

    # define shape of matrix - fill with zeros (effectively accomplishes padding)
    X = np.zeros((n_strains, 5, L_longest, n_genes))

    # for each strain and gene locus
    for idx, strain in enumerate(df_geno_pheno.index):
        for gene_index, gene in enumerate(shapes.keys()):
            one_hot_gene = df_geno_pheno.loc[strain, gene]
            X[idx, :, range(0, one_hot_gene.shape[0]), gene_index] = one_hot_gene

    return X



def make_geno_pheno_files(**kwargs):
    
    output_path = kwargs["output_path"]
    drug = kwargs["drug"]

    # get table for phenotypes and add the reference strain to match the format of the FASTA files
    df_phenos = pd.read_csv(kwargs['phenotype_file'])
    df_phenos.loc[-1, ['ROLLINGDB_ID', 'category']] = np.array(["MT_H37Rv", "reference"], dtype=object)

    # Swap na's for -1's as before 
    df_phenos.fillna(-1, inplace = True)
    
    # make table of all genotypes. Index is the identifier
    df_genos = make_genotype_df(kwargs["locus_list"], kwargs['genotype_input_directory'])
    
    # Some IDs had '-' switched to '_', fix that here
    df_genos.index = [name.replace("-", "_").split(".")[0] for name in df_genos.index]
    df_phenos["ROLLINGDB_ID"] = [name.replace("-", "_") for name in df_phenos["ROLLINGDB_ID"]]
    
    print(f"found phenotypes for {len(df_phenos)-1} strains")

    df_genos.index.rename('ROLLINGDB_ID', inplace=True)

    # this makes life easier later. Keep H37Rv in this      
    df_genos = df_genos.merge(df_phenos[["category", "ROLLINGDB_ID"]], on="ROLLINGDB_ID", how="inner")
    df_genos.to_csv(os.path.join(output_path, "df_genos.csv"), index=False)

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in kwargs["locus_list"]:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[locus + "_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))
        
    if "category" in df_genos and "category" in df_phenos:
        del df_genos["category"]

    # combined dataframe of all genotypes and phenotypes
    df_geno_pheno = df_phenos.merge(df_genos, how='inner', on="ROLLINGDB_ID")
    
    # this is to check that df_geno_pheno is in the same order as df_phenos.
    assert sum(df_phenos.ROLLINGDB_ID.values != df_geno_pheno.ROLLINGDB_ID.values) == 0

    # Create a one-hot-encoding matrix with dimensions
    # num_isolate x 5 x locus_length x num_loci
    X_all = create_X(df_geno_pheno)
    X_sparse = sparse.COO(X_all)

    # Works because index was reset to default
    train_indices = df_geno_pheno.query("category=='original_train_set'").index
    test_indices = df_geno_pheno.query("category=='original_test_set'").index
    ref_idx = df_geno_pheno.query("category=='reference'").index

    print("splitting X pkl")
    X_sparse_train = X_sparse[train_indices, :]
    X_sparse_test = X_sparse[test_indices, :]
    X_H37Rv = X_sparse[ref_idx, :]

    print("training set shape", X_sparse_train.shape)
    print("test set shape", X_sparse_test.shape)
    print("reference shape", X_H37Rv.shape)

    # Save
    sparse.save_npz(os.path.join(output_path, "pkl_sparse_train.npz"), X_sparse_train, compressed=False)
    sparse.save_npz(os.path.join(output_path, "pkl_sparse_test.npz"), X_sparse_test, compressed=False)
    sparse.save_npz(os.path.join(output_path, "pkl_sparse_ref.npz"), X_H37Rv, compressed=False)
    
    
    
    
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
    return pred_df




def compute_binary_metrics(y_val, y_pred, binary_thresh, binarize=False):
        
    # binarize using the critical concentration
    if binarize:
        y_val_binary = (y_val >= np.log2(binary_thresh)).astype(int)
        y_pred_binary = (y_pred >= np.log2(binary_thresh)).astype(int)
    else:
        y_val_binary = np.copy(y_val)
        y_pred_binary = np.copy(y_pred)
        
    assert len(np.unique(y_val_binary)) <= 2
    assert len(np.unique(y_pred_binary)) <= 2
    
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_val_binary, y_pred_binary).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)
    precision = tp / (tp+fp)
    acc = sklearn.metrics.accuracy_score(y_val_binary, y_pred_binary)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y_val_binary, y_pred_binary)

    return pd.DataFrame({"Sensitivity": sens,
                         "Specificity": spec,
                         "Precision": precision,
                         "Accuracy": acc,
                         "Balanced_Acc": balanced_acc
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
    
    # log_binary_thresh = np.log2(binary_thresh)
    # df.loc[((df[y_pred_col] < log_binary_thresh) & (df[y_true_col] < log_binary_thresh)) | ((df[y_pred_col] > log_binary_thresh) & (df[y_true_col] > log_binary_thresh)), "binary"] = 1
    # df["binary"] = df["binary"].fillna(0).astype(int)
    
    return df




def boundedLoss_predict(pred_df, binary_thresh, y_pred_col="y_pred", y_true_col="y_test", lower_bounds_col="lower", upper_bounds_col="upper"):
    '''
    y_true and y_pred are log-MICs. lower_bounds and upper_bounds are exponentiated. 
    
    This function returns bounded MAE, MSE, and the proportion of points measured within 1 MIC doubling (1 log2 unit)
    ''' 

    del_cols = [f"{y_pred_col}_exp", "within_doubling", "within_1bin", "compute_error", f"{lower_bounds_col}_rounded", f"{upper_bounds_col}_rounded"]

    for col in del_cols:
        if col in pred_df.columns:
            del pred_df[col]

    # first add essential agreement (proportion within 1 doubling dilution)
    pred_df[f"{y_pred_col}_exp"] = np.round(np.exp2(pred_df[y_pred_col]), 2)
    
    pred_df.loc[(pred_df[lower_bounds_col] / 2 <= pred_df[f"{y_pred_col}_exp"]) & 
                (pred_df[upper_bounds_col] * 2 >= pred_df[f"{y_pred_col}_exp"])
                , "within_doubling"] = 1

    # manual fix....
    pred_df.loc[(pred_df[f"{y_pred_col}_exp"] == 0.06) & (pred_df[lower_bounds_col] == 0.12), "within_doubling"] = 1
    pred_df["within_doubling"] = pred_df["within_doubling"].fillna(0).astype(int)
        
    # make copies to avoid changing the original dataframe
    lower_bounds = np.copy(pred_df[lower_bounds_col].values)
    upper_bounds = np.copy(pred_df[upper_bounds_col].values)
    
    lower_bounds[lower_bounds==0] += 1e-6
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)
    
    pred_df["compute_error"] = ((pred_df[y_pred_col] < lower_bounds) | (pred_df[y_pred_col] > upper_bounds)).astype(int)

    # compute error relative to the bounds, NOT RELATIVE TO THE MIDPOINT (y_test) of each isolate
    # np.clip returns one of the values from lower_bounds or upper_bounds, whichever is closest to the prediction, if the value is outside the bounds
    # if the value is within the bounds, the value is returned. In this case, it doesn't matter because compute_error = 0 and y_pred - bound_to_compute_error = 0
    bound_to_compute_error = np.clip(pred_df[y_pred_col].values, lower_bounds, upper_bounds)

    mae = np.mean(pred_df["compute_error"] * (np.abs(bound_to_compute_error - pred_df[y_pred_col])))
    mse = np.mean(pred_df["compute_error"] * (np.square(bound_to_compute_error - pred_df[y_pred_col])))
    
    pred_df = compute_proportion_within_1bin(pred_df, y_pred_col, y_true_col, lower_bounds_col, upper_bounds_col, binary_thresh)

    # return error and proportion within 1 doubline of the measured MIC
    return mae, mse, pred_df["within_1bin"].mean(), pred_df["within_doubling"].mean() 
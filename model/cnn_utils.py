import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

# use only 4, treat gap character as all 0s
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}

# Get one hot vector
def get_one_hot(sequence):
    """
	Returns
	-------
	np.ndarray of int
		L (seq len) x 5 one-hot encoded sequence
	"""

    seq_len = len(sequence)
    seq_in_index = [BASE_TO_COLUMN.get(b, b) for b in sequence]
    one_hot = np.zeros((seq_len, 5))

    # Assign the found positions to 1
    one_hot[np.arange(seq_len), np.array(seq_in_index)] = 1

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
    df_genos.index = [name.replace("-", "_") for name in df_genos.index]
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
        y_val_binary = (y_val > np.log(binary_thresh)).astype(int)
        y_pred_binary = (y_pred > np.log(binary_thresh)).astype(int)
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
    
    
class MtbGeneDataset(Sequence):

    def __init__(self, sparse_file, phenotype_file, lineages_mat, drug, locus_list, train_or_test, binary, cc, include_lineage=False, bounded_loss=False, data_idx=None, batch_size=128, shuffle=True):
        '''
        Sparse files for both the training and testing sets are available, so read those. 
        Use this for cross-validation, where data_idx is train_idx or test_idx
        '''
        
        # read in the one-hot encoded files and convert from sparse to dense format. read in the phenotypes file
        X = sparse.load_npz(sparse_file)
        df_phenos = pd.read_csv(phenotype_file)
        df_phenos = df_phenos.query("category==@train_or_test").reset_index(drop=True)
        lineages = lineages_mat.loc[df_phenos["ROLLINGDB_ID"]]
        
        # include lineage if this argument is True. If not, return NaNs that will get dropped later if the model should not include lineage
        if include_lineage:
            # check ordering
            assert sum(lineages.index.values != df_phenos["ROLLINGDB_ID"].values) == 0
            lineages = lineages.values
        else:
            lineages = np.ones(lineages.shape)*np.nan
            
        assert X.shape[0] == lineages.shape[0]
        ids = df_phenos["ROLLINGDB_ID"].values
        
        if binary:
            if drug+"_midpoint" in df_phenos.columns:
                y = (df_phenos[drug+"_midpoint"].values > cc).astype(int)
            else:
                y = df_phenos["phenotype"].values.astype(int)
            assert len(np.unique(y)) == 2
        else:
            y = np.log(df_phenos[drug+"_midpoint"].values)
            
        if bounded_loss:
            if df_phenos[[drug+"_lower_bound", drug+"_upper_bound"]].values.min() <= 0:
                raise ValueError("Some MIC bounds are <= 0")
            
            # some lower bounds are 0, and can't take the logarithm, so exponentiate the prediction later in the bounded loss function
            lower_bounds = df_phenos[drug+"_lower_bound"].values
            upper_bounds = df_phenos[drug+"_upper_bound"].values
        else:
            lower_bounds = np.ones(len(df_phenos))*np.nan
            upper_bounds = np.ones(len(df_phenos))*np.nan        
        
        # use this for bootstrapping. If data_idx is passed, then it must be for training data
        if data_idx is not None:
            X = X[data_idx, :]
            y = y[data_idx]
            lineages = lineages[data_idx, :]
            assert len(X) == len(y) == len(lineages)
            ids = ids[data_idx]
            lower_bounds = lower_bounds[data_idx]
            upper_bounds = upper_bounds[data_idx]                

        # save one-hot encodings in sparse format
        self.one_hot_encodings = X
        self.pheno = y
        self.ID = ids
        self.locus_list = locus_list
        self.longest_locus = X.shape[2]
        self.lineages = lineages
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        if include_lineage:
            self.num_snps = lineages.shape[1]
        else:
            self.num_snps = 0
            
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        print(f"{len(locus_list)} loci, longest locus: {X.shape[2]}, {len(self.pheno)} isolates, {self.num_snps} lineages")
        
    def on_epoch_end(self):
        '''
        Shuffle the data at the end of each epoch so that different epochs see non-identical batches.
        '''
        self.indexes = np.arange(len(self.ID))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        '''
        Returns the number of batches. 
        '''
        return int(np.ceil(len(self.pheno) / self.batch_size))
    
    def __data_generation(self, isolate_idx):
        '''
        isolate_idx is an array of indices corresponding to indices of isolates in the sparse file
        '''
        X_batch = self.one_hot_encodings[isolate_idx, :]
        lineage_batch = self.lineages[isolate_idx, :]
        lower_batch = self.lower_bounds[isolate_idx]
        upper_batch = self.upper_bounds[isolate_idx]
        
        # return values for the batch
        if pd.isnull(lineage_batch).all():
            if pd.isnull(lower_batch).all():
                return X_batch.todense(), self.pheno[isolate_idx]
            else:
                return [X_batch.todense(), lower_batch, upper_batch], self.pheno[isolate_idx]
        else:
            if pd.isnull(lower_batch).all():
                return [X_batch.todense(), lineage_batch], self.pheno[isolate_idx]
            else:
                return [X_batch.todense(), lineage_batch, lower_batch, upper_batch], self.pheno[isolate_idx]
            
        # return values for the batch
        if pd.isnull(lineage_batch).all():
            return X_batch.todense(), self.pheno[isolate_idx]
        else:
            return [X_batch.todense(), lineage_batch], self.pheno[isolate_idx]

    def __getitem__(self, batch_idx):
        '''
        Use this to get a batch of data. batch_idx is the index among the total number of batches. i.e. the 10th batch out of 48
        '''
        # Generate indexes of the batch. If batch_size = 128, then the 2nd batch is from indices 256:384
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
    
        # return the data in however many number of batches
        return self.__data_generation(isolate_idx)
    
    def __getTestData__(self, batch_idx):
        '''
        Use this to return only the isolate IDs and MICs from the generator (not the one-hot encodings). This is mainly for generating the test_predictions.csv file
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.ID[isolate_idx], self.pheno[isolate_idx]
    
    def __getBounds__(self, batch_idx):
        '''
        Use this function to return the lower and upper bounds for the bounded loss function
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.lower_bounds[isolate_idx], self.upper_bounds[isolate_idx]
    
    

def class_weighting_dictionary(y):
    '''
    Returns a dictionary of weights for the binary CNN to weight the loss and metrics functions by. 
    '''
    
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                               classes=np.unique(y),
                                               y=y
                                            )
    
    return dict(zip(np.unique(y), weights))



@tf.function
def bounded_mae(y_true, y_pred):
    '''
    y_test is the log-transformed midpoint of the lower and upper bounds
    '''
    
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
        
    # based on 2-fold dilutions. Might be difficult when integrating multiple data sources
    # but probably can do what was done in the script to generate the bounds in the first place...hopefully
    lower_bounds = tf.cast(K.log(K.exp(y_true) * 2/3), tf.float64)
    upper_bounds = tf.cast(K.log(K.exp(y_true) * 4/3), tf.float64)
    
    # need to enable eager execution to run these 2 lines, but that sometimes causes issues of repeatedly opening .h5 files that I don't understand
#     assert sum((K.greater(lower_bounds, y_true)).numpy()) == 0
#     assert sum((K.less(upper_bounds, y_true)).numpy()) == 0

    # compute the absolute errors first using the log MICs
    errors = K.abs(y_true - y_pred)

    # assign 1 to predicted points that are less than the lower bound or greater than the upper bound
    outside_bounds_mask = tf.cast(K.less(y_pred, lower_bounds) | K.greater(y_pred, upper_bounds), tf.float64)

    # multiply the tensors, so all predicted points within the bounds will have an error of 0
    masked_errors = outside_bounds_mask * errors

    return K.mean(masked_errors)


def bounded_mae_standalone(y_true, y_pred):
    '''
    Use numpy functions, this is for computing the custom error on the test values, not for compiling in tensorflow.
    '''
    
    # based on 2-fold dilutions. Might be difficult when integrating multiple data sources
    # but probably can do what was done in the script to generate the bounds in the first place...hopefully
    lower_bounds = np.log(np.exp(y_true) * 2/3)
    upper_bounds = np.log(np.exp(y_true) * 4/3)
    
    # compute the absolute errors first using the log MICs
    errors = np.abs(y_true - y_pred)
    
    # get indices of predicted values that are outside the bounds, these are assigned 1, the points within the bounds are 0
    outside_bounds_mask = (y_pred < lower_bounds) | (y_pred > upper_bounds).astype(int)
    
    masked_errors = outside_bounds_mask * errors

    # loss in units of log-MIC
    return np.mean(masked_errors)




def conv_nn(longest_locus, num_loci, num_lineages, binary, bounded_loss, filter_size=12, preSoftmax=False):
    '''
    Functional API is the recommended one for multi-input models
    '''

    cnn_input = keras.Input(shape=(5, longest_locus, num_loci), name='seq_input')
    
    if num_lineages > 0:
        mlp_input = keras.Input(shape=(num_lineages, ), name='lineage_input')

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
        dense_inputs = layers.concatenate([cnn_output, mlp_input], axis=1, name='concatenate')
    else:
        dense_inputs = layers.Flatten(name='flatten')(conv_block_2)
    # print(dense_inputs.shape)

    dense = layers.Dense(256, activation='relu', name='dense1')(dense_inputs)
    # print(dense.shape)
    dense = layers.Dense(256, activation='relu', name='dense2')(dense)
    # print(dense.shape)
        
    # for binary model, if preSoftmax is False, then return the preactivation values
    if binary and not preSoftmax:
        output = layers.Dense(1, activation='sigmoid', name='output')(dense)
    # return the pre-activation values. So pre-softmax. Which is also the same for the quantitative model
    else:
        output = layers.Dense(1, activation=None, name='output')(dense)

    if num_lineages > 0:
        model = keras.Model(inputs=[cnn_input, mlp_input], outputs=output)
    else:
        model = keras.Model(inputs=cnn_input, outputs=output)

    # loss function is different
    if binary:
        print("Fitting binary model")
        loss_func = tf.keras.losses.BinaryCrossentropy()
        metrics_lst = [tf.keras.metrics.BinaryAccuracy()]
    else:
        print("Fitting quantitative model")
        if bounded_loss:
            loss_func = bounded_mae
            print("using bounded MAE loss function")
        else:
            loss_func = "mae"
            print("using MAE loss function")
            
        metrics_lst = [tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.MeanSquaredError()
                      ]

    model.compile(optimizer=Adam(learning_rate = np.exp(-1.0 * 9)),
                  loss=loss_func,
                  metrics=metrics_lst, 
                 )
    return model
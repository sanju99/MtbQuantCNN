import numpy as np
import pandas as pd
import os, glob, sparse, pickle
from Bio import SeqIO, Seq
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
import sklearn.metrics
import sklearn.utils
import scipy.stats as st
from saliency_utils import *


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

    # remove any file extensions so that isolates are indexed by their names only WITHOUT any extensions
    seq_dict = SeqIO.to_dict(
        SeqIO.parse(filename, "fasta"),
        key_function=lambda x: x.id.replace(".eff", "").replace(".vcf", "").replace("_freebayes", "").split("/")[-1].split(".cut")[0])

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



def create_all_loci_matrices(config_file):
    '''
    Creates a dictionary of matrices with every nucleotide for every isolate in the given loci. This is so that we can get the CDS in the next function and compute the lengths of the translated proteins for each sample.
    '''

    kwargs = yaml.safe_load(open(config_file, "r"))
    locus_list = kwargs["locus_list"]
    fasta_dir = kwargs["genotype_input_directory"]
    df_phenos = pd.read_csv(kwargs["phenotype_file"])

    gene_coords, sense_dict = get_gene_coords(locus_list, fasta_dir)
    X_matrix_H37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)

    seq_all_loci = {}
    
    for locus in locus_list:
    
        locus_idx = locus_list.index(locus)
        locus_pos_lst = []
    
        seq_lst = [(seq.id, str(seq.seq)) for seq in SeqIO.parse(os.path.join(fasta_dir, f"{locus}.fasta"), "fasta")]        
        aln_len = len(seq_lst[0][1])
        seq_df = pd.DataFrame(seq_lst)
        seq_df.columns = ["Isolate", "Seq"]
        
        # fasta files contain the full VCF file name, without the .vcf extension
        # The ROLLINGDB_ID column is just the isolate name, not the full file path
        seq_df["Isolate"] = [isolate.split(".")[0] for isolate in seq_df["Isolate"].values]
        seq_df = seq_df.set_index("Isolate")
        seq_df = seq_df.loc[np.concatenate([df_phenos["ROLLINGDB_ID"].values, np.array(["MT_H37Rv"])])]
        
        nuc_matrix = seq_df["Seq"].str.split("", expand=True)
        nuc_matrix = nuc_matrix.iloc[:, 1:-1]
        
        pos_including_indels = X_matrix_H37Rv_coords[:nuc_matrix.shape[1], locus_idx]
        k = 0
        
        for pos in pos_including_indels:
            if pd.isnull(pos):
                locus_pos_lst.append(f"{locus}_{k}")
                k += 1
            else:
                locus_pos_lst.append(pos)
    
        assert sum(pd.isnull(locus_pos_lst)) == 0
        nuc_matrix.columns = locus_pos_lst
        seq_all_loci[locus] = nuc_matrix

    return seq_all_loci



def make_CDS_length_df(locus_list, fasta_dir, seqDict_fName):
    '''
    IMPORTANT: Saliency functions must be run before using this function to make seqDict_fName

    Returns: dataframe with shape N_samples x N_loci. Each value is the length of the corresponding CDS
    '''

    # get the dataframe of start and end coordinates from mycobrowser
    h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

    # read in dictionary, where each key is a locus and each value is a dataframe of all samples and positions in the alignment with nucleotides
    seqDict = pickle.load(open(seqDict_fName, "rb"))

    # check that all the same loci are there in both
    assert len(set(seqDict.keys()).symmetric_difference(locus_list)) == 0
    
    locus_peptide_lengths_df = pd.DataFrame(columns=[f"{locus}_length" for locus in locus_list])

    for locus in locus_list:

        with open(os.path.join(fasta_dir, f"{locus}.sh"), "r") as file:
            for line in file.readlines():
                if ".py" in line and ".fasta" in line:
                    lines = line.split(" ")
        
        start_end_lst = []
        
        for arg in lines:
            if arg.isdigit():
                start_end_lst.append(int(arg))
        
        assert len(start_end_lst) == 2
        locus_start, locus_end = start_end_lst[0] + 1, start_end_lst[-1]

        # all the genes in a single locus
        genes_lst = h37Rv_genes.query("Start >= @locus_start & End <= @locus_end").Symbol.values

        # sum the lengths of all the genes within the locus
        gene_peptide_lengths_df = pd.DataFrame(columns=[f"{gene}_length" for gene in genes_lst])

        for gene in genes_lst:

            sense = h37Rv_genes.query("Symbol==@gene")['Strand'].values[0]

            # reverse start and end for negative sense genes because the seqDict dataframes are in translated order (which is easy for translating the nucleotide sequences to AAs)
            if sense == "+":
                start, end = h37Rv_genes.query("Symbol==@gene")[['Start', 'End']].values[0]
            else:
                start, end = h37Rv_genes.query("Symbol==@gene")[['End', 'Start']].values[0]

            # add 1 to the 
            gene_CDS_df = seqDict[locus].loc[:, start:end+1]
    
            # subtract 1 because of the stop character
            WT_protein_length = int((np.abs(end - start) + 1) / 3) - 1
    
            for i, row in gene_CDS_df.iterrows():
            
                # remove gap characters because those represent insertions in other samples
                protein_seq = str(Seq.Seq(''.join(gene_CDS_df.loc[i].values).replace('-', '')).translate())
            
                # replace the stop character and everything after it with gap characters
                if '*' in protein_seq:
                    stop_idx = list(protein_seq).index('*') # by default gets the first index
                    protein_seq = protein_seq[:stop_idx] + "*"*(WT_protein_length - stop_idx)
            
                protein_seq = protein_seq.replace("*", "")
                gene_peptide_lengths_df.loc[i, f"{gene}_length"] = len(protein_seq)

        # sum across the genes in the locus
        locus_peptide_lengths_df[f"{locus}_length"] = gene_peptide_lengths_df.sum(axis=1)
            
    return locus_peptide_lengths_df

    
    
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
    
    output_path = kwargs["output_path"].replace("_lineage", "")
    drug = kwargs["drug"]

    # get table for phenotypes and add the reference strain to match the format of the FASTA files
    df_phenos = pd.read_csv(kwargs['phenotype_file'])
    print(f"Found phenotypes for {len(df_phenos)} strains")
    df_phenos.loc[-1, ['ROLLINGDB_ID', 'category']] = np.array(["MT_H37Rv", "reference"], dtype=object)

    # make table of all genotypes. Index is the identifier
    df_genos = make_genotype_df(kwargs["locus_list"], kwargs['genotype_input_directory'])
    
    df_genos.index = [name.split(".")[0] for name in df_genos.index]
    df_phenos["ROLLINGDB_ID"] = [name for name in df_phenos["ROLLINGDB_ID"]]
    df_genos.index.rename('ROLLINGDB_ID', inplace=True)

    # this makes life easier later. Keep H37Rv in this      
    df_genos = df_phenos[["category", "ROLLINGDB_ID"]].merge(df_genos, on="ROLLINGDB_ID", how="inner")
    df_genos.to_csv(os.path.join(output_path, "df_genos.csv"), index=False)

    # Apply one-hot encoding function to get each isolate sequence
    print('Making one hot encoding for...')
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

    # last row is H37Rv
    X_H37Rv = X_sparse[[-1], :]
    print(f"Reference shape: {X_H37Rv.shape}")
    sparse.save_npz(os.path.join(output_path, "pkl_sparse_ref.npz"), X_H37Rv, compressed=False)

    # if split_train_test:

    #     # Works because index was reset to default
    #     train_indices = df_geno_pheno.query("category=='original_train_set'").index
    #     test_indices = df_geno_pheno.query("category=='original_test_set'").index
    
    #     X_sparse_train = X_sparse[train_indices, :]
    #     X_sparse_test = X_sparse[test_indices, :]
    #     print(f"Training set shape: {X_sparse_train.shape}")
    #     print(f"Test set shape: {X_sparse_test.shape}")
        
    #     sparse.save_npz(os.path.join(output_path, "pkl_sparse_train.npz"), X_sparse_train, compressed=False)
    #     sparse.save_npz(os.path.join(output_path, "pkl_sparse_test.npz"), X_sparse_test, compressed=False)

        
    # H37Rv is the last row, so remove it before saving to avoid redundancy 
    X_data = X_sparse[:-1, :]
    print(f"Full data set shape: {X_data.shape}")
    sparse.save_npz(os.path.join(output_path, "pkl_sparse_full.npz"), X_data, compressed=False)

    
    
    
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

        if not lower <= actual_MIC:
            print("lower problem", lower, actual_MIC)
            
        if not actual_MIC <= upper:
            print("upper problem", actual_MIC, upper)

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
    lower_bounds = np.copy(pred_df[lower_bounds_col].values) #pred_df[lower_bounds_col].values / 2
    upper_bounds = np.copy(pred_df[upper_bounds_col].values) #pred_df[upper_bounds_col].values * 2
    
    lower_bounds[lower_bounds==0] += 1e-6
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)

    # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
    pred_df["compute_error"] = ((pred_df[y_pred_col] <= lower_bounds) | (pred_df[y_pred_col] > upper_bounds)).astype(int)

    # compute the error relative to the bounds, NOT RELATIVE TO THE MIDPOINT (y_test) of each isolate
    # np.clip returns one of the values from lower_bounds or upper_bounds, whichever is closest to the prediction, if the value is outside the bounds
    bound_to_compute_error = np.clip(pred_df[y_pred_col].values, lower_bounds, upper_bounds)
    mae = np.mean((np.abs(bound_to_compute_error - pred_df[y_pred_col])))
    mse = np.mean((np.square(bound_to_compute_error - pred_df[y_pred_col])))

    return mae, mse, pred_df["within_doubling"].mean()




def get_gene_coords(locus_list, fasta_dir):
    '''
    Use this function to get a dataframe of coordinates from the bash scripts used to generate the alignment FASTA files for every locus.
    
    coords_lst is a list of tuples of the start and end coordinates of the genes
    '''
    coords = []
    sense_lst = []
    
    for i, locus in enumerate(locus_list):
        
        # read the coordinates from the file
        with open(os.path.join(fasta_dir, locus + ".sh"), "r") as file:

            for line in file.readlines():
                if ".py" in line and ".fasta" in line:
                    lines = line.split(" ")

        start_end_lst = []
    
        for arg in lines:
            if arg.isdigit():
                start_end_lst.append(int(arg))
        
        assert len(start_end_lst) == 2
        coords.append([start_end_lst[0], start_end_lst[-1]])

        if "NEG" in lines or '"NEG"' in lines:
            sense_lst.append('NEG')
        elif "POS" in lines or '"POS"' in lines:
            sense_lst.append('POS')
        else:
            raise ValueError("Did not find locus sense!")

    gene_coords = pd.DataFrame(coords)
    gene_coords.columns = ["Start", "End"]
    gene_coords["Locus"] = locus_list    
                    
    gene_coords["Length"] = gene_coords["End"] - gene_coords["Start"]
    gene_coords["Sense"] = sense_lst
    gene_coords = gene_coords.set_index("Locus")

    # during this iteration, convert everything to 1-indexing because using np.arange on inverted coordinates is going to get messy
    # so add 1 to the start position, and then both coordinates should be inclusive
    for i, row in gene_coords.iterrows():
        if row["Sense"].upper() == "NEG":
            new_start = row["End"]
            new_end = row["Start"] + 1
            gene_coords.loc[i, "Start"] = new_start
            gene_coords.loc[i, "End"] = new_end
        elif row["Sense"].upper() == "POS":
            gene_coords.loc[i, "Start"] = row["Start"] + 1
            gene_coords.loc[i, "End"] = row["End"]
        else:
            raise ValueError(f"{row['Sense']} is not a valid gene sense!")
            
    assert sum(gene_coords.query("Sense=='NEG'").End > gene_coords.query("Sense=='NEG'").Start) == 0
    assert sum(gene_coords.query("Sense=='POS'").End < gene_coords.query("Sense=='POS'").Start) == 0

    return gene_coords, dict(zip(locus_list, sense_lst))



def make_h37rv_coordinates(gene_coords, locus_list, fasta_dir):
    '''
    gene_coords is 1-indexed, and for negative sense genes, start position is downstream of end position.
    '''
    dfs_list = []
    
    for locus in locus_list:
                
        # read in the sequences for the fasta file
        seqs = [(seq.id, seq.seq) for seq in SeqIO.parse(os.path.join(fasta_dir, f"{locus}.fasta"), "fasta")]
                
        # H37Rv is the last one
        H37Rv = list(seqs[-1][1])
        H37Rv_coords = pd.DataFrame(H37Rv).rename(columns={0:locus})

        # replace deletion characters with nan
        coords_count = []
        pos = gene_coords.loc[locus, "Start"]
        sense = gene_coords.loc[locus, "Sense"]
        length = gene_coords.loc[locus, "Length"]
        assert len(H37Rv) >= length

        for _, row in H37Rv_coords.iterrows():

            if row[locus] == "-":
                coords_count.append(np.nan)
            else:
                coords_count.append(pos)
                if sense.upper() == "POS":
                    pos += 1
                elif sense.upper() == "NEG":
                    pos -= 1
                else:
                    raise ValueError(f"{sense} is not a valid gene sense!")
                    
        # check that the last number is the same as the end position
        assert pd.Series(coords_count).dropna()[:length].values[-1] == gene_coords.loc[locus, "End"]
        
        # combine the locus name with the coordinates and remove the sequence
        H37Rv_coords[locus + "_coord"] = coords_count
        del H37Rv_coords[locus]
        dfs_list.append(H37Rv_coords)
        
    # this is 1-indexed now and in reverse order for negative sense genes
    return pd.concat(dfs_list, axis=1).values




def remove_redundant_sites_for_Reg(df_seq, df_ref, h37Rv_coords):

    # lineage and peptide length inputs
    df_non_seq = df_seq[df_seq.columns[df_seq.columns.str.contains("|".join(["lineage", "length"]))]]
    df_seq = df_seq[df_seq.columns[~df_seq.columns.str.contains("|".join(["lineage", "length"]))]]
    
    # this is a dataframe of length 1
    h37Rv_ref_seq = df_ref[df_ref.columns[(df_ref.loc[0] == 1)]]

    # keep only variables that are not the same everywhere because there is no signal
    df_keep = df_seq.loc[:, df_seq.nunique() > 1]    
    keep_pos = ['_'.join(val.split("_")[:-1]) for val in df_keep.columns]
    
    # when value_counts = 1, it's for indels, where the only options are indel or not. The four nucleotides are 0 for all samples and get dropped in the previous step
    single_allele_pos =  pd.Series(keep_pos).value_counts()[pd.Series(keep_pos).value_counts() == 2].index.values
    multi_allele_pos = pd.Series(keep_pos).value_counts()[pd.Series(keep_pos).value_counts() > 2].index.values
    
    # for positions with only two alleles (REF and ALT, essentially), only need to keep one because they are redundant information and perfectly correlated
    single_allele_pos_keep_cols = []
    
    # preferentially keep the alternative allele because it makes interpretability easier
    for pos in single_allele_pos:

        single_pos_cols = [col for col in df_keep.columns if "_".join(col.split("_")[:-1]) == pos]
        df_keep_single_pos = df_keep[single_pos_cols]
        
        alt_col = list(set(df_keep_single_pos.columns) - set(h37Rv_ref_seq[h37Rv_ref_seq.columns[h37Rv_ref_seq.columns.str.contains(pos)]].columns))

        # this means that the site is biallelic, but NEITHER of the alleles are H37Rv. Randomly select one to keep since they are perfect opposites of each other
        if len(alt_col) == 2:
            print(pos, ','.join(alt_col))
            alt_col = [np.random.choice(alt_col)] # randomly choose one and place it in a list so that it can be added to single_allele_pos_keep_cols, which is a lsit
        elif len(alt_col) == 0:
            raise ValueError("Possible problem: no alleles left!")
        elif len(alt_col) > 2:
            raise ValueError("Possible problem: multiallelic sites in the single_allele_pos vector")

        if len(alt_col) != 1:
            print(alt_col)
            
        single_allele_pos_keep_cols += list(alt_col)
    
    assert len(single_allele_pos_keep_cols) == len(single_allele_pos)

    # concatenate the lineage and peptide length features if they are in this model
    return pd.concat([df_keep[single_allele_pos_keep_cols], df_keep[df_keep.columns[df_keep.columns.str.contains('|'.join(multi_allele_pos))]], df_non_seq], axis=1)

    


def get_single_locus_Reg_input(locus, locus_list, df_phenos, full_matrix, ref_matrix, h37Rv_coords):

    # matrix shape: samples x 5 x locus_length x number of loci
    locus_idx = locus_list.index(locus)
    one_hot_encodings = full_matrix.shape[1]
    longest_locus = full_matrix.shape[2]
    num_loci = full_matrix.shape[3]
    assert one_hot_encodings == 5

    # turn the matrices into dataframes for easy manipulation
    df_train_test = pd.DataFrame(np.reshape(full_matrix[:, :, :, locus_idx], (full_matrix.shape[0], one_hot_encodings * longest_locus), order='F'))
    df_ref = pd.DataFrame(np.reshape(ref_matrix[:, :, :, locus_idx], (1, one_hot_encodings * longest_locus), order='F'))

    # need to get all the nucleotide positions to name the columns. This makes manipulation easier and is also useful to keep track of which positions went into the model (interpretability)
    # k is an iterator to keep track of indels
    seq_coords = []
    k = 0
    
    for coord in h37Rv_coords[:, locus_idx]:

        # indels -- position is NaN, so give unique names that are a concatenation of the locus and an index
        if pd.isnull(coord):
            coord = f"indel{k}"
            k += 1
        else:
            coord = str(int(coord))
            
        seq_coords += [f"{locus}_{coord}_{nuc}" for nuc in BASE_TO_COLUMN.keys()]
    
    assert len(seq_coords) == len(df_train_test.columns)
    df_train_test.columns = seq_coords

    # this is a dataframe of length 1
    df_ref.columns = seq_coords

    return df_train_test, df_ref




def make_H37Rv_CDS_length_df(locus_list, fasta_dir):
    '''
    IMPORTANT: Saliency functions must be run before using this function to make seqDict_fName

    Two indices: Locus and Gene because there are multiple genes in a single locus.

    The goal of this function is to make a dataframe of the H37Rv protein lengths of all genes in all loci in a model. 
    Then for insilico mutagenesis, you can get the position of the early stop codon and update the peptide length by subtracting the number of AAs that would be truncated from the H37Rv length

    NOTE: This is not supported for WHO catalog frameshift mutations that cause early stop codons. HOWEVER, in the PZA models that this is being used for, there are no such frameshift mutations
    because they increase the size of the alignment so they were removed from this analysis step. 
    '''

    # get the dataframe of start and end coordinates from mycobrowser
    h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

    gene_peptide_lengths_df = pd.DataFrame(columns=["Locus", "Gene", "Length"]).set_index(["Locus", "Gene"])

    for locus in locus_list:

        with open(os.path.join(fasta_dir, f"{locus}.sh"), "r") as file:
            for line in file.readlines():
                if ".py" in line and ".fasta" in line:
                    lines = line.split(" ")
        
        start_end_lst = []
        
        for arg in lines:
            if arg.isdigit():
                start_end_lst.append(int(arg))
        
        assert len(start_end_lst) == 2
        locus_start, locus_end = start_end_lst[0] + 1, start_end_lst[-1]

        # all the genes in a single locus
        genes_lst = h37Rv_genes.query("Start >= @locus_start & End <= @locus_end").Symbol.values
        single_locus_peptide_lengths_df = pd.DataFrame(columns=[f"{gene}_length" for gene in genes_lst])

        # sum the lengths of all the genes within the locus
        WT_protein_length = 0

        for gene in genes_lst:

            sense = h37Rv_genes.query("Symbol==@gene")['Strand'].values[0]

            # reverse start and end for negative sense genes because the seqDict dataframes are in translated order (which is easy for translating the nucleotide sequences to AAs)
            if sense == "+":
                start, end = h37Rv_genes.query("Symbol==@gene")[['Start', 'End']].values[0]
            else:
                start, end = h37Rv_genes.query("Symbol==@gene")[['End', 'Start']].values[0]

            # subtract 1 because of the stop character
            gene_peptide_lengths_df.loc[(locus, gene), :] =  int((np.abs(end - start) + 1) / 3) - 1

    return gene_peptide_lengths_df.reset_index()
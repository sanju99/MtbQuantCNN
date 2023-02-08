import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}


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
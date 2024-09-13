import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO, Seq

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
import sklearn.metrics
import sklearn.utils
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
# from saliency_utils import *

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")
model_loci[['Start', 'End']] = model_loci[['Start', 'End']].astype(int)

amino_acid_biophysical_properties = pd.read_csv("./data_processing/protein_seqs/biophysical_properties_AA.csv", index_col=[0])

# get the dataframe of start and end coordinates from mycobrowser
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")
h37Rv_regions = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_v4.csv")




def expand_dims_for_rescaling(array, dims_to_expand, X_target):
    '''
    Use this function to expand the dimensions of the mean or standard deviation array so that you can use array operations to scale inputs
    '''

    # the smallest axis index should not be negative. the largest axis index shouldn't be greater than the number of dimensions in the target array
    assert np.min(dims_to_expand) >= 0
    assert np.max(dims_to_expand) < X_target.ndim
    
    array = np.expand_dims(array, axis=dims_to_expand)

    for dim in dims_to_expand:
        array = np.repeat(array, repeats=X_target.shape[dim], axis=dim)

    return array
    


def get_genes_lst(locus_list):
    """
    Return a list of the individual genes (excluding pseudogenes) associated with the argument loci, in the same order.
    """

    # need this for both peptide lengths and amino acid property flags
    genes_lst = []

    # this was the same code used to get the genes, so the order will be the same
    for locus in locus_list:
    
        # don't need to add 1 to start because it is 1-indexed
        locus_start, locus_end = model_loci.query("Locus==@locus")[['Start', 'End']].values[0]

        # all the genes in a single locus. Exclude pseudogenes and other noncoding regions because they're very likely not translated to a functional protein
        genes_lst += list(h37Rv_genes.query("Start > @locus_start & End <= @locus_end & ~Product.str.contains('pseudogene', case=False)").Symbol.values)
        # genes_lst += list(h37Rv_regions.query("Start > @locus_start & Stop <= @locus_end & ~Product.str.contains('pseudogene', case=False)").Name.values)

    # assert len(genes_lst) >= len(locus_list)
    return genes_lst


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
        key_function=lambda x: x.id.replace(".eff", "").replace(".vcf", "").replace("_freebayes", "").replace("_variants", "").split("/")[-1].split(".cut")[0])

    # create a dictionary of identifier: sequence
    for identifier, sequence in seq_dict.items():
        seq_dict[identifier] = str(sequence.seq)

    df = pd.DataFrame.from_dict(seq_dict, orient='index')
    gene_name = os.path.basename(filename).split(".")[0]
    df.columns = [gene_name]

    return df


def make_genotype_df(genotype_input_directory, locus_list, amino_acid=False):
    """
	Parameters
	----------
	genotype_input_directory: str
		path to directory containing fasta files of genotype inputs

	Returns
	-------
	pd.DataFrame:
		indexed by isolate name, one column per locus/gene
	"""
    # Make a df that combines all genotype data
    dfs_list = []

    if amino_acid:
        fastas = [f"{genotype_input_directory}/{locus}_AA.fasta" for locus in locus_list] #glob.glob(f"{genotype_input_directory}/*_AA.fasta")
        print(f"{len(fastas)} genes!")
    else:
        fastas = [f"{genotype_input_directory}/{locus}.fasta" for locus in locus_list] #list(set(glob.glob(f"{genotype_input_directory}/*.fasta")) - set(glob.glob(f"{genotype_input_directory}/*_AA.fasta")))
        print(f"{len(fastas)} loci!")

    for fName in fastas:
        if not os.path.isfile(fName):
            raise ValueError(f"{fName} is not a valid file name")

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




def create_X(df_geno_pheno, amino_acid=False, L_longest=None):
    """
	Create an input X matrix, with output dimensions:
		n_strains x 5 (one-hot) x longest locus length x no. of loci

    L_longest should only be specified when making input matrices for additional data to get predictions on. 
    
    If the alignment length of the additional data is SMALLER than that of the training data, then need to increase the length in that axis by padding so that the shapes match
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
            if "one_hot" in column or "_biophys" in column:
                shapes[column] = df_geno_pheno.loc[df_geno_pheno.index[0], column].shape[0]

        return shapes

    shapes = _get_shapes(df_geno_pheno)

    # Length of longest gene locus
    n_genes = len(shapes)

    # if not passed in, compute from the data
    if L_longest is None:
        L_longest = max(list(shapes.values()))

    # Number of strains in model
    n_strains = df_geno_pheno.shape[0]

    # define shape of matrix - fill with zeros (effectively accomplishes padding)
    if amino_acid:
        X = np.zeros((n_strains, 3, L_longest, n_genes))
    else:
        X = np.zeros((n_strains, 5, L_longest, n_genes))

    # for each strain and gene locus
    for idx, strain in enumerate(df_geno_pheno.index):
        
        for gene_index, gene in enumerate(shapes.keys()):

            # shape is longest_locus x 3 or 5
            single_gene_matrix = df_geno_pheno.loc[strain, gene]

            # single_gene_matrix.shape[0] is the length of the locus
            X[idx, :, range(0, single_gene_matrix.shape[0]), gene_index] = single_gene_matrix

    return X



def make_nucleotide_matrices(drug, locus_list, seq_data_path, df_phenos, genotype_input_directory, split_groups=False):

    # get table for phenotypes and add the reference strain to match the format of the FASTA files
    print(f"Found phenotypes for {len(df_phenos)} strains")
    df_phenos.loc[-1, ['ROLLINGDB_ID', 'category']] = np.array(["MT_H37Rv", "reference"], dtype=object)

    # make table of all genotypes. Index is the identifier
    df_genos = make_genotype_df(genotype_input_directory, locus_list, amino_acid=False)
    
    df_genos.index = [name.split(".")[0] for name in df_genos.index]
    df_genos.index.rename('ROLLINGDB_ID', inplace=True)

    # this makes life easier later. Keep H37Rv in this
    if split_groups:
        df_genos = df_phenos[["category", "ROLLINGDB_ID"]].merge(df_genos, on="ROLLINGDB_ID", how="inner")
    else:
        df_genos = df_phenos[["ROLLINGDB_ID"]].merge(df_genos, on="ROLLINGDB_ID", how="inner")
    
    df_genos.to_csv(os.path.join(seq_data_path, "df_genos.csv"), index=False)

    # Apply one-hot encoding function to get each isolate sequence
    print('Making one hot encoding for...')
    
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1

        # df_genos[locus + "_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))
        
        # not sure why, but this has to be done first otherwise an error is thrown
        df_genos[f"{locus}_one_hot"] = None

        # get the reference amino acid sequence
        ref_seq = df_genos.loc[df_genos['ROLLINGDB_ID']=='MT_H37Rv'][locus].values[0]
        ref_seq_one_hot = get_one_hot(ref_seq)
    
        # Create a list of repeated arrays for the rows that match the condition
        matched_indices = df_genos[df_genos[locus] == ref_seq].index.values
        ref_repeat = [ref_seq_one_hot for _ in matched_indices]
        
        # Assign the repeated arrays to the appropriate rows
        df_genos.loc[matched_indices, f"{locus}_one_hot"] = ref_repeat
        
        # Apply the function only to rows that do not match the condition
        df_genos.loc[~df_genos.index.isin(matched_indices), f"{locus}_one_hot"] = df_genos.loc[~df_genos.index.isin(matched_indices), locus].apply(get_one_hot)
        
    if "category" in df_genos and "category" in df_phenos:
        del df_genos["category"]

    # combined dataframe of all genotypes and phenotypes
    df_geno_pheno = df_phenos.merge(df_genos, how='inner', on="ROLLINGDB_ID")

    # Create a one-hot-encoding matrix with dimensions
    # num_isolate x 5 x locus_length x num_loci
    X = create_X(df_geno_pheno)
    X_sparse = sparse.COO(X)

    # train-validation-test-ref splitting, which is only done for the training data (not addl data like TRUST or insilico predictions)
    if split_groups:
    
        # last row is H37Rv
        X_H37Rv = X_sparse[[-1], :]
        print(f"Reference shape: {X_H37Rv.shape}")
        sparse.save_npz(os.path.join(seq_data_path, "pkl_sparse_ref.npz"), X_H37Rv, compressed=False)
    
        # H37Rv is the last row, so remove it before saving to avoid redundancy 
        X_sparse = X_sparse[:-1, :]
        
        # Reset index first
        df_geno_pheno = df_geno_pheno.reset_index(drop=True)
        train_val_idx = df_geno_pheno.query("category in ['train_set', 'validation_set']").index
        test_idx = df_geno_pheno.query("category == 'test_set'").index

        # Separate train + val from test to keep test aside only for final predictions
        sparse.save_npz(os.path.join(seq_data_path, "pkl_sparse_train_val.npz"), X_sparse[train_val_idx, :], compressed=False)
        sparse.save_npz(os.path.join(seq_data_path, "pkl_sparse_test.npz"), X_sparse[test_idx, :], compressed=False)
    
    # save all data (including H37Rv) to a single pkl file. This is for getting model predictions on addl data
    else:
        sparse.save_npz(os.path.join(seq_data_path, "pkl_sparse_full.npz"), X_sparse, compressed=False)



def make_AA_property_matrices(drug, genes_lst, seq_data_path, df_phenos, genotype_input_directory, L_longest=None, split_groups=False):
    
    print(f"Found phenotypes for {len(df_phenos)} strains")
    df_phenos.loc[-1, ['ROLLINGDB_ID', 'category']] = np.array(["MT_H37Rv", "reference"], dtype=object)
    
    # get table for phenotypes and add the reference strain to match the format of the FASTA files      
    # make table of all genotypes. Index is the identifier
    df_genos = make_genotype_df(genotype_input_directory, genes_lst, amino_acid=True)
    
    df_genos.index = [name.split(".")[0] for name in df_genos.index]
    df_genos.index.rename('ROLLINGDB_ID', inplace=True)

    # Add the isolate IDs and category to df_genos before saving. Keep H37Rv in this
    if split_groups:
        df_genos = df_phenos[["category", "ROLLINGDB_ID"]].merge(df_genos, on="ROLLINGDB_ID", how="inner")
    else:
        df_genos = df_phenos[["ROLLINGDB_ID"]].merge(df_genos, on="ROLLINGDB_ID", how="inner")
        
    df_genos.to_csv(os.path.join(seq_data_path, "df_protein_seqs.csv"), index=False)
    
    # Apply one-hot encoding function to get each isolate sequence
    print(f'Making biophysical property matrices for {len(genes_lst)} genes...')
    for gene in genes_lst:
        
        print("...", gene)
        lengths = [len(seq) for seq in df_genos[f"{gene}_AA"]]
        assert len(np.unique(lengths)) == 1

        # df_genos[f"{gene}_biophys"] = df_genos[f"{gene}_AA"].apply(np.vectorize(convert_AA_seq_to_property_matrix))
        
        # not sure why, but this has to be done first otherwise an error is thrown
        df_genos[f"{gene}_biophys"] = None

        # get the reference amino acid sequence
        ref_seq = df_genos.loc[df_genos['ROLLINGDB_ID']=='MT_H37Rv'][f"{gene}_AA"].values[0]
        ref_AA_matrix = convert_AA_seq_to_property_matrix(ref_seq)
    
        # Create a list of repeated arrays for the rows that match the condition
        matched_indices = df_genos[df_genos[f"{gene}_AA"] == ref_seq].index.values
        ref_repeat = [ref_AA_matrix for _ in matched_indices]
        
        # Assign the repeated arrays to the appropriate rows
        df_genos.loc[matched_indices, f"{gene}_biophys"] = ref_repeat
        
        # Apply the function only to rows that do not match the condition
        df_genos.loc[~df_genos.index.isin(matched_indices), f"{gene}_biophys"] = df_genos.loc[~df_genos.index.isin(matched_indices), f"{gene}_AA"].apply(convert_AA_seq_to_property_matrix)
            
    if "category" in df_genos and "category" in df_phenos:
        del df_genos["category"]

    # combined dataframe of all genotypes and phenotypes
    df_geno_pheno = df_phenos.merge(df_genos, how='inner', on="ROLLINGDB_ID")
    
    X = create_X(df_geno_pheno, amino_acid=True, L_longest=L_longest)

    # train-validation-test-ref splitting, which is only done for the training data (not addl data like TRUST or insilico predictions)
    if split_groups:
        
        # last row is H37Rv
        X_H37Rv = X[[-1], :]
        print(f"Reference shape: {X_H37Rv.shape}")
        np.save(os.path.join(seq_data_path, "pkl_AA_ref.npy"), X_H37Rv)
        
        # H37Rv is the last row, so remove it before saving to avoid redundancy 
        X = X[:-1, :]

        # Reset index first
        df_geno_pheno = df_geno_pheno.reset_index(drop=True)
        train_val_idx = df_geno_pheno.query("category in ['train_set', 'validation_set']").index
        test_idx = df_geno_pheno.query("category == 'test_set'").index

        # Separate train + val from test to keep test aside only for final predictions
        np.save(os.path.join(seq_data_path, "pkl_AA_train_val.npy"), X[train_val_idx, :])
        np.save(os.path.join(seq_data_path, "pkl_AA_test.npy"), X[test_idx, :])
    else:
        np.save(os.path.join(seq_data_path, "pkl_AA_full.npy"), X)





def create_all_loci_matrices(config_file, fasta_dir=None, isolates_lst=None):
    '''
    Creates a dictionary of matrices with every nucleotide for every isolate in the given loci. This is so that we can get the CDS in the next function and compute the lengths of the translated proteins for each sample.
    '''

    kwargs = yaml.safe_load(open(config_file, "r"))
    locus_list = kwargs["tier1_loci"] + kwargs['tier2_loci']

    if fasta_dir is None:
        fasta_dir = kwargs['genotype_input_directory']
    
    if isolates_lst is None:
        isolates_lst = pd.read_csv(kwargs["phenotype_file"])['ROLLINGDB_ID'].values

    if "MT_H37Rv" not in isolates_lst:
        isolates_lst = list(isolates_lst) + ["MT_H37Rv"]            

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
        seq_df["Isolate"] = [isolate.replace(".eff", "").replace(".vcf", "").replace("_freebayes", "").replace("_variants", "").replace("_combinedCodons", "") for isolate in seq_df["Isolate"].values]
        seq_df = seq_df.set_index("Isolate")
        seq_df = seq_df.loc[isolates_lst]

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



def make_CDS_length_df(drug, locus_list, fasta_dir, seqDict_fName):
    '''
    Returns: dataframe with shape N_samples x N_loci. Each value is the length of the corresponding CDS
    '''
    
    # read in dictionary, where each key is a locus and each value is a dataframe of all samples and positions in the alignment with nucleotides
    seqDict = pd.read_pickle(seqDict_fName)

    # check that all the same loci are there in both
    assert len(set(seqDict.keys()).symmetric_difference(locus_list)) == 0
    
    # locus_peptide_lengths_df = pd.DataFrame(columns=[f"{locus}_length" for locus in locus_list])
    locus_peptide_lengths_df = []

    for locus in locus_list:

        # don't need to add 1 to start because it is 1-indexed
        locus_start, locus_end, locus_sense = model_loci.query("Locus==@locus")[['Start', 'End', 'Sense']].values[0]

        # all the genes in a single locus. Exclude pseudogenes because they're very likely not translated to a functional protein
        genes_lst = h37Rv_genes.query("Start >= @locus_start & End <= @locus_end & ~Product.str.contains('pseudogene', case=False)").Symbol.values

        # sum the lengths of all the genes within the locus
        gene_peptide_lengths_df = pd.DataFrame(columns=[f"{gene}_length" for gene in genes_lst])

        for gene in genes_lst:

            gene_sense = h37Rv_genes.query("Symbol==@gene")['Strand'].values[0]

            # reverse start and end for negative sense genes because the seqDict dataframes are in translated order (which is easy for translating the nucleotide sequences to AAs)
            if gene_sense == "+":
                gene_start, gene_end = h37Rv_genes.query("Symbol==@gene")[['Start', 'End']].values[0]
            elif gene_sense == '-':
                gene_start, gene_end = h37Rv_genes.query("Symbol==@gene")[['End', 'Start']].values[0]
            else:
                raise ValueError(f"{gene_sense} is not a valid gene sense!")

            # the column names are mixed floats (positions) and strings (insertion sites), so convert them all to strings, then you can use slicing
            # to get all columns between two column names. The names must all be integers or all strings, so need to do this conversion
            str_column_names = [col if type(col) == str else str(int(col)) for col in seqDict[locus].columns]
            seqDict[locus].columns = str_column_names
    
            # add 1 to the end because the end is exclusive, then convert to strings. But if the last column name is the same as end, you will get an error when you add 1. So check first
            if gene_sense == locus_sense:
                gene_CDS_df = seqDict[locus].loc[:, str(gene_start):str(gene_end)]
                # if str(end) == seqDict[locus].columns[-1]:
                #     gene_CDS_df = seqDict[locus].loc[:, str(start):str(end)]
                # else:
                #     gene_CDS_df = seqDict[locus].loc[:, str(start):str(end+1)]
            else:
                # end is exclusive and gene sense is positive while locus sense is negative, 
                if locus_sense == '-' and gene_sense == '+':
                    gene_CDS_df = seqDict[locus].loc[:, str(gene_end):str(gene_start)]
                else:
                    raise ValueError("Figure out this edge case!")
            
            # subtract 1 because of the stop character
            WT_protein_length = int((np.abs(gene_end - gene_start) + 1) / 3) - 1
            print(f"{gene} H37Rv length: {WT_protein_length}")

            # index is the isolate name
            for isolate, row in gene_CDS_df.iterrows():
            
                # remove gap characters because those represent insertions in other samples
                region_seq = Seq.Seq(''.join(gene_CDS_df.loc[isolate].values).replace('-', ''))

                # in the case that there is a gene that is of opposite sense as the rest of the locus
                if locus_sense != gene_sense:
                    region_seq = Seq.Seq(reverse_complement(region_seq))
                    
                # must be one of the start codons. Otherwise, the sequence length is 0 because no protein is translated
                if region_seq[:3] not in ['ATG', 'CTG', 'GTG', 'TTG', 'ATA', 'ATC', 'ATT']:
                    gene_peptide_lengths_df.loc[isolate, f"{gene}_length"] = 0
                else:         
                    protein_seq = str(region_seq.translate())
                    
                    # checked above that the first three nucleotides are in the start codon list above. But double check that the translated AA is one of these
                    assert protein_seq[0] in ['M', 'L', 'V', 'I']

                    if '*' in protein_seq:
                        stop_idx = list(protein_seq).index('*') # by default gets the first index
                        protein_seq = protein_seq[:stop_idx] + "*"*(WT_protein_length - stop_idx)
                
                    protein_seq = protein_seq.replace("*", "")
                    gene_peptide_lengths_df.loc[isolate, f"{gene}_length"] = len(protein_seq)                    

        # sum across the genes in the locus
        # locus_peptide_lengths_df[f"{locus}_length"] = gene_peptide_lengths_df.sum(axis=1)
        locus_peptide_lengths_df.append(gene_peptide_lengths_df)
            
    # return locus_peptide_lengths_df
    return pd.concat(locus_peptide_lengths_df, axis=1)



def create_AA_alns(drug, locus_list, fasta_dir, seqDict_fName):
    '''
    Writes FASTA files for the individual protein sequences associated with the model loci. 

    Each gene has a separate FASTA file. They are left aligned sequences, so all sequences are the same length.
    '''

    # read in dictionary, where each key is a locus and each value is a dataframe of all samples and positions in the alignment with nucleotides
    seqDict = pd.read_pickle(seqDict_fName)

    # check that all the same loci are there in both
    assert len(set(seqDict.keys()).symmetric_difference(locus_list)) == 0
    
    for locus in locus_list:
    
        # don't need to add 1 to start because it is 1-indexed
        locus_start, locus_end, locus_sense = model_loci.query("Locus==@locus")[['Start', 'End', 'Sense']].values[0]
    
        # all the genes in a single locus. Exclude pseudogenes because the genes are not fully functional. May be transcribed, probably not translated due to presence of early stop codons
        genes_lst = h37Rv_genes.query("Start >= @locus_start & End <= @locus_end & ~Product.str.contains('pseudogene', case=False)").Symbol.values
    
        for gene in genes_lst:
    
            # keep track of isolates
            translated_sequences = []
    
            gene_sense = h37Rv_genes.query("Symbol==@gene")['Strand'].values[0]
    
            # reverse start and end for negative sense genes because the seqDict dataframes are in translated order (which is easy for translating the nucleotide sequences to AAs)
            if gene_sense == "+":
                gene_start, gene_end = h37Rv_genes.query("Symbol==@gene")[['Start', 'End']].values[0]
            elif gene_sense == '-':
                gene_start, gene_end = h37Rv_genes.query("Symbol==@gene")[['End', 'Start']].values[0]
            else:
                raise ValueError(f"{gene_sense} is not a valid gene sense!")
    
            # the column names are mixed floats (positions) and strings (insertion sites), so convert them all to strings, then you can use slicing
            # to get all columns between two column names. The names must all be integers or all strings, so need to do this conversion
            str_column_names = [col if type(col) == str else str(int(col)) for col in seqDict[locus].columns]
            seqDict[locus].columns = str_column_names
    
            if gene_sense == locus_sense:
                # when you do loc like this, both values are inclusive. I don't know why
                gene_CDS_df = seqDict[locus].loc[:, str(gene_start):str(gene_end)]

            else:
                # gene sense is positive while locus sense is negative. Removed these cases so this shouldn't happen, but leaving the code here just in case
                if locus_sense == '-' and gene_sense == '+':
                    gene_CDS_df = seqDict[locus].loc[:, str(gene_end):str(gene_start)]
                else:
                    raise ValueError("Figure out this edge case!")
            
            # subtract 1 because of the stop character
            WT_protein_length = int((np.abs(gene_end - gene_start) + 1) / 3) - 1
            print(f"{gene} H37Rv length: {WT_protein_length}")
    
            # index is the isolate name
            for isolate, row in gene_CDS_df.iterrows():
            
                # remove gap characters because those represent insertions in other samples
                region_seq = Seq.Seq(''.join(gene_CDS_df.loc[isolate].values).replace('-', ''))

                # in the case that there is a gene that is of opposite sense as the rest of the locus
                if locus_sense != gene_sense:
                    region_seq = Seq.Seq(reverse_complement(region_seq))
                
                # must be one of the start codons. Otherwise, there is no sequence because no protein is translated
                if region_seq[:3] not in ['ATG', 'CTG', 'GTG', 'TTG', 'ATA', 'ATC', 'ATT']:
                    protein_seq = ''
                else:         
                    protein_seq = str(region_seq.translate())
                    
                    # checked above that the first three nucleotides are in the start codon list above. But double check that the translated AA is one of these
                    assert protein_seq[0] in ['M', 'L', 'V', 'I']
    
                    # replace first amino acid with M (because that's what happens post-translation). Can't change the first character because strings are immutable though
                    protein_seq = 'M' + protein_seq[1:]
    
                    if '*' in protein_seq:
                        stop_idx = list(protein_seq).index('*') # by default gets the first index
                        protein_seq = protein_seq[:stop_idx] + "*"*(WT_protein_length - stop_idx)
                
                    protein_seq = protein_seq.replace("*", "")
    
                translated_sequences.append([isolate, protein_seq])
    
            # find the longest protein length. All others have to be padded with '-' characters
            lengths = []
    
            for isolate, seq in translated_sequences:
                lengths.append(len(seq))
    
            longest_protein_length = np.max(lengths)
            del lengths

            # write to a new FASTA file for amino acid sequences. Put quotes around the name for special characters like in oxyR'
            with open(f"{fasta_dir}/{gene}_AA.fasta", "w+") as file:
                for isolate, seq in translated_sequences:
                    file.write(f">{isolate}\n")
                    file.write(seq + '-' * (longest_protein_length - len(seq)) + '\n')




def convert_AA_seq_to_property_matrix(aa_seq):
    '''
    Convert a single amino acid sequence to a matrix of biophysical properties. 
    
    The properties are Eisenberg-Weiss consensus hydrophobicity scale, molecular weight (molar mass), and isoelectronic point at 25 degrees C.
    '''
    
    # Convert sequences to matrix
    biophys_matrix = []

    for aa in aa_seq:
        if aa not in amino_acid_biophysical_properties.index.values:
            raise ValueError(f"{aa} is not in the biophysical property table")
            
        biophys_matrix.append(amino_acid_biophysical_properties.loc[aa, :].values.flatten())

    biophys_matrix = np.array(biophys_matrix)
    
    assert biophys_matrix.shape[0] == len(aa_seq)
    assert biophys_matrix.shape[1] == amino_acid_biophysical_properties.shape[1]

    # return the matrix, which is of shape amino acid coord x 3
    return biophys_matrix


    
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




def compute_binary_metrics(y_true, y_pred, binary_thresh, binarize=False):
        
    # binarize using the critical concentration
    # see if the upper bound is greater than the critical concentration. If so, resistant. If the upper bound is equal to the CC, then it is susceptible because it dies at the CC.
    if binarize:
        y_true_binary = (y_true > binary_thresh).astype(int)
        y_pred_binary = (y_pred > np.log2(binary_thresh)).astype(int)
    else:
        y_true_binary = np.copy(y_true)
        y_pred_binary = np.copy(y_pred)
        
    assert len(np.unique(y_true_binary)) <= 2
    assert len(np.unique(y_pred_binary)) <= 2
    
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_binary, y_pred_binary).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)
    precision = tp / (tp+fp)
    acc = sklearn.metrics.accuracy_score(y_true_binary, y_pred_binary)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true_binary, y_pred_binary)
    F1 = sklearn.metrics.f1_score(y_true_binary, y_pred_binary)

    return pd.DataFrame({"Sensitivity": sens,
                         "Specificity": spec,
                         "Precision": precision,
                         "Accuracy": acc,
                         "Balanced_Acc": balanced_acc,
                         "F1": F1,
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




def boundedLoss_predict(pred_df, y_pred_col="y_pred", lower_bounds_col="lower", upper_bounds_col="upper"):
    '''
    y_true and y_pred are log-MICs. lower_bounds and upper_bounds are exponentiated. 
    
    This function returns bounded MAE, MSE, and the proportion of points measured within 1 MIC doubling (1 log2 unit)
    ''' 
    
    del_cols = [f"{y_pred_col}_exp", "within_doubling", "within_1bin", "compute_error", f"{lower_bounds_col}_rounded", f"{upper_bounds_col}_rounded"]

    for col in del_cols:
        if col in pred_df.columns:
            del pred_df[col]

    # first add essential agreement (proportion within 1 doubling dilution)
    # not always helpful because some "doubling" dilutions are not exact, i.e. 0.3, 0.6, 0.125, 0.5. But the number is here if needed
    pred_df[f"{y_pred_col}_exp"] = np.round(np.exp2(pred_df[y_pred_col]).astype(float), 2)
    
    pred_df.loc[(pred_df[lower_bounds_col] / 2 <= pred_df[f"{y_pred_col}_exp"]) & 
                (pred_df[upper_bounds_col] * 2 >= pred_df[f"{y_pred_col}_exp"])
                , "within_doubling"] = 1

    pred_df.loc[(pred_df[f"{y_pred_col}_exp"] == 0.06) & (pred_df[lower_bounds_col] == 0.12), "within_doubling"] = 1
    pred_df["within_doubling"] = pred_df["within_doubling"].fillna(0).astype(int)
        
    # make copies to avoid changing the original dataframe
    lower_bounds = np.copy(pred_df[lower_bounds_col].values) #pred_df[lower_bounds_col].values / 2
    upper_bounds = np.copy(pred_df[upper_bounds_col].values) #pred_df[upper_bounds_col].values * 2
    
    lower_bounds[lower_bounds==0] += 1e-10
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)

    # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
    pred_df["compute_error"] = ((pred_df[y_pred_col].values <= lower_bounds) | (pred_df[y_pred_col].values > upper_bounds)).astype(int)

    # compute the error relative to the bounds, NOT RELATIVE TO THE MIDPOINT (y_test) of each isolate
    # np.clip returns one of the values from lower_bounds or upper_bounds, whichever is closest to the prediction, if the value is outside the bounds
    # if the test values are within the bounds, the values themselves are returned
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

    # don't filter by drug because some loci (Tier 2 only) are associated with multiple drugs
    gene_coords = model_loci.query("Locus in @locus_list")                    
    gene_coords["Length"] = (gene_coords["End"] - gene_coords["Start"]) # don't need to add 1 because these are 0-indexed half open intervals.
    gene_coords = gene_coords.set_index("Locus")

    # during this iteration, convert everything to 1-indexing because using np.arange on inverted coordinates is going to get messy
    # so add 1 to the start position, and then both coordinates should be inclusive
    for i, row in gene_coords.iterrows():
        if row["Sense"] == "-":
            new_start = row["End"]
            new_end = row["Start"] + 1
            gene_coords.loc[i, "Start"] = new_start
            gene_coords.loc[i, "End"] = new_end
        elif row["Sense"] == "+":
            gene_coords.loc[i, "Start"] = row["Start"] + 1
            gene_coords.loc[i, "End"] = row["End"]
        else:
            raise ValueError(f"{row['Sense']} is not a valid gene sense!")
            
    assert sum(gene_coords.query("Sense=='-'").End > gene_coords.query("Sense=='-'").Start) == 0
    assert sum(gene_coords.query("Sense=='+'").End < gene_coords.query("Sense=='+'").Start) == 0

    return gene_coords.drop_duplicates(), dict(zip(gene_coords.index.values, gene_coords.Sense.values))
    


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
        # except:
        #     print(len(H37Rv), length, locus)
        #     exit()

        for _, row in H37Rv_coords.iterrows():

            if row[locus] == "-":
                coords_count.append(np.nan)
            else:
                coords_count.append(pos)
                # increment because we're going from 5' to 3'
                if sense == "+":
                    pos += 1
                # decrement because we're going from 3' to 5'
                elif sense == "-":
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

    


def get_single_matrix_regression_input(matrix, keep_idx=None, num_keep_channels=None):

    # subset the loci if specified. num_keep_loci should be an integer
    if num_keep_channels is not None:
        matrix = matrix[:, :, :, :num_keep_channels]

    # use keep_idx to keep only specifid isolates (row index)
    if keep_idx is not None:
        matrix = matrix[keep_idx, :, :, :]

    # matrix shape: samples x 3/5 x locus_length x number of loci
    num_features = matrix.shape[1]
    longest_locus = matrix.shape[2]
    num_loci = matrix.shape[3]
    assert num_features in [3, 5]

    # reshape to a two-dimensional form that can be passed into the regression model
    # order doesn't matter because they're all treated as separate features by regression. Just need to make sure it's consistent across different inputs
    return np.reshape(matrix, (matrix.shape[0], num_features * longest_locus * num_loci), order='C')



def get_all_regression_inputs(df_train_val, df_test, seq_data_path, test_seq_data_path, locus_list, include_lineage=False, include_amino_acid_properties=False):
    '''
    Use this function to create the train, val, and test matrices for all regression problems: Ridge regression, gradient boosting, and XGBoost
    '''
    # read in matrices of input sequences. These are not in the ridge directory, they are the same as the matrices used by the CNN
    X_train_val = sparse.load_npz(f"{seq_data_path}/pkl_sparse_train_val.npz").todense()
    X_AA_train_val = np.load(f"{seq_data_path}/pkl_AA_train_val.npy")
    
    # different path for the test data because of the possibility of using different AF threshold
    X_AA_test = np.load(f"{test_seq_data_path}/pkl_AA_test.npy")

    # train_idx = df_train_val.query("category=='train_set'").index.values
    # val_idx = df_train_val.query("category=='validation_set'").index.values
    
    # nucleotide inputs
    X_train = get_single_matrix_regression_input(X_train_val, keep_idx=None, num_keep_channels=len(locus_list))
    # X_val = get_single_matrix_regression_input(X_train_val, keep_idx=val_idx, num_keep_channels=len(locus_list))
    
    # different path for the test data because of the possibility of using different AF threshold
    X_test = get_single_matrix_regression_input(sparse.load_npz(f"{test_seq_data_path}/pkl_sparse_test.npz").todense(), keep_idx=None, num_keep_channels=len(locus_list))
    
    # get the genes to keep (this is needed for both amino acid and gene peptide lengths inputs)
    # keep only the specified loci for this model
    genes_list = get_genes_lst(locus_list)
    
    # check that feature shapes are the same across the matrices
    # assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_train.shape[1] == X_test.shape[1]
    
    # get the number of unique values per site in the training dataset
    unique_values = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X_train)
    
    # features with variation, only keep these for computational efficiency. These will be indexes
    # it's most important to do this for the nucleotide features because they are the largest
    # drop sites with no signal, meaning all isolates have the same value, so unique_values == 1
    keep_features = np.where(unique_values > 1)[0]
    print(f"Keeping {len(keep_features)} nucleotide features in the model")
    
    X_train = X_train[:, keep_features]
    # X_val = X_val[:, keep_features]
    X_test = X_test[:, keep_features]
    
    if include_lineage:
    
        lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
        assert len(np.unique(lineages)) == 2

        X_train = np.concatenate([X_train, lineages.loc[df_train_val['ROLLINGDB_ID']]], axis=1)
        # X_train = np.concatenate([X_train, lineages.loc[df_train_val['ROLLINGDB_ID']].values[train_idx, :]], axis=1)
        # X_val = np.concatenate([X_val, lineages.loc[df_train_val['ROLLINGDB_ID']].values[val_idx, :]], axis=1)
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
    
            # X_AA_train = X_AA_train_val[train_idx, :]
            train_mean = X_AA_train_val.mean(axis=(0, 2))
            train_std = X_AA_train_val.std(axis=(0, 2))
    
            np.save(train_mean_fName, train_mean)
            np.save(train_std_fName, train_std)
    
        else:
            train_mean = np.load(train_mean_fName)
            train_std = np.load(train_std_fName)
    
        # train_mean and train_std are only 2 dimensions. So need to duplicate the arrays to make the full dataset and protein sequence lengths
        # scale all 3 matrices
        X_AA_train_val = (X_AA_train_val - expand_dims_for_rescaling(train_mean, (0, 2), X_AA_train_val)) / expand_dims_for_rescaling(train_std, (0, 2), X_AA_train_val)
        X_AA_test = (X_AA_test - expand_dims_for_rescaling(train_mean, (0, 2), X_AA_test)) / expand_dims_for_rescaling(train_std, (0, 2), X_AA_test)
        
        train_AA_matrix = get_single_matrix_regression_input(X_AA_train_val, keep_idx=None, num_keep_channels=len(genes_list))
        # val_AA_matrix = get_single_matrix_regression_input(X_AA_train_val, keep_idx=val_idx, num_keep_channels=len(genes_list))
        test_AA_matrix = get_single_matrix_regression_input(X_AA_test, keep_idx=None, num_keep_channels=len(genes_list))
        
        X_train = np.concatenate([X_train, train_AA_matrix], axis=1)
        # X_val = np.concatenate([X_val, val_AA_matrix], axis=1)
        X_test = np.concatenate([X_test, test_AA_matrix], axis=1)
    
    
    # check that feature shapes are the same across the matrices
    # assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_train.shape[1] == X_test.shape[1]

    # return X_train, X_val, X_test
    return X_train, X_test
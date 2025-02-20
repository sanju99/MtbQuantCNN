import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

# need lineage processing functions from here
from data_utils import *

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"

model_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")


class MtbGeneDataset(Sequence):

    def __init__(self, drug, df_phenos, nuc_sparse_file, aa_property_file, seq_data_path, binary, cc, tier1_loci, tier2_loci=None, no_lineage_SNPs=False, data_subset=None, data_idx=None, shuffle_phenos=False, include_lineage=False, include_peptide_lengths=False, include_amino_acid_properties=False, lowAF=False, bounded_loss=False, batch_size=128, shuffle_batches=True):
        '''
        Sparse files for both the training and testing sets are available, so read those. 

        There is a single pickle file for the sequence matrix input. 

        If data_subset is specified, then the data will be subsetted to get ONLY the specified subset

        If data_idx is specified, then within the pickle file (after data_subset has been applied), get the specified indices of the data (actual indices, not sample names)
        '''

        if lowAF:
            file_suffix = "_lowAF"
        else:
            file_suffix = ""
        
        # read in the one-hot encoded nucleotide file and the protein embeddings file. Both are in the same sample order
        X = sparse.load_npz(nuc_sparse_file.replace(".npz", f"{file_suffix}.npz"))

        if tier2_loci is not None:
            nuc_locus_list = list(tier1_loci) + list(tier2_loci)
        else:
            nuc_locus_list = list(tier1_loci)

        # will be the same dimension as the last one of X if tier2 is included. If not, will be the length of tier1_loci
        last_locus_idx = len(nuc_locus_list)

        # when only including tier 1 genes, get only the first N idx
        # dimensions are num_isolates x 5 x longest_locus x N_loci
        X = X[:, :, :, :last_locus_idx]

        # need this for both peptide lengths and amino acid property flags
        genes_lst = get_genes_lst(nuc_locus_list)

        # get only the training or testing set
        if data_subset is not None:

            # get indices to subset the pickle file, then reset the index
            keep_idx = df_phenos.query("category==@data_subset").index.values
            X = X[keep_idx, :]            
            df_phenos = df_phenos.query("category==@data_subset").reset_index(drop=True)

        ids = df_phenos["ROLLINGDB_ID"].values

        # get phenotypes only if bounded_loss is True. If it's False, then we're just getting predictions on this dataset 
        if bounded_loss:
            if binary:
                if f"{drug}_midpoint" in df_phenos.columns:
                    y = (df_phenos[f"{drug}_midpoint"].values >= cc).astype(int)
                else:
                    y = df_phenos["Binary"].values.astype(int)
                assert len(np.unique(y)) == 2
            else:
                y = np.log2(df_phenos[f"{drug}_midpoint"]).values
        else:
            y = np.ones(len(df_phenos)) * np.nan

        # use this for cross-validation. If data_idx is passed, then it must be for training data
        if data_idx is not None:
            X = X[data_idx, :]
            y = y[data_idx]
            ids = ids[data_idx]

        # don't need to be standard-scaled because the only possible values are 0 and 1, just like with the nucleotide inputs
        if include_lineage:

            lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])

            # for insilico mutagenesis, the samples will not be in the lineage matrix. They need to be all zeros anyway
            if no_lineage_SNPs:
                lineages = np.zeros((df_phenos.shape[0], lineages.shape[1]))
                assert len(np.unique(lineages)) == 1
            else:
                # order lineages in the same order as the phenotypes dataframe (which is in the same order as the sparse file)
                lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]].values
                assert len(np.unique(lineages)) == 2
            
            if data_idx is not None:
                lineages = lineages[data_idx, :]

        # # must be scaled because they range from 0 to inf technically
        # if include_peptide_lengths:

        #     locus_peptide_lengths = pd.read_csv(os.path.join(seq_data_path, f"gene_peptide_lengths{file_suffix}.csv"), index_col=[0])

        #     # # get the lengths of peptides in H37Rv (reference) in dictionary form mapping column names to ref lengths
        #     # ref_lengths = dict(locus_peptide_lengths.loc['MT_H37Rv'])

        #     # reorder the isolates to match the rest of the data, then get the values to make it a matrix
        #     locus_peptide_lengths = locus_peptide_lengths.loc[df_phenos["ROLLINGDB_ID"]]

        #     # keep only loci in the specified list
        #     locus_peptide_lengths = locus_peptide_lengths[[f"{locus}_length" for locus in genes_lst]]

        #     # # get the effective length of each gene, so the range should be 0 - ~1.5, which is similar to the range of other features
        #     # # the other features are all in the 0 - 1 range, so don't need to normalize them            
        #     # for col in locus_peptide_lengths.columns:
        #     #     locus_peptide_lengths[col] = locus_peptide_lengths[col] / ref_lengths[col]
            
        #     # convert to matrix form
        #     locus_peptide_lengths = locus_peptide_lengths.values
            
        #     # there should not be any NaNs or negative values
        #     assert np.sum(pd.isnull(locus_peptide_lengths)) == 0
        #     assert np.min(locus_peptide_lengths) >= 0
            
        #     if data_idx is not None:
        #         locus_peptide_lengths = locus_peptide_lengths[data_idx, :]

        # must be scaled because they range from 0 to inf technically due to the amino acid molecular weights
        if include_amino_acid_properties:

            # read in AA property matrix
            X_amino_acid = np.load(aa_property_file)

            # compute the mean and SD of the training set to scale validation and test data later
            # scale across the sample axis (0) and the length of the amino acid sequence (2). Don't scale different biophysical properties together (1), or different genes together (3)
            train_mean_fName = os.path.join(seq_data_path, "AA_train_mean.npy")
            train_std_fName = os.path.join(seq_data_path, "AA_train_std.npy")
            
            if not os.path.isfile(train_mean_fName) or not os.path.isfile(train_std_fName):

                df_train = pd.read_csv(os.path.join(data_dir, drug, "data_for_model.csv")).query("category in ['train_set', 'validation_set']").reset_index(drop=True)
                train_idx = df_train.query("category=='train_set'").index.values
                del df_train

                # need to generalize this. Doesn't work for both AF = 25% and TRUST/in silico muts
                # X_amino_acid_train = np.load(os.path.join(seq_data_path, "pkl_AA_train_val.npy"))
                X_amino_acid_train = np.load(os.path.join(results_dir, drug, "pkl_AA_train_val.npy"))
                X_amino_acid_train = X_amino_acid_train[train_idx, :]
                
                train_mean = X_amino_acid_train.mean(axis=(0, 2))
                train_std = X_amino_acid_train.std(axis=(0, 2))
                del X_amino_acid_train

                np.save(train_mean_fName, train_mean)
                np.save(train_std_fName, train_std)

            else:
                train_mean = np.load(train_mean_fName)
                train_std = np.load(train_std_fName)

            # train_mean and train_std are only 2 dimensions. So need to duplicate the arrays to make the full dataset and protein sequence lengths
            train_mean = expand_dims_for_rescaling(train_mean, (0, 2), X_amino_acid)
            train_std = expand_dims_for_rescaling(train_std, (0, 2), X_amino_acid)
            
            # scale
            X_amino_acid_rescaled = (X_amino_acid - train_mean) / train_std

            # check shapes match
            assert X_amino_acid.shape == X_amino_acid_rescaled.shape
            del X_amino_acid

            # same as for the nucleotide inputs, but need to get the genes list first
            last_protein_idx = len(genes_lst)

            # get only the genes in tier 1 if needed
            X_amino_acid_rescaled = X_amino_acid_rescaled[:, :, :, :last_protein_idx]
            
            # this can be done after the rescaling because rescaling uses the full training set to compute mean and std
            if data_subset is not None:
                X_amino_acid_rescaled = X_amino_acid_rescaled[keep_idx, :]
                
            if data_idx is not None:
                X_amino_acid_rescaled = X_amino_acid_rescaled[data_idx, :]
            
        if bounded_loss:
            lower_bounds = df_phenos[f"{drug}_lower_bound"].values
            upper_bounds = df_phenos[f"{drug}_upper_bound"].values

            if data_idx is not None:
                lower_bounds = lower_bounds[data_idx]
                upper_bounds = upper_bounds[data_idx]   
            
        # shuffle -- use this for performing the permutation test for saliency scores. WORKS IN PLACE
        # need to shuffle all 3 arrays because the custom loss function for ordinal regression relies on lower_bounds and upper_bounds NOT y actually
        # though technically don't need y at all, but keep in case we switch back to an older loss function where it is used
        if shuffle_phenos:
            np.random.shuffle(y)
            np.random.shuffle(lower_bounds)
            np.random.shuffle(upper_bounds)        

        # save one-hot encodings in sparse format
        self.one_hot_encodings = X           
        self.pheno = y.reshape(-1, 1)
        self.ID = ids
        
        self.nuc_locus_list = nuc_locus_list
        self.longest_locus = self.one_hot_encodings.shape[2]
        self.genes_lst = genes_lst
        self.bounded_loss = bounded_loss        
        self.include_lineage = include_lineage
        self.include_peptide_lengths = include_peptide_lengths
        self.include_amino_acid_properties = include_amino_acid_properties
        
        if bounded_loss:
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds
            
        if include_lineage:
            self.num_snps = lineages.shape[1]
            self.lineages = lineages
        else:
            self.num_snps = 0

        if include_peptide_lengths:
            self.num_peptide_lengths = locus_peptide_lengths.shape[1]
            self.peptide_lengths = locus_peptide_lengths
            assert len(self.genes_lst) == self.num_peptide_lengths
        else:
            self.num_peptide_lengths = 0

        self.mlp_data_shape = self.num_snps + self.num_peptide_lengths

        if include_amino_acid_properties:
            self.amino_acid_properties = X_amino_acid_rescaled
            self.num_proteins = self.amino_acid_properties.shape[3]
            self.longest_protein = self.amino_acid_properties.shape[2]
        else:
            self.num_proteins = 0
            self.longest_protein = 0
            
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.on_epoch_end()

        print(f"{len(self.nuc_locus_list)} nucleotide loci, longest locus: {self.longest_locus}, {self.num_proteins} proteins for {','.join(self.genes_lst)}, longest protein: {self.longest_protein}, {len(self.pheno)} isolates, {self.num_snps} lineages, {self.num_peptide_lengths} peptide lengths")


    def on_epoch_end(self):
        '''
        Shuffle the data at the end of each epoch so that different epochs see non-identical batches.
        '''
        self.indexes = np.arange(len(self.ID))
        
        if self.shuffle_batches:
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

        inputs_lst = [self.one_hot_encodings[isolate_idx, :].todense()]

        # the first dimension is always the sample dimension, so subset on that
        if self.include_amino_acid_properties:
            inputs_lst += [self.amino_acid_properties[isolate_idx, :]]

        # add each MLP component if needed, then concatentate them into a single vector
        mlp_only_inputs = []

        # the first dimension is always the sample dimension, so subset on that
        if self.include_lineage:
            mlp_only_inputs.append(self.lineages[isolate_idx, :])

        if self.include_peptide_lengths:
            mlp_only_inputs.append(self.peptide_lengths[isolate_idx, :])

        # combine these two into a single vector. The nucleotide and AA inputs are kept as separate elements in the list
        if self.include_lineage or self.include_peptide_lengths:
            mlp_only_inputs = np.concatenate(mlp_only_inputs, axis=1)

            # convert the MLP inputs into a list and append to the inputs list. If not, then leave the list as it is
            inputs_lst += [mlp_only_inputs]

        if self.bounded_loss:
            inputs_lst += [self.lower_bounds[isolate_idx]]
            inputs_lst += [self.upper_bounds[isolate_idx]]

        # check that the length is in [1, 5]: 5 = nucleotide, AA, MLP vector, lower, upper
        assert len(inputs_lst) >= 1 and len(inputs_lst) <= 5

        return inputs_lst, self.pheno[isolate_idx]


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
        Use this to return only the isolate IDs and MICs from the generator (not the one-hot encodings). This is just for generating the test_predictions.csv file
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.ID[isolate_idx], np.squeeze(self.pheno[isolate_idx])
    
    def __getBounds__(self, batch_idx):
        '''
        Use this function to return the lower and upper bounds for the bounded loss function
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.lower_bounds[isolate_idx], self.upper_bounds[isolate_idx]
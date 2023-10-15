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



class MtbGeneDataset(Sequence):

    def __init__(self, sparse_file, phenotype_file, drug, locus_list, fasta_dir, binary, cc, train_or_test=None, shuffle_phenos=False, include_lineage=False, include_peptide_length=False, bounded_loss=False, data_idx=None, batch_size=128, shuffle=True):
        '''
        Sparse files for both the training and testing sets are available, so read those. 

        There is a single pickle file for the sequence matrix input. 

        If train_or_test is specified, then the data will be subsetted to get ONLY the train or test set

        If data_idx is specified, then within the pickle file (after train_or_test has been applied), get the specified indices of the data (actual indices, not sample names)
        '''
        
        # read in the one-hot encoded files and convert from sparse to dense format. read in the phenotypes file
        X = sparse.load_npz(sparse_file)
        df_phenos = pd.read_csv(phenotype_file)
        
        # get only the training or testing set
        if train_or_test is not None:

            # get indices to subset the pickle file, then reset the index
            keep_idx = df_phenos.query("category==@train_or_test").index.values
            X = X[keep_idx, :]
            
            df_phenos = df_phenos.query("category==@train_or_test").reset_index(drop=True)

        ids = df_phenos["ROLLINGDB_ID"].values

        if binary:
            if f"{drug}_midpoint" in df_phenos.columns:
                y = (df_phenos[f"{drug}_midpoint"].values >= cc).astype(int)
            else:
                y = df_phenos["Binary"].values.astype(int)
            assert len(np.unique(y)) == 2
        else:
            y = np.log2(df_phenos[f"{drug}_midpoint"]).values

        # use this for cross-validation. If data_idx is passed, then it must be for training data
        if data_idx is not None:
            X = X[data_idx, :]
            y = y[data_idx]
            ids = ids[data_idx]
        
        if include_lineage:
            lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
            assert len(np.unique(lineages.values)) == 2

            # order lineages in the same order as the phenotypes dataframe (which is in the same order as the sparse file)
            lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]].values

            if data_idx is not None:
                lineages = lineages[data_idx, :]

        if include_peptide_length:

            locus_peptide_lengths = pd.read_csv(os.path.join(os.path.dirname(sparse_file).replace('_lineage', '').replace('_peptide', ''), "locus_peptide_lengths.csv"), index_col=[0])

            # reorder the isolates to match the rest of the data, then get the values to make it a matrix
            locus_peptide_lengths = locus_peptide_lengths.loc[df_phenos["ROLLINGDB_ID"]].values

            if data_idx is not None:
                locus_peptide_lengths = locus_peptide_lengths[data_idx, :]
                    
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
        self.locus_list = locus_list
        self.longest_locus = X.shape[2]
        
        self.bounded_loss = bounded_loss        
        self.include_lineage = include_lineage
        self.include_peptide_length = include_peptide_length

        if bounded_loss:
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds
            
        if include_lineage:
            self.num_snps = lineages.shape[1]
            self.lineages = lineages
        else:
            self.num_snps = 0

        if include_peptide_length:
            self.num_peptides = locus_peptide_lengths.shape[1]
            self.peptide_lengths = locus_peptide_lengths
        else:
            self.num_peptides = 0
            
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        print(f"{len(locus_list)} loci, longest locus: {X.shape[2]}, {len(self.pheno)} isolates, {self.num_snps} lineages, {self.num_peptides} summed peptide lengths")
        
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

        inputs_lst = [X_batch.todense()]
        
        # return values for the batch
        if self.include_lineage:
            if self.include_peptide_length:
                inputs_lst += [np.concatenate([self.lineages[isolate_idx, :], self.peptide_lengths[isolate_idx, :]], axis=1)]
            else:
                inputs_lst += [self.lineages[isolate_idx, :]]
        else:
            if self.include_peptide_length:
                inputs_lst += [self.peptide_lengths[isolate_idx, :]]
            
        if self.bounded_loss:
            inputs_lst += [self.lower_bounds[isolate_idx]]
            inputs_lst += [self.upper_bounds[isolate_idx]]

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
        Use this to return only the isolate IDs and MICs from the generator (not the one-hot encodings). This is mainly for generating the test_predictions.csv file
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.ID[isolate_idx], np.squeeze(self.pheno[isolate_idx])
    
    def __getBounds__(self, batch_idx):
        '''
        Use this function to return the lower and upper bounds for the bounded loss function
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.lower_bounds[isolate_idx], self.upper_bounds[isolate_idx]
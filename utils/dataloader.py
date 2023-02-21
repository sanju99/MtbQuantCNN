import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence



class MtbGeneDataset(Sequence):

    def __init__(self, sparse_file, phenotype_file, drug, locus_list, train_or_test, binary, cc, shuffle_phenos=False, include_lineage=False, bounded_loss=False, data_idx=None, batch_size=128, shuffle=True):
        '''
        Sparse files for both the training and testing sets are available, so read those. 
        Use data_idx only when performing cross-validation, where data_idx is train_idx or test_idx
        '''
        
        # read in the one-hot encoded files and convert from sparse to dense format. read in the phenotypes file
        X = sparse.load_npz(sparse_file)
        df_phenos = pd.read_csv(phenotype_file)
        
        # make lineage matrix. Do this before getting only the train or test set so that all lineages are there
        lineages = pd.get_dummies(df_phenos["Lineage"])
        lineages.index = df_phenos["ROLLINGDB_ID"]
        
        # the following checks are a bit extra, but including them anyway to be very careful
        # check that the sum of each row (isolate) is 1, i.e. each isolate has only 1 lineage
        assert lineages.sum(axis=1).unique() == np.array([1])

        # minimum number of samples in a particular lineage group should not be 0
        assert lineages.sum(axis=0).min() > 0
        
        # get only the training or testing set and subset the lineage matrix too
        df_phenos = df_phenos.query("category==@train_or_test").reset_index(drop=True)
        lineages = lineages.loc[df_phenos["ROLLINGDB_ID"]]
        
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
            
        # shuffle -- use this for performing the permutation test for saliency scores
        if shuffle_phenos:
            np.random.shuffle(y)
            
        if bounded_loss:
            if df_phenos[[f"{drug}_lower_bound", f"{drug}_upper_bound"]].values.min() < 0:
                raise ValueError("Some MIC bounds are < 0")
            
            # some lower bounds are 0, and can't take the logarithm, so exponentiate the prediction later in the bounded loss function
            lower_bounds = df_phenos[f"{drug}_lower_bound"].values
            upper_bounds = df_phenos[f"{drug}_upper_bound"].values
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
        self.pheno = y.reshape(-1, 1)
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
                return [X_batch.todense(), 
                        lower_batch,
                        upper_batch
                       ], self.pheno[isolate_idx]
        else:
            if pd.isnull(lower_batch).all():
                return [X_batch.todense(), lineage_batch], self.pheno[isolate_idx]
            else:
                return [X_batch.todense(), 
                        lineage_batch, 
                        lower_batch, 
                        upper_batch
                       ], self.pheno[isolate_idx]
            
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
        return self.ID[isolate_idx], np.squeeze(self.pheno[isolate_idx])
    
    def __getBounds__(self, batch_idx):
        '''
        Use this function to return the lower and upper bounds for the bounded loss function
        '''
        isolate_idx = self.indexes[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.lower_bounds[isolate_idx], self.upper_bounds[isolate_idx]
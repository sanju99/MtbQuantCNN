### script to execute GeO score calculation for a single protein
### meant to be executed as a standalone job

import os, glob, sys, argparse, warnings
import pandas as pd
import numpy as np
from evcouplings.compare import DistanceMap
from copy import deepcopy
from scipy.stats import ks_2samp
from evcouplings.visualize.pymol import (
    pymol_pair_lines, pymol_mapping
)
import time, yaml

import warnings
warnings.filterwarnings("ignore")



def compute_weight_matrix(dist_matrix):
    """
    Creates a weight matrix using the input distance matrix
    
    Parameters
    ----------
    dist_matrix: Evcouplings.compare.DistanceMap.dist_matrix
    
    Returns
    -------
    np.array
        LxL matrix of 1/distance for all pairs of residues, 0 on diagonal
    """
    
    weights = np.power(dist_matrix, -1)
    np.fill_diagonal(weights, 0)

    return weights



def calculate_G_scores(X, w):
    """
    Calculates the Getis-Ord statistic for all L positions in the structure
    
    Parameters:
    X: np.array
        an Lx1 matrix of the mutations for each
    
    w: np.array
        an LxL weight matrix 
    
    """
    G_scores = np.zeros(X.shape)
    
    # calculate number of positions in the DistanceMap
    n = w.shape[0]

    # Iterate through each position in the distance map
    for i in range(X.shape[0]):
        
        # Need to calculate the GeO score WITHOUT using the number of mutations
        # at the current position
        zeroed_X = deepcopy(X)
        zeroed_X[i] = 0
        
        # sum of weights times X
        sum_of_wx = np.dot(w[i,:].reshape(1,-1), X)[0][0]

        # mean of x times sum of weights
        mean_x_times_weights = np.mean(zeroed_X) * np.sum(w[i, :])

        S = np.sqrt(np.sum(np.power(zeroed_X,2))/n - np.power(np.mean(zeroed_X),2))

        K = np.sqrt(n * np.sum(np.power(w[i,:], 2)) - np.power(np.sum(w[i,:]),2)) / np.sqrt(n-1)

        # Compute G_score
        G_i = (sum_of_wx - mean_x_times_weights) / (S * K)  

        G_scores[i,0] = G_i
        
    return G_scores



def random_G_score_table(runs, X, w, dm):
    # Simulating random distributions of mutations

    column_dict = {}
    column_dict["i"] = list(dm.residues_i.id)
    random_X = deepcopy(X)
    for r in range(runs):
        np.random.shuffle(random_X)

        G_scores = calculate_G_scores(random_X, w)
        column_dict[f"iteration_{r}"] = G_scores.flatten()
    shuffle_table = pd.DataFrame(column_dict)
    return shuffle_table



def compute_GeO_score_with_permutation(values_fName):

    to_analyze = pd.read_csv(values_fName)
    
    # not all residues may have been resolved in the crystal structure, so keep only those in the test statistic dataframe
    keep_residues = dm.residues_i.id.values
    to_analyze = to_analyze.query("residue in @keep_residues").reset_index(drop=True)
    
    print(f"Computing GeO scores for {len(to_analyze)} residues with {NUM_SHUFFLES} permutations")
    
    w = compute_weight_matrix(dm.dist_matrix)
    
    # X = create_X(to_analyze, dm.residues_i, column="average")
    X = to_analyze[['average']].values
    G_scores = calculate_G_scores(X, w)
    
    df = pd.DataFrame([dm.residues_i.id.values, # full name (chain_residue)
                       [res.split('_')[0] for res in dm.residues_i.id.values], # chain name
                       dm.residues_i.coord_id.values, # within-chain residue name 
                       G_scores.flatten()]).T
    
    df.columns = ["residue", "chain", "coordinate", "G_score"]
    df.to_csv(f"{absolute_path}/{output_path}/G_scores.csv", index=False)
    G_score_df = df

    shuffle_table = random_G_score_table(NUM_SHUFFLES, X, w, dm)
    shuffle_table.to_csv(f"{absolute_path}/{output_path}/random_GeO_iterations_{NUM_SHUFFLES}.csv.gz", compression='gzip', index=False)

    # don't need this because we're doing per-residue clustering, not global protein clustering
    # ks_results = []
    
    # # is the distribution of per-residue GeO scores significantly different from the per-residue scores of a randomly shuffled protein?
    # for i in range(1,NUM_SHUFFLES+1):
    #     ks = ks_2samp(G_scores.flatten(), shuffle_table.iloc[:,i].values.flatten())
    #     ks_results.append([ks.statistic, ks.pvalue])
    
    # result_table = pd.DataFrame(ks_results, columns=["KS_score", "pvalue"])
    # result_table.sort_values("pvalue", inplace=True)
    # result_table.to_csv(f"{absolute_path}/{output_path}/{prefix}_random_GeO_pvalues_{NUM_SHUFFLES}.csv.gz", compression='gzip', index=False)
    
    # ### Now compute GeO versus combination of all shuffled isolates
    # ks = ks_2samp(G_scores.flatten(), shuffle_table.iloc[:,1::].values.flatten())
    # result_table_full = pd.DataFrame([[ks.statistic, ks.pvalue]], columns=["score", "pvalue"])
    # result_table_full.to_csv(f"{absolute_path}/{output_path}/random_GeO_full_distribution_{NUM_SHUFFLES}.csv")



parser = argparse.ArgumentParser()

parser.add_argument("-o", dest='output_path', type=str, required=True, help='Name of the directory where the values dataframe is stored and where reuslts will be saved. Must exist before running this script')
parser.add_argument('-p', dest='prot_id', type=str, required=True, help='Name of the protein to perform clustering on. Must correspond to a distance map in the distance_maps directory')

cmd_line_args = parser.parse_args()
output_path = cmd_line_args.output_path
prot_id = cmd_line_args.prot_id
NUM_SHUFFLES = 10000
absolute_path = "/home/sak0914/MtbQuantCNN/spatial_clustering"

# read the distance map
# DON'T INCLUDE .CSV OR .NPY IN THE FILE EXTENSION
dm = DistanceMap.from_file(f"{absolute_path}/distance_maps/{prot_id}")

# will threshold afterwards based on the value in the "average" column to distinguish between neutral mutations and S-associated mutations
compute_GeO_score_with_permutation(f"{absolute_path}/{output_path}/values_to_cluster.csv")
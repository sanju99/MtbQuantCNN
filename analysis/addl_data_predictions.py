import pandas as pd
import numpy as np
import glob, os, yaml, sparse, itertools, subprocess, sys, argparse, pickle, tracemalloc
from Bio import SeqIO, Seq

import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# load all utils functions
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
sys.path.append("utils")
from data_utils import *
from analysis_utils import *
from model_utils import *
from dataloader import MtbGeneDataset

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

coll_2014 = pd.read_csv("/home/sak0914/who-analysis/data/coll2014_SNP_scheme.tsv", sep="\t")
coll_2014["lineage"] = coll_2014["#lineage"].str.replace("lineage", "")
del coll_2014["#lineage"]

lineages_matrix = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])

# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

# boolean argument for including lineage SNPs, default value False. If you include the flag, it is considered True
parser.add_argument('--lineage', action='store_true', help='Flag to add lineage SNPs to model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--tier2', action='store_true', help='Flag to add tier 2 loci to the model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--amino-acid', dest='amino_acid', action='store_true', help='Flag to add amino acid biophysical properties to the model')

parser.add_argument('--TRUST', dest='TRUST_data', action='store_true', help='Get MIC predictions on TRUST data')

parser.add_argument('--insilico-muts', dest='insilico_muts', action='store_true', help='Get MIC predictions for insilico mutations')

parser.add_argument('--saturation-muts', dest='saturation_muts', action='store_true', help='Get MIC predictions for insilico mutations')

parser.add_argument("--locus", type=str, help='For in silico mutagenesis, specify the locus for which you want to get variant predictions')

parser.add_argument("--gene", type=str, help='For saturation mutagenesis, specify the gene for which you want to get variant predictions')

parser.add_argument('--predict', action='store_true', help='Get MIC predictions. If not specified, just create input data for the additional samples')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
TRUST_data = cmd_line_args.TRUST_data
insilico_muts = cmd_line_args.insilico_muts
saturation_muts = cmd_line_args.saturation_muts
locus = cmd_line_args.locus
gene = cmd_line_args.gene
get_predictions = cmd_line_args.predict

count_flags_true = 0
flags_lst = np.array([TRUST_data, insilico_muts, saturation_muts])

if sum(flags_lst) > 1:
    raise ValueError("Please only specify only one of optional arguments: TRUST-data, insilico-muts, isolates-span-cc, and saturation-muts")

if sum(flags_lst) == 0:
    raise ValueError("Please specify one of optional arguments: TRUST-data, insilico-muts, isolates-span-cc, and saturation-muts")
        
kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
tier1_loci = kwargs["tier1_loci"]

if include_tier2:
    tier2_loci = kwargs["tier2_loci"]
else:
    tier2_loci = []

filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
phenotype_file = kwargs["phenotype_file"]
binary_thresh = kwargs["binary_thresh"]

if 'output_path' in kwargs.keys():
    output_path = kwargs["output_path"]
else:
    output_path = f"{results_path}/{drug}"
    
seq_data_path = f"{results_path}/{drug}"
training_data_path = output_path

binary = False
bounded_loss = False # only need predictions here, no training, so it will save on memory
    
if include_lineage:
    output_path += "_lineage"

if include_tier2:
    output_path += "_tier2"

if include_amino_acid_properties:
    output_path += "_amino_acid"
    
# add an additional subdirectory for the additional dataset
if TRUST_data:
    subdir = "TRUST"
    no_lineage_SNPs = False # default behavior to include lineage information in training / predictions
    
if insilico_muts:
    subdir = f"inSilico_analysis/{locus}"

    # set True so that lineage SNPs are excluded and all SNPs are 0
    no_lineage_SNPs = True

    # this command line argument must be specified
    if get_predictions:
        assert locus is not None

    # this would only happen for in silico mutagenesis, if there are no mutations for that locus
    if not os.path.isdir(os.path.join(data_dir, drug, subdir)):
        print(f"There are no in silico mutations for {locus}")
        exit()

if saturation_muts:

    # this command line argument must be specified
    assert gene is not None
    
    if gene not in drug_loci.Locus.values:
        
        locus = drug_loci.query("Locus.str.contains(@gene)").Locus.values

        if len(locus) == 0:
            if gene in ['gyrA', 'gyrB']:
                locus = ['gyrBA']
            elif gene in ['rpoB', 'rpoC']:
                locus = ['rpoBC']
            elif gene in ['mmpL5', 'mmpS5']:
                locus = ['mmpLS5']
            
        assert len(locus) == 1
        locus = locus[0]

    subdir = f"inSilico_analysis/saturation_mutagenesis/{gene}"

    # set True so that lineage SNPs are excluded and all SNPs are 0
    no_lineage_SNPs = True

data_dir = os.path.join(data_dir, drug, subdir)
genotype_input_directory = f"{data_dir}/fastas"

seq_data_path =  os.path.join(seq_data_path, subdir)
output_path = os.path.join(output_path, subdir)
print(f"Saving prediction results to {output_path}")

if not os.path.isdir(seq_data_path):
    os.makedirs(seq_data_path)

print(f"seq data path: {seq_data_path}")

if not os.path.isdir(output_path):
    os.makedirs(output_path)
    
# get the lists of samples to get MIC predictions for
if TRUST_data:
    # df_samples = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/20240124_MIC_pass_genoQC.csv")

    df_samples = pd.DataFrame([os.path.basename(val).split(".")[0] for val in pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/vcf_full_paths.txt", sep='\t', header=None)[0].values])

    # rename the column to ROLLINGDB_ID for consistency with the TRUST dataframe
    df_samples.columns = ['ROLLINGDB_ID']

    # increased the size of the alignment for Rv3236c
    if drug == 'PZA':
        df_samples = df_samples.query("ROLLINGDB_ID not in ['MFS-370', 'MFS-371']")

elif insilico_muts:

    df_samples = pd.DataFrame([os.path.basename(val).split(".")[0] for val in pd.read_csv(f"{data_dir}/WHO_mutations.txt", sep='\t', header=None)[0].values])

    # rename the column to ROLLINGDB_ID for consistency with the TRUST dataframe
    df_samples.columns = ['ROLLINGDB_ID']

elif saturation_muts:
    
    df_samples = pd.DataFrame([os.path.basename(val).split(".")[0] for val in pd.read_csv(f"{data_dir}/{gene}_mutations.txt", sep='\t', header=None)[0].values])

    # rename the column to ROLLINGDB_ID for consistency with the TRUST dataframe
    df_samples.columns = ['ROLLINGDB_ID']


# add MT_H37Rv to the dataframe so that it is included in the predictions
df_samples.loc[-1, 'ROLLINGDB_ID'] = 'MT_H37Rv'

print(f"Getting predictions for {len(df_samples)} strains")

# create input files if they don't exist
pkl_fName = os.path.join(seq_data_path, "pkl_sparse_full.npz")
pkl_AA_fName = os.path.join(seq_data_path, "pkl_AA_full.npy")
gene_peptide_lengths_fName = os.path.join(seq_data_path, "gene_peptide_lengths.csv")


if not os.path.isfile(pkl_fName):

    print(f"Creating nucleotide matrices and saving to {pkl_fName}")

    make_nucleotide_matrices(drug, 
                             kwargs["tier1_loci"] + kwargs['tier2_loci'],
                             seq_data_path,
                             df_samples,
                             genotype_input_directory,
                             split_groups=False
                            )


# # make peptide lengths dataframe if it has not already been created
# if include_peptide_lengths and not os.path.isfile(gene_peptide_lengths_fName):

#     print("Creating gene peptide lengths dataframe")

#     if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
#         all_loci_seq = create_all_loci_matrices(config_file, fasta_dir=genotype_input_directory, isolates_lst=df_samples['ROLLINGDB_ID'].values)
        
#         pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))

#     gene_peptide_lengths = make_CDS_length_df(drug, kwargs['tier1_loci'] + kwargs['tier2_loci'], genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))
    
#     # keep index because that's the samples column
#     gene_peptide_lengths.to_csv(gene_peptide_lengths_fName)



if include_amino_acid_properties and not os.path.isfile(pkl_AA_fName):

    print(f"Creating amino acid biophysical properties matrix and saving to {pkl_AA_fName}")

    # need to make the full pickle file of all sequences to translate, then get the amino acid properties
    if not os.path.isfile(os.path.join(seq_data_path, "seqDict.pkl")):
        
        all_loci_seq = create_all_loci_matrices(config_file, fasta_dir=genotype_input_directory, isolates_lst=df_samples['ROLLINGDB_ID'].values)
        
        pickle.dump(all_loci_seq, open(os.path.join(seq_data_path, "seqDict.pkl"), "wb"))

    # make protein FASTA files for all loci, both tiers
    create_AA_alns(drug, kwargs["tier1_loci"] + kwargs["tier2_loci"], genotype_input_directory, os.path.join(seq_data_path, "seqDict.pkl"))

    # because the protein left-aln length may be shorter than for the training data, get the longest protein from the training data
    df_protein_seqs = pd.read_csv(f"{training_data_path}/df_protein_seqs.csv", index_col=[0])
    
    L_longest = np.max([df_protein_seqs[col_name].apply(lambda x: len(x)).max() for col_name in df_protein_seqs.columns])

    genes_lst = get_genes_lst(kwargs["tier1_loci"] + kwargs['tier2_loci'])

    make_AA_property_matrices(drug, 
                              genes_lst,
                              seq_data_path, 
                              df_samples, 
                              genotype_input_directory,
                              L_longest=L_longest,
                              split_groups=False
                             )

data_generator = MtbGeneDataset(
        drug,
        df_samples,
        pkl_fName,
        pkl_AA_fName,
        seq_data_path=seq_data_path,
        binary=binary,
        cc=binary_thresh,
        tier1_loci=tier1_loci,
        tier2_loci=tier2_loci,
        no_lineage_SNPs=no_lineage_SNPs,
        include_lineage=include_lineage,
        include_amino_acid_properties=include_amino_acid_properties, 
        bounded_loss=bounded_loss,
        shuffle_batches=False, # don't need to shuffle validation data because order doesn't matter,
    )

num_loci = len(data_generator.nuc_locus_list)
longest_locus = data_generator.longest_locus
longest_protein = data_generator.longest_protein
num_proteins = data_generator.num_proteins
num_peptide_lengths = data_generator.num_peptide_lengths

# since we set bounded_loss = False, the last element in the inputs list is the MLP vector
if include_lineage:
    additional_data_len = data_generator[0][0][-1].shape[1]
else:
    additional_data_len = 0

# initialize a new model and load the weights of the best model
if include_amino_acid_properties:
    best_model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
else:
    best_model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)
        
# best model is one directory up. For saturation mutagenesis, two directories up
if get_predictions:

    print(f"output path: {output_path}")
    
    if saturation_muts:
        model_weights_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(output_path))), "best_model.h5")
    elif insilico_muts:
        model_weights_file = os.path.join(os.path.dirname(os.path.dirname(output_path)), "best_model.h5")
    else:
        model_weights_file = os.path.join(os.path.dirname(output_path), "best_model.h5")

    print(f"\nLoading model {model_weights_file}")
    
    best_model.load_weights(model_weights_file)
    
    y_pred = best_model.predict(
        x=data_generator,
        workers=4,
        use_multiprocessing=True,
    )
    
    df_samples["log2_pred_MIC"] = y_pred
    df_samples["pred_MIC"] = np.exp2(y_pred)

    df_samples['ROLLINGDB_ID'] = df_samples['ROLLINGDB_ID'].str.replace('_p_', '_p.').str.replace('_c_', '_c.').str.replace('+', '*')
    df_samples.to_csv(os.path.join(output_path, "test_predictions.csv"), index=False)
    
    # get lineage predictions if the model has lineage in it. Predict MIC for H37Rv with a single lineage SNP
    if insilico_muts and include_lineage:
    
        # put Coll 2014 dataframe in same order as the lineages matrix to get correct order of lineage predictions
        coll_2014 = coll_2014.set_index('position').loc[lineages_matrix.columns.astype(int)].reset_index().rename(columns={'index': 'position'})
    
        # prediction for each lineage. Also add MT_H37Rv (no SNP)
        df_lineage_pred = pd.DataFrame({'Lineage': list(coll_2014['lineage'].values) + ['MT_H37Rv'], 
                                        'Position': list(coll_2014['position'].values) + [0]
                                       })
    
        # easier to do this without data generators because otherwise need to save files of H37Rv repeated 62 times, which is inefficient
        # this keeps the lineage SNPs in the same order that was used for training (because coll_2014 is what was used to get training lineage SNPs)
        lineages = coll_2014.copy()
        lineages["Count"] = 1
        lineages = lineages.pivot(index="lineage", columns="position", values="Count").fillna(0).astype(int)
    
        assert np.min(lineages.sum(axis=1).values) == 1
        assert np.max(lineages.sum(axis=1).values) == 1
    
        # add MT_H37Rv (no SNP) to get that prediction
        lineages.loc['MT_H37Rv', :] = 0
    
        # check ordering
        assert sum(df_lineage_pred.Position.values[:-1] != lineages_matrix.columns.astype(int)) == 0
        assert sum(lineages.columns != lineages_matrix.columns.astype(int)) == 0
                          
        # get longest locus from the pickle file. The first dimension is the isolates, and H37Rv is the last one
        X_H37Rv_nuc = sparse.load_npz(os.path.join(seq_data_path, 'pkl_sparse_full.npz'))[[-1], :].todense()
    
        # keep only the number of loci specified, which is the last dimension
        X_H37Rv_nuc = X_H37Rv_nuc[:, :, :, :num_loci]
    
        # copy so that there is one reference sequence for each lineage
        X_H37Rv_nuc = np.repeat(X_H37Rv_nuc, lineages.shape[0], axis=0)
        inputs_lst = [X_H37Rv_nuc]
    
        if include_amino_acid_properties:
    
            # the last one is H37Rv
            X_ref = np.load(os.path.join(seq_data_path, "pkl_AA_full.npy"))[[-1], :]
            
            X_ref = np.repeat(X_ref, lineages.shape[0], axis=0)

            # scale across the sample axis (0) and the length of the amino acid sequence (2). Don't scale different biophysical properties together (1), or different genes together (3)
            # load in mean and std of the training data
            train_mean = np.load(os.path.join(seq_data_path, "AA_train_mean.npy"))
            train_std = np.load(os.path.join(seq_data_path, "AA_train_std.npy"))

            # train_mean and train_std are only 2 dimensions. So need to duplicate the arrays to make the full dataset and protein sequence lengths
            train_mean = expand_dims_for_rescaling(train_mean, (0, 2), X_ref)
            train_std = expand_dims_for_rescaling(train_std, (0, 2), X_ref)
            
            # scale
            X_ref = (X_ref - train_mean) / train_std

            # keep only the number of genes specified, which is the last dimension
            X_ref = X_ref[:, :, :, :num_proteins]
            
            inputs_lst.append(X_ref)

        # add the lineage SNPs
        inputs_lst.append(lineages.values)
    
        # # Don't need the actual lengths because we normalized to get effective lengths
        # if include_peptide_lengths:

        #     # get reference peptide lengths
        #     gene_peptide_lengths = pd.read_csv(gene_peptide_lengths_fName, index_col=[0])
    
        #     # all 1s (because effective length is relative to H37Rv) of the same shape as the data
        #     # H37Rv_peptide_lengths = np.ones((lineages.shape[0], num_peptide_lengths))
        #     H37Rv_peptide_lengths = np.repeat(gene_peptide_lengths.loc[['MT_H37Rv'], :].values[:, :num_peptide_lengths], lineages.shape[0], axis=0)
        
        #     mlp_input = np.concatenate([lineages.values,
        #                                 H37Rv_peptide_lengths
        #                                ], axis=1
        #                               ).astype(float)
            
        #     inputs_lst.append(mlp_input)
    
        # else:
        #     inputs_lst.append(lineages.values)
        
        df_lineage_pred["log2_pred_MIC"] = best_model.predict(inputs_lst, batch_size=BATCH_SIZE).flatten()
        df_lineage_pred["pred_MIC"] = np.exp2(df_lineage_pred['log2_pred_MIC'])

        # save one directory up because it's not locus-dependent
        df_lineage_pred.to_csv(os.path.join(os.path.dirname(output_path), "lineage_SNP_predictions.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
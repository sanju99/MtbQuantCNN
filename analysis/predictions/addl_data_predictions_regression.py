import pandas as pd
import numpy as np
import glob, os, yaml, sparse, itertools, subprocess, sys, argparse, pickle, tracemalloc
from Bio import SeqIO, Seq

import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# load all utils functions
sys.path.append("utils")
from data_utils import *
from analysis_utils import *
from model_utils import *
from dataloader import MtbGeneDataset

results_path = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results" # change
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs" # change

data_utils_dir = "./data_processing/data_utils"

coll_2014 = pd.read_csv(f"{data_utils_dir}/coll2014_SNP_scheme.tsv", sep="\t")
coll_2014["lineage"] = coll_2014["#lineage"].str.replace("lineage", "")
del coll_2014["#lineage"]

lineages_matrix = pd.read_csv(f"{data_utils_dir}/lineage_matrix_Coll2014.csv", index_col=[0])
drug_loci = pd.read_csv(f"{data_utils_dir}/drug_loci.csv")

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

# use the model trained on variants with presence at an AF threshold different from the default
parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--permutation', action='store_true', help='If specified, get predictions using the permuted models')

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
AF_thresh = cmd_line_args.AF_thresh
permutation = cmd_line_args.permutation
get_predictions = cmd_line_args.predict

# use the non-75% AF thresh for the test data generator if specified
if AF_thresh > 1:
    AF_thresh /= 100

count_flags_true = 0
flags_lst = np.array([TRUST_data, insilico_muts, saturation_muts])

if sum(flags_lst) > 1:
    raise ValueError("Please only specify only one of optional arguments: TRUST-data, insilico-muts, isolates-span-cc, and saturation-muts")

if sum(flags_lst) == 0:
    raise ValueError("Please specify one of optional arguments: TRUST-data, insilico-muts, isolates-span-cc, and saturation-muts")
        
kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
filter_size = kwargs["filter_size"]
phenotype_file = kwargs["phenotype_file"]
binary_thresh = kwargs["binary_thresh"]

locus_list = kwargs['tier1_loci']

if include_tier2:
    locus_list += kwargs['tier2_loci']

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

if AF_thresh != 0.75:
    output_path += f"_AF{int(AF_thresh * 100)}"

output_path = os.path.join(output_path, "ridge")

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

# train_seq_data_path is already present because it contains the training data. Need to make the other two though
train_seq_data_path = seq_data_path
seq_data_path =  os.path.join(seq_data_path, subdir)
output_path = os.path.join(output_path, subdir)
print(f"Saving prediction results to {output_path}")

if not os.path.isdir(output_path):
    os.makedirs(output_path)

if not os.path.isdir(seq_data_path):
    os.makedirs(seq_data_path)

# get the lists of samples to get MIC predictions for
if TRUST_data:
    # df_samples = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/20240124_MIC_pass_genoQC.csv")

    df_samples = pd.DataFrame([os.path.basename(val).split(".")[0] for val in pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/vcf_full_paths.txt", sep='\t', header=None)[0].values])

    # rename the column to ROLLINGDB_ID for consistency with the TRUST dataframe
    df_samples.columns = ['ROLLINGDB_ID']

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

# get the input matrices using the helper function. This function will do the rescaling for the amino acid properties if needed
# rescaline is necessary using the mean and standard deviation of the training set because the values vary in magnitude considerably

if TRUST_data:
    lineage_SNPs_zero = False
    samples_lst = df_samples['ROLLINGDB_ID'].values
else:
    lineage_SNPs_zero = True
    samples_lst = None

X_for_prediction = get_all_regression_inputs_for_addl_predictions(seq_data_path, 
                                                                  train_seq_data_path,
                                                                  locus_list, # only the loci indicated, not both tiers 
                                                                  lineage_SNPs_zero=lineage_SNPs_zero, 
                                                                  samples_lst=samples_lst, 
                                                                  include_lineage=include_lineage, 
                                                                  include_amino_acid_properties=include_amino_acid_properties
                                                                 )



def get_all_sublineages_from_single_lineage(lineage_str):

    lineage_levels = lineage_str.split('.')

    all_lineages = []

    for i, _ in enumerate(lineage_levels):

        all_lineages.append('.'.join(lineage_levels[:i+1]))

    return all_lineages



def get_predictions_single_model(model_weights_file, df_samples, output_path, coll_2014, fName_suffix='', insilico_muts=True, include_lineage=False, include_amino_acid_properties=False, include_tier2=False):

    locus_list = kwargs['tier1_loci']

    if include_tier2:
        locus_list += kwargs['tier2_loci']

    genes_list = get_genes_lst(locus_list)
    
    # save the model in case it's needed for later (i.e. TRUST predictions or something)
    model = pickle.load(open(model_weights_file, "rb"))

    # get predictions on the test set, including the isolates that span the CC
    y_pred = np.squeeze(model.predict(X_for_prediction))
    
    df_samples["log2_pred_MIC"] = y_pred
    df_samples["pred_MIC"] = np.exp2(y_pred)

    df_samples['ROLLINGDB_ID'] = df_samples['ROLLINGDB_ID'].str.replace('_p_', '_p.').str.replace('_c_', '_c.').str.replace('+', '*')
    df_samples.to_csv(os.path.join(output_path, f"test_predictions{fName_suffix}.csv"), index=False)

    # get lineage predictions if the model has lineage in it. Predict MIC for H37Rv with a single lineage SNP
    if insilico_muts and include_lineage:
                
        # put Coll 2014 dataframe in same order as the lineages matrix to get correct order of lineage predictions
        coll_2014 = coll_2014.set_index('position').loc[lineages_matrix.columns.astype(int)].reset_index().rename(columns={'index': 'position'})
        
        # remove the asterisks so that they are found to be components of other lineages
        coll_2014.loc[coll_2014['lineage'].str.contains('\*'), 'lineage'] = coll_2014['lineage'].str.replace('*', '')
        
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
        
        # check ordering
        assert sum(df_lineage_pred.Position.values[:-1] != lineages_matrix.columns.astype(int)) == 0
        assert sum(lineages.columns != lineages_matrix.columns.astype(int)) == 0

        for lineage_str in lineages.index.values:
            
            # get all the sublineages
            all_sublineages = get_all_sublineages_from_single_lineage(lineage_str)
                
            # make all the position values 1 for SNP present
            lineages.loc[lineage_str, coll_2014.query("lineage in @all_sublineages").position.values] = 1

        # add MT_H37Rv (no SNP) to get that prediction
        lineages.loc['MT_H37Rv', :] = 0

        # get longest locus from the pickle file. The first dimension is the isolates, and H37Rv is the last one
        X_H37Rv_nuc = sparse.load_npz(os.path.join(seq_data_path, 'pkl_sparse_full.npz'))[[-1], :].todense()
        
        # copy so that there is one reference sequence for each lineage
        X_H37Rv_nuc = np.repeat(X_H37Rv_nuc, lineages.shape[0], axis=0)

        # flatten to two-dimensional format, then keep only the features used for training
        X_H37Rv_nuc = get_single_matrix_regression_input(X_H37Rv_nuc, keep_idx=None, num_keep_channels=len(locus_list))

        # keep_features is a list of indices of columns to keep. Need to save it to get those features only when getting additional model predictions
        # do this separately for tier 1 vs. tier 1 + tier 2 because the get_single_matrix_regression_input function does the tier splitting, and it's run before determining unique values
        if include_tier2:
            keep_NT_features = np.load(f"{train_seq_data_path}/regression_train_NT_features_idx_tier2.npy")
        else:
            keep_NT_features = np.load(f"{train_seq_data_path}/regression_train_NT_features_idx.npy")

        # concatenate lineage SNPs with the NT inputs. The order of inputs is NT, lineage SNPs, then AA features
        X_input_for_lineage_pred = np.concatenate([X_H37Rv_nuc[:, keep_NT_features], lineages], axis=1)
    
        if include_amino_acid_properties:
    
            # the last one is H37Rv
            X_H37Rv_AA = np.load(os.path.join(seq_data_path, "pkl_AA_full.npy"))[[-1], :]

            # duplicate the sequence, one for each lineage SNP
            X_H37Rv_AA = np.repeat(X_H37Rv_AA, lineages.shape[0], axis=0)

            # scale across the sample axis (0) and the length of the amino acid sequence (2). Don't scale different biophysical properties together (1), or different genes together (3)
            # load in mean and std of the training data
            train_mean = np.load(os.path.join(train_seq_data_path, "AA_train_mean.npy"))
            train_std = np.load(os.path.join(train_seq_data_path, "AA_train_std.npy"))

            # train_mean and train_std are only 2 dimensions. So need to duplicate the arrays to make the full dataset and protein sequence lengths
            X_H37Rv_AA = (X_H37Rv_AA - expand_dims_for_rescaling(train_mean, (0, 2), X_H37Rv_AA)) / expand_dims_for_rescaling(train_std, (0, 2), X_H37Rv_AA)

            # flatten to a two dimensional matrix
            X_H37Rv_AA = get_single_matrix_regression_input(X_H37Rv_AA, keep_idx=None, num_keep_channels=len(genes_list))
    
            # keep only the features used for training the models
            if include_tier2:
                keep_AA_features = np.load(f"{train_seq_data_path}/regression_train_AA_features_idx_tier2.npy")
            else:
                keep_AA_features = np.load(f"{train_seq_data_path}/regression_train_AA_features_idx.npy")
    
            # concatenate nucleotide, lineage, and AA inputs
            X_input_for_lineage_pred = np.concatenate([X_input_for_lineage_pred, X_H37Rv_AA[:, keep_AA_features]], axis=1)

        df_lineage_pred["log2_pred_MIC"] = np.squeeze(model.predict(X_input_for_lineage_pred))
        df_lineage_pred["pred_MIC"] = np.exp2(df_lineage_pred['log2_pred_MIC'])

        # save one directory up because it's not locus-dependent
        df_lineage_pred.to_csv(os.path.join(os.path.dirname(output_path), f"lineage_SNP_predictions{fName_suffix}.csv"), index=False)


# best model is one directory up. For saturation mutagenesis, two directories up
if get_predictions:

    print(f"output path: {output_path}")
    
    if saturation_muts:
        model_weights_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(output_path)))), "ridge", "best_model.sav")
    elif insilico_muts:
        model_weights_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(output_path))), "ridge", "best_model.sav")
    else:
        model_weights_file = os.path.join(os.path.dirname(os.path.dirname(output_path)), "ridge", "best_model.sav")

    # if permutation, there are 10 models to get with glob
    if permutation:
        print(os.path.dirname(model_weights_file))
        model_weights_files = glob.glob(os.path.join(os.path.dirname(model_weights_file), "saliency", "permutation_test", "permutation_*.h5"))
        del model_weights_file
        print(f"Getting predictions for {len(model_weights_files)} models in {os.path.dirname(model_weights_files[0])}")

        for fName in model_weights_files:

            fName_suffix = os.path.basename(fName).split('permutation')[-1].split('.h5')[0]
            
            get_predictions_single_model(model_weights_file, 
                                         df_samples, 
                                         output_path, 
                                         coll_2014,
                                         fName_suffix=fName_suffix, 
                                         insilico_muts=insilico_muts, 
                                         include_lineage=include_lineage, 
                                         include_amino_acid_properties=include_amino_acid_properties,
                                         include_tier2=include_tier2
                                        )
    else:
        print(f"\nLoading model {model_weights_file}")

        get_predictions_single_model(model_weights_file, 
                                     df_samples, 
                                     output_path, 
                                     coll_2014,
                                     fName_suffix='', 
                                     insilico_muts=insilico_muts, 
                                     include_lineage=include_lineage, 
                                     include_amino_acid_properties=include_amino_acid_properties,
                                     include_tier2=include_tier2
                                    )

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
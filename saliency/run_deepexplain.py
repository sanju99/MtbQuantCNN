import argparse, sparse, sys, os, glob, yaml, tracemalloc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from deepexplain.tensorflow import DeepExplain

# utils files are in the model folder
sys.path.append("utils")
from model_utils import *
from inSilicoMut_utils import *
from dataloader import MtbGeneDataset

# This was suggested in a pull request in the DeepExplain repo to make this compatible with TF v2 models.
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# don't use GPU for this script because it's not training intensive
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

# boolean argument for including lineage SNPs, default value False. If you include the flag, it is considered True
parser.add_argument('--lineage', action='store_true', help='Flag to add lineage SNPs to model')

# boolean argument for including peptide lengths, default value False. If you include the flag, it is considered True
parser.add_argument('--peptide-lengths', dest='peptide_lengths', action='store_true', help='Flag to add peptide lengths to model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--tier2', action='store_true', help='Flag to add tier 2 loci to the model')

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--amino-acid', dest='amino_acid', action='store_true', help='Flag to add amino acid biophysical properties to the model')

# boolean argument for whether the model is a binary or quantitative CNN (the results are stored in different directories)
parser.add_argument('--binary', action='store_true', help='If provided, assumes the model is a binary CNN. If not, quantitative')

# boolean argument for whether or not to compute saliency scores for the permutated models
parser.add_argument('--permutation', action='store_true', help='If provided, compute saliency scores for permutation models')

parser.add_argument('--AF-thresh', dest='AF_thresh', default=0.75, type=float, help='Allele fraction threshold. Default = 0.75')

parser.add_argument('--augment', dest='augment', action='store_true', help='If True, use the {drug}_augment directory')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
include_lineage = cmd_line_args.lineage
include_peptide_lengths = cmd_line_args.peptide_lengths
include_tier2 = cmd_line_args.tier2
include_amino_acid_properties = cmd_line_args.amino_acid
binary = cmd_line_args.binary
compute_permuted_models = cmd_line_args.permutation
AF_thresh = cmd_line_args.AF_thresh
augment = cmd_line_args.augment

# use the non-75% AF thresh for the test data generator if specified
if AF_thresh > 1:
    AF_thresh /= 100
    
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
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
bounded_loss = False # for all models, set bounded_loss = False so that the bounds are not returned

output_path = f"/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/{drug}"

if augment:
    output_path += '_augment'
    
if drug == 'PZA':
    patience_epochs = 150

df_phenos = pd.read_csv(phenotype_file)
df_train_val = df_phenos.query("category in ['train_set', 'validation_set']").reset_index(drop=True)

seq_data_path = output_path

if include_peptide_lengths:
    output_path += "_peptide"
    
if include_lineage:
    output_path += "_lineage"

if include_tier2:
    output_path += "_tier2"

if include_amino_acid_properties:
    output_path += "_amino_acid"

if AF_thresh != 0.75:

    # separate input paths for different AF
    seq_data_path = f"{seq_data_path}_AF{int(AF_thresh*100)}"

    output_path = f"{output_path}_AF{int(AF_thresh*100)}"

    genotype_input_directory = f"{genotype_input_directory.replace('fastas', 'AF_thresh_25/fastas')}"

# the bounds are not necessary for this script, so it's easier to just omit them instead of putting dummy variables into ref_data
# use train + validation data for computing saliency scores
data_generator = MtbGeneDataset(
    drug,
    df_train_val,
    os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'),
    os.path.join(seq_data_path, 'pkl_AA_train_val.npy'),
    seq_data_path=seq_data_path,
    binary=binary,
    cc=binary_thresh,
    tier1_loci=tier1_loci,
    tier2_loci=tier2_loci,
    data_subset='train_set',
    include_lineage=include_lineage,
    include_peptide_lengths=include_peptide_lengths,
    include_amino_acid_properties=include_amino_acid_properties,
    bounded_loss=bounded_loss, # don't need bounded loss for this, so don't load the bounds in
    batch_size=BATCH_SIZE*2, # increase the batch size so that it runs faster. Memory requirements should be lower because no models are being trained here
    shuffle_batches=False, # don't need to shuffle test data because order doesn't matter
)

num_loci = len(data_generator.nuc_locus_list)
longest_locus = data_generator.longest_locus
longest_protein = data_generator.longest_protein
num_proteins = data_generator.num_proteins
num_peptide_lengths = data_generator.num_peptide_lengths

if include_lineage or include_peptide_lengths:
    additional_data_len = data_generator[0][0][-1].shape[1]
else:
    additional_data_len = 0

lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
num_lineages = lineages.shape[1]

# get reference data for saliency computation
X_ref_nuc = sparse.load_npz(os.path.join(seq_data_path, 'pkl_sparse_ref.npz')).todense()

# keep only the number of loci specified, which is the last dimension
X_ref_nuc = X_ref_nuc[:, :, :, :num_loci]

ref_data = [X_ref_nuc]

# gene_peptide_lengths = pd.read_csv(os.path.join(seq_data_path, "gene_peptide_lengths.csv"), index_col=[0])

if include_amino_acid_properties:

    # the last one is H37Rv
    X_ref_AA = np.load(os.path.join(seq_data_path, "pkl_AA_ref.npy"))[[-1], :]

    # keep only the number of genes specified, which is the last dimension
    X_ref_AA = X_ref_AA[:, :, :, :num_proteins]

    ref_data.append(X_ref_AA)

# additional inputs to concatenate before the dense layers
if include_lineage:
    
    # no lineage SNPs in H37Rv, must be of shape 1 x num_lineages
    ref_lineages = np.zeros((1, lineages.shape[1]))
    
    if include_peptide_lengths:

        # both
        ref_ref_mlp_input = np.concatenate([ref_lineages,
                                            gene_peptide_lengths.loc[['MT_H37Rv'], :].values[:, :num_peptide_lengths]
                                           ], axis=1
                                          ).astype(float)
    else:
        # lineage only
        ref_mlp_input = ref_lineages
else:
    # peptide lengths only
    if include_peptide_lengths:
        ref_mlp_input = gene_peptide_lengths.loc[['MT_H37Rv'], :].values[:, :num_peptide_lengths]
    else:
        ref_mlp_input = []

if len(ref_mlp_input) > 0:
    ref_data.append(ref_mlp_input)

# this is so that when checking that the baseline and test data formats match, it will be an array 
if len(ref_data) == 1:
    ref_data = ref_data[0]

# losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))

# # get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
# losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
# select_alpha = losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0]
# print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")

def get_saliency_scores(weights_path, data_generator, ref_data, saliency_dir, file_suffix=""):
    
    genetic_attr_by_nuc = []
    genetic_attr = []
    AA_property_attr_by_pos = []
    AA_property_attr = []
    mlp_attr = []

    # set bounded loss to False so that you don't have to specify lower and upper bounds. It only affects training, and this model is not being trained
    if include_amino_acid_properties:
        model = multi_conv_nn(binary, longest_locus, num_loci, longest_protein, num_proteins, additional_data_len, bounded_loss, filter_size, reg_strength=0)
    else:
        model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=0)

    model.load_weights(weights_path)
    
    with DeepExplain(session=tf.compat.v1.keras.backend.get_session()) as de:

        # initialize a DeepExplain model using the same inputs and outputs as the original model
        # For quantitative models, we want to compute saliencies for the output layer. For binary, get saliencies for the output of the penultimate layer (before the last layer, which applies Softmax)
        if binary:
            de_model = tf.keras.Model(inputs = model.inputs, outputs = model.layers[-2].output)
            predict_model = tf.keras.Model(inputs = model.inputs, outputs = model.outputs)
        else:
            de_model = tf.keras.Model(inputs = model.inputs, outputs = model.outputs)

        # this is the layer for which we want to compute saliency scores
        target_layer = de_model(model.inputs)

        # check that the original and DeepExplain models give the same predictions on the H37Rv input
        if binary:
            assert np.abs(np.max(predict_model.predict(ref_data)-model.predict(ref_data))) < 1e-5
            del predict_model
        else:
            assert np.abs(np.max(de_model.predict(ref_data)-model.predict(ref_data))) < 1e-5

        for idx, batch in enumerate(data_generator):

            print(f"Working on batch {idx+1} of {len(data_generator)}")

            # the first index is the sequence data, the second index is the phenotypes
            X_train = batch[0]

            # when there are no additional inputs (only sequence inputs), an extra dimension gets added, so it's (1, 128, 5, longest_locus, num_loci)
            # check that the length of the batch is 1 for that case
            # the data generator class always puts the inputs in a list, even if the length of the list is 1
            if len(X_train) == 1:
                X_train = np.squeeze(X_train[0])
        
            # compute attributions for the training set
            attributions = de.explain(method='deeplift', 
                                      T=target_layer, # target tensor to get attributions for
                                      X=model.inputs, # symbolic input to the network
                                      xs=X_train, 
                                      baseline=ref_data, # if method_name == deeplift, then provide the reference data as the baseline
                                     )


            # create an array of the indices to ignore. These are the indices of the reference nucleotide at each position
            nt_idx_to_ignore = np.argmax(np.squeeze(X_ref_nuc), axis=0)
                
            # genetic scores shape should be num_samples x 5 x longest_locus x num_loci -- sum scores across nucleotides, which is the second dimension
            # this means that there are multiple inputs (not just NT), so there are separate scores for the convolutional block and the flattened MLP input
            if type(ref_data) == list:
                
                # NT scores matrix is always first. Keep the full array of scores here, disaggregated by nucleotide
                genetic_attr_by_nuc.append(attributions[0])
                                                
                for pos, nuc_idx in enumerate(nt_idx_to_ignore):
                    # set the index of the reference nucleotide to 0
                    # samples x 5 x position x 1
                    # when the scores are summed across the nucleotides in the next line, the ref nucleotide doesn't contribute
                    attributions[0][:, nuc_idx, pos, :] = 0

                # then after setting the reference nucleotides to 0, keep the sum across the position axis
                genetic_attr.append(np.sum(attributions[0], axis=1))

                if include_amino_acid_properties:

                    # different from NT case in that we don't need to set the reference as 0 because there is no categorical reference
                    # the second dimension of the input matrix is the 3 different biophysical properties, so you will get 3 saliency scores at each amino acid site
                    # then it doesn't make sense to change any of them like with the NT scores, so keep them as is
                    # take the sum of saliency scores across the property dimension to get a single score for the position
                    AA_property_attr_by_pos.append(attributions[1])        
                    AA_property_attr.append(np.sum(attributions[1], axis=1))

                # if amino acid features are included, then it will be between NT and MLP features. MLP should always be last though
                if additional_data_len > 0:
                    mlp_attr.append(attributions[-1])

            # NT sequence inputs only
            else:
                # full scores matrix
                genetic_attr_by_nuc.append(attributions)
                
                for pos, nuc_idx in enumerate(nt_idx_to_ignore):
                    # set the index to ignore to 0
                    # samples x 5 x position x 1
                    attributions[:, nuc_idx, pos, :] = 0
    
                genetic_attr.append(np.sum(attributions, axis=1))
        
        # combine scores for all isolates along the first axis, which is the number of samples axis
        genetic_attr_by_nuc = np.concatenate(genetic_attr_by_nuc, axis=0)
        genetic_attr = np.concatenate(genetic_attr, axis=0) 

        print(f"Saving scores to {saliency_dir}")

        # don't save scores by nucleotide for the permutation test
        if file_suffix == "":
            sparse.save_npz(os.path.join(saliency_dir, f"genetic_scores_unpooled_nuc{file_suffix}.npy"), sparse.COO(genetic_attr_by_nuc), compressed=True)
            sparse.save_npz(os.path.join(saliency_dir, f"genetic_scores{file_suffix}.npy"), sparse.COO(genetic_attr), compressed=True)

        # save mean, max, and min scores
        np.save(os.path.join(saliency_dir, f"scores_max{file_suffix}.npy"), np.max(genetic_attr, axis=0))
        np.save(os.path.join(saliency_dir, f"scores_min{file_suffix}.npy"), np.min(genetic_attr, axis=0))
        
        if include_amino_acid_properties:

            # combine scores for all isolates along the first axis, which is the number of samples axis
            AA_property_attr_by_pos = np.concatenate(AA_property_attr_by_pos, axis=0)
            AA_property_attr = np.concatenate(AA_property_attr, axis=0) 
            
            # don't save individual genetic scores (or scores by nucleotide) for the permutation test
            if file_suffix == "":
                sparse.save_npz(os.path.join(saliency_dir, f"AA_property_scores_unpooled{file_suffix}.npy"), sparse.COO(AA_property_attr_by_pos), compressed=True)
                sparse.save_npz(os.path.join(saliency_dir, f"AA_property_scores{file_suffix}.npy"), sparse.COO(AA_property_attr), compressed=True)

            # save mean, max, and min scores
            np.save(os.path.join(saliency_dir, f"AA_scores_max{file_suffix}.npy"), np.max(AA_property_attr, axis=0))
            np.save(os.path.join(saliency_dir, f"AA_scores_min{file_suffix}.npy"), np.min(AA_property_attr, axis=0))
        
        if additional_data_len > 0:
            mlp_attr = np.concatenate(mlp_attr, axis=0)
            print(mlp_attr.shape)
            np.save(os.path.join(saliency_dir, f"mlp_scores{file_suffix}.npy"), mlp_attr)

    K.clear_session()


# get saliency results for the full model
saliency_dir = os.path.join(output_path, "saliency")
print(f"Saving results to {saliency_dir}")

if not os.path.isdir(saliency_dir):
    os.makedirs(os.path.join(saliency_dir))

# get saliency scores for all the permuted models
if compute_permuted_models:

    # this path should already exist because that's where the models are stored
    saliency_dir = os.path.join(output_path, "saliency", "permutation_test")
    print(f"Saving results to {saliency_dir}")
    
    weights_lst = glob.glob(os.path.join(saliency_dir, "*.h5"))
        
    for i, weights_path in enumerate(weights_lst):

        # make sure the name suffix for the saliency scores matches the permutation model
        print(f"Computing saliency scores for model {i+1} out of {len(weights_lst)}")
        model_num = os.path.basename(weights_path).split(".")[0].split("_")[-1]
        
        # # check if scores have already been computed for the model
        # if include_lineage or include_peptide_lengths:
        #     check_file = os.path.join(saliency_dir, f"mlp_scores_{model_num}.npy")
        # else:
        #     check_file = os.path.join(saliency_dir, f"scores_max_{model_num}.npy")
            
        # if not os.path.isfile(check_file):
        #     # compute saliency scores for a single permuted model
        #     get_saliency_scores(weights_path, data_generator, ref_data, saliency_dir, file_suffix=f"_{model_num}")

        get_saliency_scores(weights_path, data_generator, ref_data, saliency_dir, file_suffix=f"_{model_num}")

# get saliency scores for the primary model
else:
    get_saliency_scores(os.path.join(output_path, "best_model.h5"), data_generator, ref_data, saliency_dir, file_suffix="")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB\n")
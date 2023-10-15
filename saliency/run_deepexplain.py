import sparse, sys, os, glob, yaml, tracemalloc
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
from data_utils import *
from dataloader import MtbGeneDataset

# This was suggested in a pull request in the DeepExplain repo to make this compatible with TF v2 models.
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# don't use GPU for this script because it's not training intensive
os.environ["CUDA_VISIBLE_DEVICES"] = ""


_, config_file, permutation_test = sys.argv

tracemalloc.start()

permutation_test = [True if permutation_test == "True" else False][0]

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
locus_list = kwargs["locus_list"]
filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
N_epochs = kwargs["N_epochs"]
patience_epochs = kwargs["patience_epochs"]

output_path = kwargs["output_path"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary = kwargs["binary"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]
include_peptide_length = kwargs["include_peptide_length"]

num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path.replace("_lineage", ""), 'pkl_sparse_ref.npz')).todense()

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
print(f"Longest locus: {longest_locus}")

# for all models, set bounded_loss = False so that the bounds are not returned
# the bounds are not necessary for this script, so it's easier to just omit them instead of putting dummy variables into ref_data
# use all the data (train + test) for computing saliency scores. Test data won't be reflected in the trained model, but an allele different from reference may have an influence
data_generator = MtbGeneDataset(
    os.path.join(output_path.replace("_lineage", ""), 'pkl_sparse_full.npz'),
    phenotype_file,
    drug,
    locus_list,
    fasta_dir=genotype_input_directory,
    binary=binary,
    cc=binary_thresh,
    train_or_test=None,
    shuffle_phenos=False,
    include_lineage=include_lineage,
    include_peptide_length=include_peptide_length,
    bounded_loss=False,
    data_idx=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)

lineages = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
num_lineages = lineages.shape[1]
del lineages

if include_peptide_length:
    peptide_lengths_df = make_H37Rv_CDS_length_df(locus_list, genotype_input_directory)

    # ensure same order as locus_list because those are the lengths that the model was trained on
    H37Rv_peptide_lengths = pd.DataFrame(peptide_lengths_df.groupby("Locus")["Length"].sum()).loc[locus_list].T.reset_index(drop=True)

    if include_lineage:
        # the lineage SNP schemes all use H37Rv as the reference, so it's easy because the H37Rv SNPs are all 0
        ref_mlp_data = np.concatenate([np.zeros((1, num_lineages)), H37Rv_peptide_lengths.values], axis=1)
    else:
        ref_mlp_data = H37Rv_peptide_lengths.values
else:
    if include_lineage:
        ref_mlp_data = np.zeros((1, num_lineages))
    else:
        ref_mlp_data = None

# both features are combined into the same vector, so if at least one of them is True, there is a vector
if include_lineage or include_peptide_length:
    additional_data_len = data_generator[0][0][1].shape[1]
    ref_data = [X_h37rv, ref_mlp_data]
else:
    additional_data_len = 0
    ref_data = X_h37rv


# creat output directories
if binary:
    model_prefix = "binary_"
    save_prefix = "binary"
else:
    model_prefix = ""
    save_prefix = "quant"


losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))

# get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
select_alpha = losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0]
print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")


def get_saliency_scores(weights_path, data_generator, ref_data, saliency_dir, file_suffix=""):
    
    genetic_attr_by_nuc = []
    genetic_attr = []
    mlp_attr = []

    # set bounded loss to False so that you don't have to specify lower and upper bounds. It only affects training, and this model is not being trained
    model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss=False, filter_size=filter_size, reg_strength=select_alpha)
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

            # the second index is the phenotypes
            X_train = batch[0]

            # when there are no additional inputs (only sequence inputs), an extra dimension gets added, so it's (1, 128, 5, longest_locus, num_loci)
            if additional_data_len == 0:
                X_train = np.squeeze(X_train)

            # compute attributions for the training set            
            attributions = de.explain(method='deeplift', 
                                      T=target_layer, # target tensor to get attributions for
                                      X=model.inputs, # symbolic input to the network
                                      xs=X_train, 
                                      baseline=ref_data, # if method_name == deeplift, then provide the reference data as the baseline
                                     )


            # create an array of the indices to ignore. These are the indices of the reference nucleotide at each position
            idx_to_ignore = np.argmax(np.squeeze(X_h37rv), axis=0)
                
            # genetic scores shape should be num_samples x 5 x longest_locus x num_loci -- sum scores across nucleotides, which is the second dimension
            # this means that there is an MLP block, so there are separate scores for the convolutional block and the flattened MLP input
            if include_lineage or include_peptide_length:
                
                # full scores matrix
                genetic_attr_by_nuc.append(attributions[0])
                                                
                for pos, nuc_idx in enumerate(idx_to_ignore):
                    # set the index of the reference nucleotide to 0
                    # samples x 5 x position x 1
                    # when the scores are summed across the nucleotides in the next line, the ref nucleotide doesn't contribute
                    attributions[0][:, nuc_idx, pos, :] = 0
    
                genetic_attr.append(np.sum(attributions[0], axis=1))
                
                mlp_attr.append(attributions[1])
            
            else:
                # full scores matrix
                genetic_attr_by_nuc.append(attributions)
                
                for i, nuc_idx in enumerate(idx_to_ignore):
                    # set the index to ignore to 0
                    # samples x 5 x position x 1
                    attributions[:, nuc_idx, i, :] = 0
    
                genetic_attr.append(np.sum(attributions, axis=1))
        
        # combine scores for all isolates along the first axis, which is the number of samples axis
        genetic_attr_by_nuc = np.concatenate(genetic_attr_by_nuc, axis=0)
        genetic_attr = np.concatenate(genetic_attr, axis=0) 
        print(genetic_attr.shape, genetic_attr_by_nuc.shape)

        print(f"Saving scores to {saliency_dir}")

        # don't save individual genetic scores (or scores by nucleotide) for the permutation test
        if file_suffix == "":
            sparse.save_npz(os.path.join(saliency_dir, f"genetic_scores_unpooled_nuc{file_suffix}.npy"), sparse.COO(genetic_attr_by_nuc), compressed=True)
            sparse.save_npz(os.path.join(saliency_dir, f"genetic_scores{file_suffix}.npy"), sparse.COO(genetic_attr), compressed=True)

        # save mean, max, and min scores
        np.save(os.path.join(saliency_dir, f"scores_max{file_suffix}.npy"), np.max(genetic_attr, axis=0))
        np.save(os.path.join(saliency_dir, f"scores_min{file_suffix}.npy"), np.min(genetic_attr, axis=0))
        # np.save(os.path.join(saliency_dir, f"scores_mean{file_suffix}.npy"), np.mean(genetic_attr, axis=0))

        if include_lineage or include_peptide_length:
            mlp_attr = np.concatenate(mlp_attr, axis=0)
            print(mlp_attr.shape)
            np.save(os.path.join(saliency_dir, f"mlp_scores{file_suffix}.npy"), mlp_attr)

    K.clear_session()



if not permutation_test:

    saliency_dir = os.path.join(output_path, "saliency", save_prefix)
    print(f"Saving results to {saliency_dir}")
    
    if not os.path.isdir(saliency_dir):
        os.makedirs(os.path.join(saliency_dir))
        
    # compute saliency scores for the single model
    get_saliency_scores(os.path.join(output_path, f"{model_prefix}best_model.h5"), data_generator, ref_data, saliency_dir, file_suffix="")

else:
    # this path should already exist because that's where the models are stored
    saliency_dir = os.path.join(output_path, "saliency", save_prefix, "permutation_test")
    print(f"Saving results to {saliency_dir}")
    
    weights_lst = glob.glob(os.path.join(saliency_dir, "*.h5"))
        
    for i, weights_path in enumerate(weights_lst):
                
        print(f"Computing saliency scores for model {i+1} out of {len(weights_lst)}")
        model_num = os.path.basename(weights_path).split(".")[0].split("_")[-1]
        
        # check if scores have already been computed for the model
        if include_lineage or include_peptide_length:
            check_file = os.path.join(saliency_dir, f"mlp_scores_{model_num}.npy")
        else:
            check_file = os.path.join(saliency_dir, f"scores_max_{model_num}.npy")
            
        if not os.path.isfile(check_file):
            # compute saliency scores for a single permuted model
            get_saliency_scores(weights_path, data_generator, ref_data, saliency_dir, file_suffix=f"_{model_num}")
            

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"Maximum memory used: {script_memory} GB")
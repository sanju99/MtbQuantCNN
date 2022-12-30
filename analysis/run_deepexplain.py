import sparse, sys, os, glob, yaml, tracemalloc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

# cnn_utils is one level up in the directory tree
sys.path.append(os.path.dirname(os.getcwd()))
from cnn_utils import *

# disable v2 stuff to make this compatible with TF v2 models. This was suggested in a pull request in DeepExplain
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

_, config_file = sys.argv

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
permutation_test = kwargs["permutation_test"]

num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)
    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz')).todense()

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
print(f"Longest locus: {longest_locus}")

# for all models, set bounded_loss = False so that the bounds are not returned
# the bounds are not necessary for this script, so it's easier to just omit them instead of putting dummy variables into ref_data
train_generator = MtbGeneDataset(
    os.path.join(output_path, 'pkl_sparse_train.npz'),
    phenotype_file,
    drug,
    locus_list,
    train_or_test="original_train_set",
    binary=binary,
    cc=binary_thresh,
    include_lineage=include_lineage,
    bounded_loss=False,
    data_idx=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)
            
if include_lineage:
    num_lineages = train_generator[0][0][1].shape[1]
    
    lineages = pd.get_dummies(df_phenos["Lineage"])
    if "4" not in lineages.columns:
        raise RuntimeError("Lineage 4 is not in the lineage matrix")

    # get the index of L4 and make the one-hot encoding for H37Rv as :L4
    l4_idx = list(lineages).index("4")
    ref_lineages = np.zeros((1, num_lineages))
    ref_lineages[:, l4_idx] = 1
    
    ref_data = [X_h37rv, ref_lineages]
else:
    num_lineages = 0
    ref_data = X_h37rv

# creat output directories
if binary:
    model_prefix = "binary_"
    save_prefix = "binary"
else:
    model_prefix = ""
    save_prefix = "quant"


def get_saliency_scores(model, weights_path, train_generator, ref_data, output_path, file_suffix=None):
    
    genetic_attr = []
    lineage_attr = []
    
    model.load_weights(weights_path)
    
    with DeepExplain(session=K.get_session()) as de:

        # initialize a DeepExplain model using the same inputs and outputs as the original model
        de_model = tf.keras.Model(inputs = model.inputs, outputs = model.outputs)

        # get the target layer to get attributions for. For quantitative models, we want to target the output layer 
        if binary:
            pass
            # TODO: figure out how to target the second to last layer before sigmoid activation(logits)
        else:
            target_layer = de_model(model.inputs)

        # check that the original and DeepExplain models give the same predictions on the H37Rv input
        assert np.abs(np.max(de_model.predict(ref_data)-model.predict(ref_data))) < 1e-5

        for idx, batch in enumerate(train_generator):

            print(f"Working on batch {idx+1} of {len(train_generator)}")

            # the second index is the phenotypes
            X_train = batch[0]

            # Remove the batch dimension, which is the first dimension. If the lengths of the dimensions are the same, then the batch dimension is in the reference
            # for some reason, this needs to be done for every batch. If it is done outside of this loop, ref_data gets another dimension in each input at the end of each batch
            if include_lineage:
                if ref_data[0].shape[0] == 1:
                    ref_data[0] = ref_data[0][0]

                if ref_data[1].shape[0] == 1:
                    ref_data[1] = ref_data[1][0]
            else:
                if ref_data.shape[0] == 1:
                    ref_data = ref_data[0]

            # compute attributions for the training set            
            attributions = de.explain(method='deeplift', 
                                      T=target_layer, # target tensor to get attributions for
                                      X=model.inputs, # symbolic input to the network
                                      xs=X_train, 
                                      baseline=ref_data, # if method_name == deeplift, then provide the reference data as the baseline
                                     )


            # genetic scores shape should be num_samples x 5 x longest_locus x num_loci -- sum scores across nucleotides, which is the second dimension
            if include_lineage:
                genetic_attr.append(np.sum(attributions[0], axis=1))
                lineage_attr.append(attributions[1])
            else:
                genetic_attr.append(np.sum(attributions, axis=1))

        # combine scores for all isolates along the first axis, which is the number of samples axis
        genetic_attr = np.concatenate(genetic_attr, axis=0)    

        print(f"Saving scores to {output_path}")
        sparse.save_npz(os.path.join(output_path, f"genetic_scores{file_suffix}.npy"), sparse.COO(genetic_attr), compressed=True)

        # save mean, max, and min scores
        np.save(os.path.join(output_path, f"scores_max{file_suffix}.npy"), np.max(genetic_attr, axis=0))
        np.save(os.path.join(output_path, f"scores_min{file_suffix}.npy"), np.min(genetic_attr, axis=0))
        np.save(os.path.join(output_path, f"scores_mean{file_suffix}.npy"), np.mean(genetic_attr, axis=0))

        if include_lineage:
            lineage_attr = np.concatenate(lineage_attr, axis=0)
            print(lineage_attr.shape)
            np.save(os.path.join(output_path, f"lineage_scores{file_suffix}.npy"), lineage_attr)


# get model from cnn_utils. Build using TF v1, then load weights
model = conv_nn(binary=binary, longest_locus=longest_locus, num_loci=num_loci, num_lineages=num_lineages, bounded_loss=False, filter_size=filter_size)


if not permutation_test:

    output_path = os.path.join(output_path, "saliency", save_prefix)
    
    if not os.path.isdir(output_path):
        os.makedirs(os.path.join(output_path))
        
    # compute saliency scores for the single model
    get_saliency_scores(model, os.path.join(output_path, f"{model_prefix}best_model.h5"), train_generator, ref_data, output_path, file_suffix=None)

else:
    # this path should already exist because that's where the models are stored
    output_path = os.path.join(output_path, "saliency", save_prefix, "permutation_test")
    
    weights_lst = glob.glob(os.path.join(output_path, "*.h5"))
        
    for i, weights_path in enumerate(weights_lst):
        
        print(f"Computing saliency scores for model {i+1} out of {len(weights_lst)}")
        
        model_num = os.path.basename(weights_path).split(".")[0].split("_")[-1]
        
        # compute saliency scores for the single model
        get_saliency_scores(model, weights_path, train_generator, ref_data, output_path, file_suffix=f"_{model_num}")
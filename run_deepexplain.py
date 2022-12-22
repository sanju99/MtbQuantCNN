import sparse, sys, os, glob, yaml, tracemalloc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain

from cnn_utils import *
import warnings
warnings.filterwarnings("ignore")

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

num_loci = len(locus_list)
df_phenos = pd.read_csv(phenotype_file)
    
# get longest locus from the pickle file
# train_set = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_train.npz')).todense()
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
    shuffle=True
)
            
if include_lineage:
    num_lineages = train_generator[0][0][1].shape[1]
    ref_data = [X_h37rv, np.zeros((1, num_lineages))]
else:
    num_lineages = 0
    ref_data = [X_h37rv]

# creat output directories
if binary:
    model_prefix = "binary_"
    save_prefix = "binary"
else:
    model_prefix = ""
    save_prefix = "quant"
    
    
# get model from cnn_utils. Build using TF v1, then load weights
model = conv_nn(binary=binary, longest_locus=longest_locus, num_loci=num_loci, num_lineages=num_lineages, bounded_loss=False, filter_size=filter_size)
model.load_weights(os.path.join(output_path, f"{model_prefix}best_model.h5"))

# update output path for the saliency folder
output_path = os.path.join(output_path, "saliency", save_prefix)
    
if not os.path.isdir(output_path):
    os.makedirs(os.path.join(output_path))
    
genetic_attr = []
lineage_attr = []
    
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

        if include_lineage:
            X_train = batch[0][:2]
        else:
            X_train = batch[0][0]

        # Remove the batch dimension, which is the first dimension. If the lengths of the dimensions are the same, then the batch dimension is in the reference
        # for some reason, this needs to be done for every batch. If it's done outside of this loop, ref_data gets another dimension in each input at the end of each batch
        if len(ref_data[0].shape) == len(X_train[0].shape):
            ref_data[0] = ref_data[0][0]

        if include_lineage:
            if len(ref_data[1].shape) == len(X_train[1].shape):
                ref_data[1] = ref_data[1][0]

        # compute attributions for the training set            
        attributions = de.explain(method='deeplift', 
                                  T=target_layer, # target tensor to get attributions for
                                  X=model.inputs, # symbolic input to the network
                                  xs=X_train, 
                                  baseline=ref_data, # if method_name == deeplift, then provide the reference data as the baseline
                                 )


        # genetic scores shape should be num_samples x 5 x longest_locus x num_loci -- sum scores across nucleotides, which is the second dimension
        attributions[0] = np.sum(attributions[0], axis=1)
        genetic_attr.append(attributions[0])

        if include_lineage:
            lineage_attr.append(attributions[1])
            
    # combine scores for all isolates along the first axis, which is the number of samples axis
    genetic_attr = np.concatenate(genetic_attr, axis=0)    

    print(f"Saving scores to {output_path}")
    sparse.save_npz(os.path.join(output_path, "genetic_scores.npy"), sparse.COO(genetic_attr), compressed=True)
    
    # save mean, max, and min scores
    np.save(os.path.join(output_path, "scores_max.npy"), np.max(genetic_attr, axis=0))
    np.save(os.path.join(output_path, "scores_min.npy"), np.min(genetic_attr, axis=0))
    np.save(os.path.join(output_path, "scores_mean.npy"), np.mean(genetic_attr, axis=0))
    
    if include_lineage:
        lineage_attr = np.concatenate(lineage_attr, axis=0)
        np.save(os.path.join(output_path, "lineage_scores.npy"), attributions[1])
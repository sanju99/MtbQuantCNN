'''
Converts our tensorflow/keras model to a deeplift model
This script takes a test isolate (the reference), converts our keras model
to a DeepLIFT-compatible version, and checks that the two models have the same
predictions. It does not save the deeplift version, but does save the keras one

Author: Sanjana Kulkarni

Note: This requires tensorflow v1!! The CNN model must be saved in tf1

Based on Google Colab notebook DeepLIFT notebook genomics_simulation.ipynb
'''

import sparse, os, sys, yaml, h5py, json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

import deeplift
import deeplift.conversion.kerasapi_conversion as kc
from collections import OrderedDict

from cnn_utils import *
import warnings
warnings.filterwarnings("ignore")

####### Section 1: Read in reference data for sanity check ####################

_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
BATCH_SIZE = kwargs["batch_size"]
locus_list = kwargs["locus_list"]
filter_size = kwargs["filter_size"]
output_path = kwargs["output_path"]
num_loci = len(locus_list)
phenotype_file = kwargs["phenotype_file"]
binary = kwargs["binary"]
binary_thresh = kwargs["binary_thresh"]
include_lineage = kwargs["include_lineage"]
bounded_loss = kwargs["bounded_loss"]

X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz')).todense()

# shape = 1 x 5 x longest_locus x num_loci
longest_locus = X_h37rv.shape[2]
print(f"Longest locus: {longest_locus}")

####### Section 2: Prepare the model and save in json format ##################
# Convert the keras model to a deeplift model
# Deeplift model saved as a json file to be loaded in the future

# Input/Output Paths 
if binary:
    model_prefix = "binary_"
    save_prefix = "binary"
    preSoftmax_var = True
else:
    model_prefix = ""
    save_prefix = "quant"
    preSoftmax_var = False
    
if not os.path.isdir(os.path.join(output_path, "deeplift_outputs", save_prefix)):
    os.makedirs(os.path.join(output_path, "deeplift_outputs", save_prefix))
    
deeplift_trialweights = os.path.join(output_path, f"{model_prefix}best_model.h5")
print(f"Original model: {deeplift_trialweights}")

if not os.path.isfile(deeplift_trialweights):
    raise ValueError("No model weights found!")
    
deeplift_model_json = os.path.join(output_path, "deeplift_outputs", save_prefix, "model.json")
print(f"Deeplift model: {deeplift_model_json}")


# batch load data and compute saliency scores because inputs are too large, don't shuffle inputs
# also need this to get the number of SNPs to instantiate the correct model
train_generator = MtbGeneDataset(
    os.path.join(output_path, 'pkl_sparse_train.npz'),
    phenotype_file,
    drug,
    locus_list,
    train_or_test="original_train_set",
    binary=binary,
    cc=binary_thresh,
    include_lineage=include_lineage,
    bounded_loss=bounded_loss,
    data_idx=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)

if include_lineage:
    num_lineages = train_generator[0][0][1].shape[1]
else:
    num_lineages = 0

# get model and load weights
if bounded_loss:
    model = custom_loss_quant_CNN(longest_locus, num_loci, num_lineages, filter_size=filter_size)
else:
    model = standard_CNN(longest_locus, num_loci, num_lineages, binary, filter_size, preSoftmax=preSoftmax_var)

print(f"{model.count_params()} parameters in the model")

model.load_weights(deeplift_trialweights)

# Save json
model_json = model.to_json()
with open(deeplift_model_json, "w") as json_file:
    json_file.write(model_json)
    

###### Step 4: Read in Deeplift model and define method (rules) to use for assessment #####
## Load our model from the json file
our_model = model_from_json(open(deeplift_model_json).read())
our_model.load_weights(deeplift_trialweights)

# if not include_lineage:
scoring_method = kc.convert_model_from_saved_files(
    h5_file=deeplift_trialweights,
    json_file=deeplift_model_json,
    nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
)


###### Step 5: sanity check make sure that our predictions match with keras and deeplift#####

if include_lineage:
    
    # get indices of input layers
    layer_names = list(scoring_method.get_name_to_layer().keys())
    input_layers = [name for _, name in enumerate(layer_names) if ("input" in name or "bounds" in name)]
    
    # get the layer names using the indices determine in the previous line. There are only 2 indices
    output_layer = layer_names[-1]
    print(f"Input layer names: {input_layers}")
    print(f"Output layer name: {output_layer}")

    if bounded_loss:
        ref_data = [X_h37rv, np.zeros((1, num_lineages)), np.array([[0]]), np.array([[0.03]])]
    else:
        ref_data = [X_h37rv, np.zeros((1, num_lineages))]
        
    inputs = [scoring_method.get_name_to_layer()[input_layers[i]].get_activation_vars() for i in range(len(ref_data))]
    
    deeplift_prediction_func = deeplift.util.compile_func(inputs=inputs,
                                                          outputs=scoring_method.get_name_to_layer()[output_layer].get_activation_vars()
                                                         )
    
else:
    deeplift_prediction_func = deeplift.util.compile_func(inputs=[scoring_method.get_layers()[0].get_activation_vars()],
                                                          outputs=scoring_method.get_layers()[-1].get_activation_vars()
                                                         )
    if bounded_loss:
        ref_data = [X_h37rv, np.array([[0]]), np.array([[0.03]])]
    else:
        ref_data = [X_h37rv]


original_model_predictions = our_model.predict(ref_data, batch_size=200)

converted_model_predictions = deeplift.util.run_function_in_batches(func=deeplift_prediction_func,
                                                                    input_data_list=ref_data,
                                                                    batch_size=200,
                                                                    progress_update=None
                                                                   )


print(original_model_predictions)
print(converted_model_predictions)
print("maximum difference in predictions:", np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)))
assert np.max(np.abs(np.array(converted_model_predictions)-np.array(original_model_predictions))) < 1e-5

########## Step 6: Compute importances 

first_batch = train_generator[0][0]

if include_lineage:  
    
    scoring_func = scoring_method.get_target_contribs_func(find_scores_layer_name=[input_layers[0], input_layers[1]],
                                                           pre_activation_target_layer_name=output_layer
                                                          )
    
    # each of these has length 128(ish), one for each isolate
    # add the scores for the first batch. first 0 = first batch. second 0 = inputs ([1] = output MICs)
    print(f"Working on batch 1 of {len(train_generator)}")
    
    combined_genetic_scores, combined_lineage_scores = np.array(scoring_func(task_idx=0,
                                                                            input_data_list=first_batch,
                                                                            input_references_list=ref_data,
                                                                            batch_size=10,
                                                                            progress_update=None
                                                                           )
                                                               )
    
    # # need to do np.squeeze for predictions in this case
    # if num_loci > 1:
    #     combined_genetic_scores = np.squeeze(combined_genetic_scores)
    
    # there are 128 samples. Sum along the first axis (length = 5) for each one, and combined into a single array. Shape should be 128 x longest_locus x num_loci
    combined_genetic_scores = np.array([score.sum(axis=0) for score in combined_genetic_scores])
    print(combined_genetic_scores.shape)
    
    # don't get why this needs to be done. Now it should have shape 128 x num_lineages
    combined_lineage_scores = np.array([score for score in combined_lineage_scores])
    print(combined_lineage_scores.shape)
    
    for idx, batch in enumerate(train_generator):
        if idx > 0:
            print(f"Working on batch {idx+1} of {len(train_generator)}")

            genetic_scores, lineage_scores = np.array(scoring_func(task_idx=0,
                                                                            input_data_list=batch[0],
                                                                            input_references_list=ref_data,
                                                                            batch_size=10,
                                                                            progress_update=None
                                                                           )
                                                                       )

            # need to do np.squeeze for predictions in this case
            if num_loci > 1:
                genetic_scores = np.squeeze(genetic_scores)

            # there are 128 samples. Sum along the first axis (length = 5) for each one, and combined into a single array. Shape should be 128 x longest_locus x num_loci
            genetic_scores = np.array([score.sum(axis=0) for score in genetic_scores])

            # don't get why this needs to be done. Now it should have shape 128 x num_lineages
            lineage_scores = np.array([score for score in lineage_scores])
            
            # combine them into a single array. The first axis is number of samples
            combined_genetic_scores = np.concatenate([combined_genetic_scores, genetic_scores], axis=0)
            combined_lineage_scores = np.concatenate([combined_lineage_scores, lineage_scores], axis=0)
            
    print(combined_genetic_scores.shape)
    print(combined_lineage_scores.shape)
    
    sparse.save_npz(os.path.join(output_path, "deeplift_outputs", save_prefix, "scores_all_strains.npy"), sparse.COO(combined_genetic_scores), compressed=True)
    sparse.save_npz(os.path.join(output_path, "deeplift_outputs", save_prefix, "scores_lineages.npy"), sparse.COO(combined_lineage_scores), compressed=True)

    # # Read in metadata
    # df_genos = pd.read_csv(os.path.join(output_path, "df_genos.csv"))
    # df_genos = df_genos.query("category=='original_train_set'")
    # print(len(df_genos))
    # print(scores_combined.shape[0])

    # Take max, min, median, and mean of saliency at each position
    max_score = np.max(combined_genetic_scores, axis=0)
    min_score = np.min(combined_genetic_scores, axis=0)
    median_score = np.median(combined_genetic_scores, axis=0)
    mean_score = np.mean(combined_genetic_scores, axis=0)
    
else:
    scoring_func = scoring_method.get_target_contribs_func(find_scores_layer_idx=0,
                                                           target_layer_idx=-1
                                                          )

    # add the scores for the first batch. first 0 = first batch. second 0 = inputs ([1] = output MICs)
    print(f"Working on batch 1 of {len(train_generator)}")
    scores_combined = np.array(
                                scoring_func(
                                            task_idx=0,
                                            input_data_list=[first_batch],
                                            input_references_list=ref_data,
                                            batch_size=10,
                                            progress_update=None
                                            )
                      )


    # output shape = 128 x 5 x longest_locus x num_loci. 
    # Sum along the first axis (5 nucleotides), which results in a shape of 128 x longest_locus x num_loci
    scores_combined = np.sum(scores_combined, axis=1)
    print(scores_combined.shape)

    for idx, batch in enumerate(train_generator):
        if idx > 0:
            print(f"Working on batch {idx+1} of {len(train_generator)}")

            # compute scores for the current batch
            scores = np.array(scoring_func(
                                            task_idx=0,
                                            input_data_list=[batch[0]],
                                            input_references_list=ref_data,
                                            batch_size=10,
                                            progress_update=None
                                          )
                             )

            # The sum over the ACGT axis in the code below is important! Recall that DeepLIFT
            # assigns contributions based on difference-from-reference; if
            # a position is [1,0,0,0] (i.e. 'A') in the actual sequence and [0.3, 0.2, 0.2, 0.3]
            # in the reference, importance will be assigned to the difference (1-0.3)
            # in the 'A' channel, (0-0.2) in the 'C' channel,
            # (0-0.2) in the G channel, and (0-0.3) in the T channel. You want to take the importance
            # on all four channels and sum them up, so that at visualization-time you can project the
            # total importance over all four channels onto the base that is actually present (i.e. the 'A'). If you
            # don't do this, your visualization will look very confusing as multiple bases will be highlighted at
            # every position and you won't know which base is the one that is actually present in the sequence!
            scores = np.sum(scores, axis=1)

            # combine with the running array 
            scores_combined = np.concatenate([scores_combined, scores], axis=0)


    # shape of this should be num_isolates x longest_locus x num_loci
    print(scores_combined.shape)
    sparse.save_npz(os.path.join(output_path, "deeplift_outputs", save_prefix, "scores_all_strains.npy"), sparse.COO(scores_combined), compressed=True)

    # # Read in metadata
    # df_genos = pd.read_csv(os.path.join(output_path, "df_genos.csv"))
    # df_genos = df_genos.query("category=='original_train_set'")
    # print(len(df_genos))
    # print(scores_combined.shape[0])

    # Take max, min, median, and mean of saliency at each position
    max_score = np.max(scores_combined, axis=0)
    min_score = np.min(scores_combined, axis=0)
    # median_score = np.median(scores_combined, axis=0)
    mean_score = np.mean(scores_combined, axis=0)

# save to file
np.save(os.path.join(output_path, "deeplift_outputs", save_prefix, "deeplift_max.npy"), max_score)
np.save(os.path.join(output_path, "deeplift_outputs", save_prefix, "deeplift_min.npy"), min_score)
# np.save(os.path.join(output_path, "deeplift_outputs", save_prefix, "deeplift_median.npy"), median_score)
np.save(os.path.join(output_path, "deeplift_outputs", save_prefix, "deeplift_mean.npy"), mean_score)
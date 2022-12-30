import sparse, sys, os, glob, yaml, tracemalloc, warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
warnings.filterwarnings("ignore")

# cnn_utils is one level up in the directory tree
sys.path.append(os.path.dirname(os.getcwd()))
from cnn_utils import *

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
bounded_loss = kwargs["bounded_loss"]
num_loci = len(locus_list)


# creat output directories
if binary:
    model_prefix = "binary_"
    save_prefix = "binary"
else:
    model_prefix = ""
    save_prefix = "quant"

    
# get longest locus from the pickle file
X_h37rv = sparse.load_npz(os.path.join(output_path, 'pkl_sparse_ref.npz'))
longest_locus = X_h37rv.shape[2]
del X_h37rv

# update output path for the saliency folder. Save the permutation models in a new subdirectory
saliency_dir = os.path.join(output_path, "saliency", save_prefix, "permutation_test")
    
if not os.path.isdir(saliency_dir):
    os.makedirs(os.path.join(saliency_dir))    
    
replicates = 10
for rep in range(replicates):
    
    print(f"Working on replicate {rep+1}/{replicates}")
    
    # for each replicate, randomly shuffle the MICs, so get new training data each time
    train_generator = MtbGeneDataset(
        os.path.join(output_path, 'pkl_sparse_train.npz'),
        phenotype_file,
        drug,
        locus_list,
        train_or_test="original_train_set",
        binary=binary,
        cc=binary_thresh,
        shuffle_phenos=True,
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


    # initialize the model using the function from cnn_utils and the optimizer
    model = conv_nn(binary, longest_locus, num_loci, num_lineages, bounded_loss, filter_size)
    optimizer = Adam(learning_rate = np.exp(-1.0 * 9))
    
    if bounded_loss:
        
        @tf.function
        def train_step(x, y):
            '''
            This is the training step for a single batch. Iterating over batches and epochs is done separately. Redefine and recompile this function for every permutation test model. 
            '''

            # the bounds are the last 2 elements of the x list
            lower_bounds, upper_bounds = x[-2:]

            with tf.GradientTape() as tape:

                # Make predictions using the model
                y_hat = model(x, training=True)

                # Calculate the loss using the two bounds tensors. custom_bounded_mae is imported from cnn_utils
                loss = boundedLoss_CNN(lower_bounds, upper_bounds)(y, y_hat)

            # Calculate the gradients
            gradients = tape.gradient(loss, model.trainable_weights)

            # run the optimizer
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            # return loss
            return loss


        # train the model with shuffled MICs. Don't keep track of losses; the only thing we want is the final model
        for epoch in range(N_epochs):

            for train_idx, (x_batch_train, y_batch_train) in enumerate(train_generator):

                _ = train_step(x_batch_train, y_batch_train) 

    else:
        
        if binary:
            loss_func = tf.keras.losses.BinaryCrossentropy()
            
            # get class weights for the training data only
            df_phenos = pd.read_csv(phenotype_file)
            
            if f"{drug}_midpoint" in df_phenos.columns:
                y_train = (df_phenos.query("category=='original_train_set'")[f"{drug}_midpoint"].values > binary_thresh).astype(int)
            else:
                y_train = df_phenos.query("category=='original_train_set'")["phenotype"].values.astype(int)

            assert len(np.unique(y_train)) == 2
            class_weights = class_weighting_dictionary(y_train)
            del y_train
            
        else:
            loss_func = tf.keras.losses.MeanSquaredError()
            class_weights = None
        
        model.compile(loss=loss_func, optimizer=optimizer)
        
        model.fit(x=train_generator, 
                  epochs=N_epochs,
                  use_multiprocessing=True,
                  workers=4,
                  class_weight=class_weights,
                )
                
    # save the model
    model.save(os.path.join(saliency_dir, f"permutation_{rep+1}.h5"))
    
K.clear_session()
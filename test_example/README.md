# Predicting MICs for new Isolates

The files in this directory contain the input data for 10 samples, which can be used to get a predicted rifampicin MIC. 

File descriptions:

<ul>
    <li><code>AA_train_mean.npy</code>: Array of mean values to scale the amino acid properties to a similar scale. This is because the three values are on very different scales, which can cause problems with modeling.</li>
    <li><code>AA_train_std.npy</code>: Array of standard deviations to scale the amino acid properties.</li>
    <li><code>best_model.h5</code>: Pre-trained CNN model</li>
    <li><code>cnn_predict.py</code>: Prediction script</li>
    <li><code>df_genos.csv</code>: Helper file containing the full sequences of the relevant regions for rifampicin MIC prediction. Note that additiona loci are included, but the models were only trained on the rpoBC region.</li>
    <li><code>df_phenos.csv</code>: Measured MICs for the 10 samples.</li>
    <li><code>full_model_results.csv</code>: Model statistics 
    <li><code>pkl_AA_test.npy</code>: Array of amino acid features encoded with hydrophobicity, molecular weight, and </li>
    <li><code>pkl_sparse_test.npz</code>: Sparse array of one-hot encoded nucleotides.</li>
    <li><code>seqDict.pkl</code>: Dictionary of dataframes for each of the genetic loci. Each dataframe is a matrix of the full sequence of each region of interest for each sample in the example dataset.</li>
    <li><code>test_predictions.csv</code>: Dataframe of predicted MICs obtained by running the code in the next section.</li>
</ul>
    
## Getting predictions

MIC predictions from the CNN model can be obtained by running the following, assuming that you have created the environment `tf2_models` from `../envs/tensorflow2.yaml`. 

```
conda activate tf2_models
cd ~/MtbQuantCNN
python3 test_example/cnn_predict.py -c config_files/config_rif.yaml -d test_example --lineage --amino-acid 
```

The `.npy` and `.npz` files and `df_genos.csv` are generated from a list of multiple sequence alignment FASTA files using the helper functions in `utils/data_utils.py`. See code in `test_example/cnn_predict.py` that calls the helper functions.

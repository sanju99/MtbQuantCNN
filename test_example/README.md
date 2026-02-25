# Predicting MICs for new Isolates

The files in this directory contain the input data for 10 samples, which can be used to get a predicted rifampicin MIC. 

File descriptions:

<ul>
    <li>`AA_train_mean.npy`: Array of mean values to scale the amino acid properties to a similar scale. This is because the three values are on very different scales, which can cause problems with modeling.</li>
    <li>`AA_train_std.npy`: Array of standard deviations to scale the amino acid properties.</li>
    <li>`best_model.h5`: Pre-trained CNN model</li>
    <li>`cnn_predict.py`: Prediction script</li>
    <li>`df_genos.csv`: Helper file containing the full sequences of the relevant regions for rifampicin MIC prediction. Note that additiona loci are included, but the models were only trained on the rpoBC region.</li>
    <li>`df_phenos.csv`: Measured MICs for the 10 samples.</li>
    <li>`full_model_results.csv`: Model statistics 
    <li>`pkl_AA_test.npy`: Array of amino acid features encoded with hydrophobicity, molecular weight, and </li>
    <li>`pkl_sparse_test.npz`: Sparse array of one-hot encoded nucleotides.</li>
</ul>
    
## Getting predictions

MIC predictions from the CNN model can be obtained by running the following, assuming that you have created the environment `tf2_models` from `../envs/tensorflow2.yaml`. 

```
conda activate tf2_models
cd ~/MtbQuantCNN
python3 test_example/cnn_predict.py -c config_files/config_rif.yaml -d test_example --lineage --amino-acid 
```
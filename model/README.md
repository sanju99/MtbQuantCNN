# Model Files

1. <code>cnn_custom_loss.py</code>: Trains quantitative or binary CNNs (with or without early stopping) using the custom bounded loss function. The quantiative CNN uses the custom binned loss function described in the manuscript. This script trains models for 5-fold cross-validation and a full model on all the data for downstream analysis.
2. <code>cnn_permutation_test.py</code>: Trains 10 permuted models, for which the inputs and outputs have been randomly shuffled.
3. <code>regression_cv.py</code>: Trains Ridge regression models using <code>sklearn</code>. It trains models for 5-fold cross-validation and a full model on all the data for downstream analysis.
4. <code>catalog_model.py</code>: Computes binary classification statistics using the 2023 WHO mutation catalog method. For this method, an isolate is predicted to be resistant to a given drug if it contains any Group 1 or 2 resistance mutation and susceptible if it does not contain any.

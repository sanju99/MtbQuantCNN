# Descriptions of Model Files

1. <code>data_utils.py</code>: Functions to process data to feed into models.
2. <code>model_utils.py</code>: Functions to create and train CNNs and compute performance metrics.
3. <code>dataloader.py</code>: Creates the <code>MtbGeneDataset</code> class to load data in batches for the CNN.
4. <code>cnn_custom_loss.py</code>: Trains a quantitative CNN with early stopping using the custom bounded loss function.
5. <code>cnn_standard.py</code>: Trains a quantitative or binary CNN using off-the-shelf TensorFlow loss functions.
6. <code>cv_custom_loss.py</code>: Trains 10 bootstrapped quantitative CNNs with the bounded loss function.
7. <code>cv_standard.py</code>: Trains 10 bootstrapped quantitative or binary CNNs with off-the-shelf loss functions.
8. <code>ridge_regression.py</code>: Trains and bootstraps Ridge regression models using <code>sklearn</code>.
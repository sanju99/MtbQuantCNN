# Descriptions of Model Files

1. <code>cnn_custom_loss.py</code>: Trains a quantitative CNN (with or without early stopping) using the custom bounded loss function.
2. <code>cnn_standard.py</code>: Trains a quantitative or binary CNN (with or without early stopping) using off-the-shelf TensorFlow loss functions.
3. <code>cv_custom_loss.py</code>: Trains 10 bootstrapped quantitative CNNs with the bounded loss function.
4. <code>cv_standard.py</code>: Trains 10 bootstrapped quantitative or binary CNNs with off-the-shelf loss functions.
5. <code>ridge_regression.py</code>: Trains and bootstraps Ridge regression models using <code>sklearn</code>.

# Descriptions of utils files

1. <code>data_utils.py</code>: Functions to process data to feed into models. Includes some functions for getting predictions from CNNs. 
2. <code>model_utils.py</code>: Functions to create and train CNNs and compute performance metrics.
3. <code>dataloader.py</code>: Creates the <code>MtbGeneDataset</code> class to load data in random batches for the CNN.
4. <code>analysis_utils.py</code>: Functions to generate summary dataframes and plots to compare models that have been fit.
5. <code>inSilicoMut_utils.py</code>: Functions to introduce mutations into H37Rv to create "synthetic" VCF files.
6. <code>saliency_utils.py</code>: Functions for computing and plotting saliency scores from DeepLIFT.

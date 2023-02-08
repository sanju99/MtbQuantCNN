# create environment with TF v1 for computing saliency scores
conda create --name cnn_saliency numpy pandas pyyaml sparse scikit-learn biopython tensorflow-gpu=1.15.0 h5py=2.10.0

# activate the environment
conda activate cnn_saliency
cd ~

# install deepexplain in the home directory
pip install -e git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain
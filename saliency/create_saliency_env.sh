# create environment with TF v1 for computing saliency scores
conda create --name tf1_saliency python=3.7 numpy pandas pyyaml sparse scikit-learn biopython tensorflow=1.15.0 h5py=2.10.0

# activate the environment
conda activate tf1_saliency
# cd ~
cd anaconda3/envs/tf1_saliency

# install deepexplain in the home directory
pip install -e "git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain"
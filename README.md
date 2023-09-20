# Convolutional Neural Networks to Predict Mtb Drug MICs

## Data Extraction and Cleaning

Please see <code>data_processing/README.md</code> for instructions.

## Bounded Loss Function


## Instructions for Making an Environment

```bash
module load gcc/6.2.0
module load cuda/11.2
conda create --file environment.yaml

conda activate tf2_models
pip install tensorflow
```

## Instructions for Running on GPU Nodes on O2 Cluster

```bash
module load gcc/6.2.0
module load cuda/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/app/cuda/11.2/

source activate tf2_models
python3 -u <script_name.py>
```

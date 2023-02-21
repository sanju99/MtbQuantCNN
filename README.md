# Convolutional Neural Networks to Predict Mtb Drug MICs

## Data Extraction and Cleaning

Please see <code>data_processing/README.md</code> for instructions.

## Bounded Loss Function


## Instructions for Making an Environment

```bash
module load gcc/6.2.0
module load cuda/11.2
conda create --name MtbQuantCNN --file /home/sak0914/MtbQuantCNN/environment_reqs.txt

conda activate MtbQuantCNN
pip install tensorflow
pip install evcouplings
```

## Instructions for Running on GPU Nodes on O2 Cluster

```bash
module load gcc/6.2.0
source activate MtbQuantCNN
module load cuda/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/app/cuda/11.2/

cd MtbQuantCNN/model
python3 -u <script_name.py>
```
import torch
import esm
from Bio import SeqIO
from Bio.Seq import Seq
import glob, os, argparse, shutil
import numpy as np
import pandas as pd

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

parser = argparse.ArgumentParser()

parser.add_argument("-d", dest='drug', default='config.ini', type=str, required=True)
parser.add_argument("-g", dest='gene', default='config.ini', type=str, required=True)
parser.add_argument('--TRUST-data', dest='TRUST_data', action='store_true', help='Flag to add TRUST samples paths to the scripts')

cmd_line_args = parser.parse_args()
drug = cmd_line_args.drug
gene = cmd_line_args.gene
TRUST_data = cmd_line_args.TRUST_data

if TRUST_data:
    embeddings_dir = f"{data_dir}/{drug}/TRUST/embeddings"
    fastas_dir = f"{data_dir}/{drug}/TRUST/fastas"
else:
    embeddings_dir = f"{data_dir}/{drug}/embeddings"
    fastas_dir = f"{data_dir}/{drug}/fastas"

print(embeddings_dir)
print(fastas_dir)

num_layers = 36
out_file = f"{embeddings_dir}/{gene}.csv.gz"

if os.path.isfile(out_file):
    print(f"Already converted pytorch embeddings for {gene} to a CSV")
    exit()

results_dir = f"{embeddings_dir}/{gene}"

if not os.path.isdir(results_dir):
    raise ValueError(f"There is no embeddings folder {results_dir}")

if len(os.listdir(results_dir)) == 0:
    raise ValueError(f"{results_dir} is an empty directory")

samples_lst = [seq.id for seq in list(SeqIO.parse(f"{fastas_dir}/{gene}_AA.fasta", "fasta"))]
assert samples_lst[-1] == 'MT_H37Rv'

print(f"Getting {gene} embeddings for {len(samples_lst)} samples")

embeddings_df = {}

for sample in samples_lst:

    # load in embeddings file from pytorch
    # the embedding is an N x 1280 matrix. 1280 is the output dimension of the ESM var model. N is the length of the protein, so there is an embedding for each AA
    embeddings = torch.load(f"{results_dir}/{sample}.pt")

    # dictionary object
    if 'representations' in embeddings.keys():
        matrix = embeddings['representations'][num_layers]
    elif 'mean_representations' in embeddings.keys():
        matrix = embeddings['mean_representations'][num_layers]

    if matrix.ndim == 2:
        embeddings_df[sample] = matrix.mean(axis=0)
    elif matrix.ndim == 1:
        embeddings_df[sample] = matrix

embeddings_df = pd.DataFrame(embeddings_df).T
print(embeddings_df.shape)

# keep index because that's where the samples are stored!!!
embeddings_df.to_csv(out_file, compression="gzip")

# then delete the directory of individual embeddings because it's redundant information
shutil.rmtree(results_dir)
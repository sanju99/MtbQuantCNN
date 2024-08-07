import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sparse, sys, warnings, yaml, vcf, pickle, shutil, tracemalloc
import scipy.optimize

import seaborn as sns
import scipy.stats as st
plt.rcParams['figure.dpi'] = 150

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
cnn_results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"

# starting the memory monitoring
tracemalloc.start()

_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs["drug"]
tier1_loci = kwargs["tier1_loci"]
embed_genes_list = kwargs["tier2_genes"] + kwargs["LASSO_genes"] # both of these groups are embedded
filter_size = kwargs["filter_size"]
BATCH_SIZE = kwargs["batch_size"]
N_PC = kwargs["N_PC"]
phenotype_file = kwargs["phenotype_file"]
genotype_input_directory = kwargs["genotype_input_directory"]
binary_thresh = kwargs["binary_thresh"]
output_path = f"/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/{drug}"

print(f"Performing PCA with {N_PC} components on protein embeddings for {drug}")

if not os.path.isdir(os.path.join(cnn_results_dir, drug)):
    os.makedirs(os.path.join(cnn_results_dir, drug))

df_phenos = pd.read_csv(phenotype_file)

# read in the original embeddings files for all genes in the directory. Column merge them. KEEP THEM IN ORDER OF EMBED_GENES_LIST FOR CONSISTENCY WITH VALIDATION DATA
df_embeddings = pd.concat([pd.read_csv(f"{data_dir}/{drug}/embeddings/{gene}.csv.gz", compression='gzip', index_col=[0]) for gene in embed_genes_list], axis=1, ignore_index=False)
print(df_embeddings.shape)

pca = PCA(n_components=N_PC)
scaler = StandardScaler()

X_pca = pca.fit_transform(scaler.fit_transform(df_embeddings.values))

# save PCA model to transform validation data later with the same model
pickle.dump(pca, open(f"{cnn_results_dir}/{drug}/PCA_{N_PC}.sav", "wb"))

print(f"PCA explained variance sum across {N_PC} components: {np.sum(pca.explained_variance_ratio_)}")

X_pca = pd.DataFrame(X_pca)
X_pca.columns = [f"PC{num}" for num in X_pca.columns]
X_pca['ROLLINGDB_ID'] = df_embeddings.index.values

# add lineages and MICs to the dataframe. Merge left to keep H37Rv in the dataframe. Need the embeddings later for doing in silico validation and saliency score computation
X_pca = X_pca.merge(df_phenos, on='ROLLINGDB_ID', how='left')

# keep index because the sample IDs are there
X_pca.set_index('ROLLINGDB_ID').to_csv(f"{cnn_results_dir}/{drug}/protein_embeddings_PC{N_PC}_transformed.csv")

# also save the PC explained variants
np.save(f"{cnn_results_dir}/{drug}/protein_embeddings_PC{N_PC}_explained_var_ratios", pca.explained_variance_ratio_)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
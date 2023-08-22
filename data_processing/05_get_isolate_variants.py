import numpy as np
import pandas as pd
import sys, subprocess, os, tracemalloc
from Bio import Seq, SeqIO

tracemalloc.start()

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_catalog_clean.csv")
who_variants["gene"] = [val.split("_")[0] for val in who_variants.mutation.values]

h37Rv_path = "/n/data1/hms/dbmi/farhat/Sanjana/H37Rv"
h37Rv_seq = SeqIO.read(os.path.join(h37Rv_path, "GCF_000195955.2_ASM19595v2_genomic.gbff"), "genbank")
h37Rv_genes = pd.read_csv(os.path.join(h37Rv_path, "mycobrowser_h37rv_genes_v4.csv"))
h37Rv_coords = pd.read_csv(os.path.join(h37Rv_path, "h37Rv_coords_to_gene.csv"))
h37Rv_coords_dict = dict(zip(h37Rv_coords["pos"].values, h37Rv_coords["region"].values))

_, drug = sys.argv

# use the original dataframes to get all isolates
df_train = pd.read_csv(os.path.join(data_dir, drug, "data_with_paths.csv"))

if os.path.isfile(os.path.join(data_dir, drug, "validation_data.csv")):
    df_val = pd.read_csv(os.path.join(data_dir, drug, "validation_data.csv"))
else:
    df_val = pd.DataFrame(columns=["ROLLINGDB_ID"])

with open(os.path.join(data_dir, drug, "isolates.txt"), "w") as file:

    for isolate in df_train["ROLLINGDB_ID"].values:
        file.write(isolate + "\n")

    for isolate in df_val["ROLLINGDB_ID"].values:
        file.write(isolate + "\n")

print(f"Extracting variants for {len(df_train) + len(df_val)} isolates")

train_isolate_variants = pd.read_csv(f"{data_dir}/trainVal_isolateVariants_AllDrugs.tsv", sep="\t", usecols=["Isolate"])
missing_isolates = set(df_train.ROLLINGDB_ID) - set(train_isolate_variants.Isolate)

if len(missing_isolates) > 0:
    print(f"{len(missing_isolates)} isolates are not in {data_dir}/isolate_variants.tsv")
    print(missing_isolates)
    exit()

get_isolate_variants_cmd = "awk -F '\t' 'FNR==NR {values[$1]; next} FNR==1 || $NF in values' " + f"{data_dir}/{drug}/isolates.txt " + f"{data_dir}/trainVal_isolateVariants_AllDrugs.tsv " + f"| sed 's/\t/,/g' > {data_dir}/{drug}/isolate_variants.csv"

# create isolate_variants.csv
subprocess.run(get_isolate_variants_cmd, shell=True)

# get a list of genes with category 1 mutations in the WHO catalog and save them to another text file
high_conf_genes = who_variants.query("drug==@drug & confidence=='1) Assoc w R'").gene.unique()
assert len(h37Rv_genes.query("Start > End")) == 0
assert len(high_conf_genes) == len(h37Rv_genes.query("Symbol in @high_conf_genes"))
print(f"Keeping variants in {high_conf_genes} with Category 1 R-associated mutations for {drug}")

if len(high_conf_genes) > 0:

    # this dataframe contains all variants in any Category 1 gene. The variants themselves may not be 100% accurate because SNPEff does not accurately annotated
    # multiple SNPs on the same codon that result in a different MNP. This will be fixed in the next script
    isolate_variants = pd.read_csv(f"{data_dir}/{drug}/isolate_variants.csv", usecols=["GENE"])
    keep_idx = isolate_variants.query("GENE.str.contains('|'.join(@high_conf_genes))").index.values

    # add 1 to the indices because bash uses 1-indexing, and Python uses 0-indexing, AND the header is included in the index
    pd.Series(keep_idx + 2).to_csv(os.path.join(data_dir, drug, "high_conf_variant_idx.txt"), sep="\t", index=False, header=None)

    get_high_conf_isolate_variants_cmd = "awk 'FNR==NR {indices[$1]; next} FNR in indices || FNR==1' " + f"{data_dir}/{drug}/high_conf_variant_idx.txt {data_dir}/{drug}/isolate_variants.csv > {data_dir}/{drug}/isolate_variants_high_conf.csv"
    
    subprocess.run(get_high_conf_isolate_variants_cmd, shell=True)
    subprocess.run(f"gzip -f {data_dir}/{drug}/isolate_variants.csv", shell=True)

    # remove extra files to clean up the directories (not because they take up space)
    os.remove(os.path.join(data_dir, drug, "high_conf_variant_idx.txt"))
    os.remove(os.path.join(data_dir, drug, "isolates.txt"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
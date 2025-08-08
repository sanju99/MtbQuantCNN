import numpy as np
import pandas as pd
import glob, os, subprocess, argparse, yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

drug_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")


def create_MSA_script(locus, drug, AF_thresh=0.75, TRUST_data=False, insilico_muts=False, saturation_muts=False, gene=None, hours=3, memory=1):

    constant_lines = ['#!/bin/bash', '#SBATCH -c 1', f'#SBATCH -t 0-0{hours}:00', '#SBATCH -p short', f'#SBATCH --mem={memory}G', '#SBATCH -o /home/sak0914/Errors/zerrors_%j.out', '#SBATCH -e /home/sak0914/Errors/zerrors_%j.err', '#SBATCH --mail-type=ALL', '#SBATCH --mail-user=skulkarni@g.harvard.edu']
        
    start, end, sense = drug_loci.query("Locus==@locus")[['Start', 'End', 'Sense']].replace('+', 'POS').replace('-', 'NEG').values[0]

    # in case they are stored as floats
    start = int(start)
    end = int(end)

    with open(os.path.join(out_dir, "bash_scripts", f"{locus}.sh"), "w+") as file:

        for line in constant_lines:
            file.write(line)
            file.write("\n")

        file.write("source activate MtbQuantCNN")
        file.write("\n")

        # for these, need to make an additional subdirectory for the variable locus, and the other loci FASTAs (all H37Rv ref seqs) will be in that subdirectory
        if insilico_muts:
            cmd = f"python3 -u ~/MtbQuantCNN/data_processing/make_MSA.py -f {drug_data_dir}/combined_paths_for_aln.txt -start {start} -end {end} -sense {sense} -o {out_dir}/{locus}/fastas/{locus}.fasta --save-fasta"
        elif saturation_muts:
            cmd = f"python3 -u ~/MtbQuantCNN/data_processing/make_MSA.py -f {drug_data_dir}/combined_paths_for_aln.txt -start {start} -end {end} -sense {sense} -o {out_dir}/{gene}/fastas/{locus}.fasta --save-fasta"
        else:
            cmd = f"python3 -u ~/MtbQuantCNN/data_processing/make_MSA.py -f {drug_data_dir}/combined_paths_for_aln.txt -start {start} -end {end} -sense {sense} -o {out_dir}/fastas/{locus}.fasta --save-fasta"
            
        if TRUST_data:
            cmd += " --f2 /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/vcf_full_paths.txt"

        if insilico_muts:
            cmd += f" --insilico-muts-file {out_dir}/{locus}/WHO_mutations.txt"

        # saturation mutagenesis is done at the gene level. Can get the gene name by splitting on the - character for loci like katG-furA and fabG1-inhA
        if saturation_muts:
            assert gene is not None
            cmd += f" --insilico-muts-file {out_dir}/{gene}/{gene}_mutations.txt"
        
        file.write(cmd + f" --AF-thresh {AF_thresh} \n")


parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

parser.add_argument('--model_suffix', dest='model_suffix', help='If specified, use the {drug}_{model_suffix} directory. Must be one of "binary" or "augment" if specified.')

parser.add_argument('--TRUST', dest='TRUST_data', action='store_true', help='Flag to add TRUST samples paths to the scripts')

parser.add_argument('--insilico-muts', dest='insilico_muts', action='store_true', help='Flag to add insilico mutation VCF paths to the scripts')

parser.add_argument('--saturation-muts', dest='saturation_muts', action='store_true', help='Flag to align site-saturation mutagenesis VCFs')

parser.add_argument('--locus', dest='locus', type=str)

parser.add_argument('--gene', dest='gene', type=str)

parser.add_argument('--AF-thresh', dest='AF_thresh', type=float, default=0.75, help='AF threshold for determining if variants are present vs. absent. Default = 0.75')

cmd_line_args = parser.parse_args()
config_file = cmd_line_args.config_file
model_suffix = cmd_line_args.model_suffix
assert model_suffix in ['binary', 'augment', None]
TRUST_data = cmd_line_args.TRUST_data
insilico_muts = cmd_line_args.insilico_muts
saturation_muts = cmd_line_args.saturation_muts
locus = cmd_line_args.locus
gene = cmd_line_args.gene
AF_thresh = cmd_line_args.AF_thresh

if AF_thresh > 1:
    AF_thresh /= 100

if TRUST_data and insilico_muts:
    raise ValuError(f"Both TRUST-data and insilico-muts arguments can't be specified. Please select only one.")

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs['drug']
locus_list = kwargs['tier1_loci'] + kwargs['tier2_loci']

# for this, the locus flag must also be specified. Only make the bash script for the locus to get in silico predictions for
if insilico_muts or saturation_muts:
    assert locus is not None
    locus_list = [locus]

if saturation_muts:
    assert gene is not None

if model_suffix is not None:
    drug_data_dir = os.path.join(data_dir, f"{drug}_{model_suffix}")
else:
    drug_data_dir = os.path.join(data_dir, drug)
    
# this is to remove training/validation/testing data from the alignments if we're only interested in TRUST data to save space and avoid having to re-encode nucleotide sequences for this data
df_phenos = pd.read_csv(os.path.join(drug_data_dir, "data_for_model.csv")) 

# create output directories
out_dir = drug_data_dir

if AF_thresh != 0.75:
    out_dir = f"{drug_data_dir}/AF_thresh_{int(AF_thresh*100)}"

if TRUST_data:
    out_dir = f"{drug_data_dir}/TRUST"

if insilico_muts:
    out_dir = f"{drug_data_dir}/inSilico_analysis"

if saturation_muts:
    out_dir = f"{drug_data_dir}/inSilico_analysis/saturation_mutagenesis"
    
print(f"Output directory: {out_dir}")
    
if TRUST_data:
    if not os.path.isdir(os.path.join(out_dir, "fastas")):
        os.makedirs(os.path.join(out_dir, "fastas"))

if not os.path.isdir(os.path.join(out_dir, "bash_scripts")):
    os.makedirs(os.path.join(out_dir, "bash_scripts"))


###################################### STEP 1: CREATE MSA OR SNP CONCATENATOR SCRIPT, DEPENDING ON THE WAY THE GENE/LOCUS WILL BE ENCODED ###################################### 


print(f"Generating nucleotide MSA scripts for {len(locus_list)} loci")

for locus in locus_list:
    create_MSA_script(locus, drug, AF_thresh=AF_thresh, TRUST_data=TRUST_data, insilico_muts=insilico_muts, saturation_muts=saturation_muts, gene=gene)


###################################### STEP 2: RUN MSA OR SNP CONCATENATOR SCRIPT ###################################### 

    
    if insilico_muts:
        out_file = f"{out_dir}/{locus}/fastas/{locus}.fasta"
    elif saturation_muts:
        out_file = f"{out_dir}/{gene}/fastas/{locus}.fasta"
    else:
        out_file = f"{out_dir}/fastas/{locus}.fasta"

    # don't rerun if the FASTA file already exists to save time
    if not os.path.isfile(out_file):
        subprocess.run(f"bash {out_dir}/bash_scripts/{locus}.sh", shell=True)


# then do the same for the Tier 1 loci if additional data is specified to reduce the size of these FASTA files and skip re-encoding the training data in one-hot format
if TRUST_data or insilico_muts or saturation_muts:
    
    for locus in locus_list:

        if insilico_muts:
            fName = f"{out_dir}/{locus}/fastas/{locus}.fasta"
        elif saturation_muts:
            fName = f"{out_dir}/{gene}/fastas/{locus}.fasta"
        else:
            fName = f"{out_dir}/fastas/{locus}.fasta"
        
        seq_df = pd.DataFrame([(seq.id, str(seq.seq)) for seq in SeqIO.parse(fName, "fasta")])
        seq_df.columns = ['Isolate', 'Seq']
        seq_df = seq_df.drop_duplicates()

        # remove training data isolates
        seq_df = seq_df.query("Isolate not in @df_phenos.ROLLINGDB_ID.values").reset_index(drop=True)
        print(f"Kept {len(seq_df)} additional sequences for {locus}")

        if TRUST_data:
            
            # then check that the length of the alignment is the same as for the original    
            new_seq = seq_df.query("Isolate=='MT_H37Rv'")['Seq'].values[0]
            old_seq = [(seq.id, str(seq.seq)) for seq in SeqIO.parse(fName.replace('/TRUST', ''), "fasta") if seq.id == "MT_H37Rv"][0][1] # first and only element is H37Rv, second element is sequence
    
            if len(new_seq) != len(old_seq):
                # raise ValueError(f"Alignments length for {drug}, {locus} differ -/+ TRUST isolates")
                # exit()
                print(f"Alignments length for {drug}, {locus} differ -/+ TRUST isolates")
    
        with open(fName, "w+") as file:
            for i, row in seq_df.iterrows():
                file.write(f">{row['Isolate']}\n")
                file.write(f"{row['Seq']}\n")
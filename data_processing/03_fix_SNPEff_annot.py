import numpy as np
import pandas as pd
import glob, os, sys, vcf, pickle, yaml, itertools
import Bio
import Bio.SeqUtils
from Bio import Seq, SeqIO

sys.path.append("utils")
from data_utils import *
from inSilicoMut_utils import *

# add stop codon to dictionary
Bio.SeqUtils.IUPACData.protein_letters_1to3["*"] = "*"

_, input_file, START, END = sys.argv

# BOTH ARE INCLUSIVE!!!
START = int(START)
END = int(END)

# python3 -u /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/RIF/rpoBC_variants_10543isolates.csv 759611 767320
# python3 -u /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/MXF/gyrBA_variants_8569isolates.csv 4998 9818
isolate_variants_df = pd.read_csv(input_file)
output_file = input_file.replace(".csv", "_fixed_annot.csv")

vcf_dir = "/n/scratch3/users/s/sak0914/annotated_VCF"

h37Rv_seq = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")

h37Rv_coords = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/h37Rv_coords_to_gene.csv")
h37Rv_coords.columns = ["POS", "Region"]
protein_coding_regions = h37Rv_coords.query("~Region.str.contains('NC_')").reset_index(drop=True)

# get the sense for each gene
h37Rv_genes_annot = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/mycobrowser_h37rv_genes_v4.csv")
# sense_dict = dict(zip(h37Rv_genes_annot["Symbol"], h37Rv_genes_annot["Strand"]))


def get_variants_each_gene(sample_id, isolate_variants_file):
    
    variants_in_each_gene = {key: [] for key in protein_coding_regions.query("POS >= @START & POS <= @END")["Region"].unique()}

    single_isolate_variants = isolate_variants_file.query("Isolate == @sample_id").reset_index(drop=True)
    
    for _, row in single_isolate_variants.iterrows():

        pos = int(row["POS"])

        if pos >= START and pos <= END:

            # check if the variant is in a protein-coding region
            if pos in protein_coding_regions["POS"].values:

                # get the corresponding gene
                assert len(protein_coding_regions.query("POS==@pos")["Region"].values) == 1
                gene = protein_coding_regions.query("POS==@pos")["Region"].values[0]

                variants_in_each_gene[gene].append(pos)
                
    return variants_in_each_gene




def within_same_codon(pos_lst):
    
    all_pairs = list(itertools.product(pos_lst, pos_lst))

    pos_to_fix = []

    for x, y in all_pairs:

        if np.abs(x - y) <= 2 and x != y:
            pos_to_fix.append(x)
        
    return np.unique(pos_to_fix)





def get_codon_idx_at_pos(genome_seq, pos, start, end):
    '''
    Get the codon that a nucleotide is in. Required arguments:
    
        genome_seq: full sequence of the genome
        codon_num: 1-indexed number of the desired codon to return. Uses 1-indexing because that's the standard numbering convention. i.e. 10th codon would be the 9th indexed codon, at indices 27-29
        start: gene start (inclusive)
        end: gene end (inclusive)
    
    Returns the codon nucleotides and their coordinates in H37Rv
    '''
    
    if pos > end or pos < start:
        raise ValueError(f"Position {pos} is out of the range [{start}, {end}]")
    
    # genome_seq is the full H37Rv sequence
    gene_seq = genome_seq.seq[start-1:end]
    
    # check that it is amultiple of 3, so it is only the protein-coding region
    assert len(gene_seq) % 3 == 0
        
    return gene_seq





def fix_mutations_single_sample(sample_id, isolate_variants_df):
    
    # get list of variants in each gene for a given sample
    variants_in_each_gene = get_variants_each_gene(sample_id, isolate_variants_df)    
    
    single_sample_variants = isolate_variants_df.query("Isolate == @sample_id").reset_index(drop=True)
    
    for gene in variants_in_each_gene.keys():
        pos_to_fix = within_same_codon(variants_in_each_gene[gene])

        if len(pos_to_fix) > 0:
            
            # list to keep track of which codons have been done
            fixed_pos = np.array([])

            for _, row in single_sample_variants.iterrows():

                pos = row["POS"]
                
                # only change SNPs, not indels
                if len(row["REF"]) == 1 and len(row["ALT"]) == 1:

                    if pos in pos_to_fix:

                        if pos not in fixed_pos:

                            gene = row["GENE"]
                            gene_start, gene_end = h37Rv_genes_annot.query("Symbol==@gene")[["Start", "End"]].values[0]

                            codon_num = int(((pos - gene_start) // 3) + 1)
                            codon_seq, codon_pos = get_codon_from_seq(h37Rv_seq.seq, codon_num, gene_start, gene_end)
                            ref_codon = str(codon_seq)
                            codon_seq = list(str(codon_seq))

                            mutated_codon_df = single_sample_variants.query("POS in @codon_pos")[["POS", "REF", "ALT"]].reset_index(drop=True)

                            for k in range(len(codon_pos)):
                                if codon_pos[k] in mutated_codon_df.POS.values:
                                    ref, alt = mutated_codon_df.loc[mutated_codon_df["POS"]==codon_pos[k], ["REF", "ALT"]].values[0]

                                    assert ref == codon_seq[k]
                                    # if ref != codon_seq[k]:
                                    #     print(sample_id, ref, codon_seq[k])

                                    codon_seq[k] = alt

                            mutated_codon = "".join(codon_seq)
                            # print(pos, ref_codon, mutated_codon)

                            for pos in codon_pos:
                                if pos in single_sample_variants["POS"].values:
                                    idx = single_sample_variants.query("POS == @pos").index.values[0]
                                    single_sample_variants.loc[idx, "mutation"] = row["GENE"] + "_p." + Bio.SeqUtils.IUPACData.protein_letters_1to3[Seq(ref_codon).translate()] + str(codon_num) + Bio.SeqUtils.IUPACData.protein_letters_1to3[Seq(mutated_codon).translate()]

                                    if Seq(mutated_codon).translate() == "*":
                                        single_sample_variants.loc[idx, "EFFECT"] = "stop_gained"
                                    else:
                                        if ref_codon == mutated_codon:
                                            single_sample_variants.loc[idx, "EFFECT"] = "synonymous_variant"
                                        else:
                                            single_sample_variants.loc[idx, "EFFECT"] = "missense_variant"

                            fixed_pos = np.unique(np.concatenate([fixed_pos, np.array(codon_pos)]))
                        
    return single_sample_variants



df_new_lst = []
samples_lst = np.unique(isolate_variants_df.loc[~pd.isnull(isolate_variants_df["POS"])]["Isolate"].values)

print(f"Fixing annotations for {len(samples_lst)} samples in the range [{START},{END}] and saving to {output_file}")
for i, sample_id in enumerate(samples_lst):
    
    df_new_lst.append(fix_mutations_single_sample(sample_id, isolate_variants_df))
    
    if i % 500 == 0:
        print(i)
        pd.concat(df_new_lst, axis=0).to_csv(output_file, index=False)

pd.concat(df_new_lst, axis=0).to_csv(output_file, index=False)
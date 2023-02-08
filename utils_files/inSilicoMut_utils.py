import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, sys, vcf

from Bio import SeqIO
from Bio.Seq import Seq
import Bio.SeqUtils
import Bio.Data


h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/mycobrowser_h37rv_genes_v4.csv")


def get_aa_to_codon_table():
    '''
    Creates a dataframe mapping each amino acid to all the codons that encode it. Length = 64, the number of codons.
    '''
    
    # make 3-letter AA code to codon dataframe. Can't make a dictionary because there would be one key mapping to multiple values
    bases = "TCAG"
    codons = [a + b + c for a in bases for b in bases for c in bases]
    amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
    codon_table = dict(zip(codons, amino_acids))

    aa_to_codon_table = pd.DataFrame(columns=["AA", "Codon"])

    for i, (key, val) in enumerate(codon_table.items()):

        if val in Bio.SeqUtils.IUPACData.protein_letters_1to3.keys():
            aa_to_codon_table.loc[i, :] = [Bio.SeqUtils.IUPACData.protein_letters_1to3[val], key]
        else:
            aa_to_codon_table.loc[i, :] = ["*", key]
            
    return aa_to_codon_table
        


def get_codon_from_seq(genome_seq, codon_num, start, end):
    '''
    Get a codon of interest from a genome sequence. Required arguments:
    
        genome_seq: full sequence of the genome
        codon_num: 1-indexed number of the desired codon to return. Uses 1-indexing because that's the standard numbering convention. i.e. 10th codon would be the 9th indexed codon, at indices 27-29
        start: gene start (inclusive)
        end: gene end (inclusive)
    
    Returns the codon nucleotides and their coordinates in H37Rv
    '''
    
    # genome_seq is the full H37Rv sequence
    gene_seq = genome_seq[start-1:end]
    
    if codon_num < 1:
        raise ValueError(f"Codon must be a natural number. {codon_num} is not")
    elif codon_num > len(gene_seq) / 3:
        raise ValueError(f"Protein length is {int(len(seq)/3)}. {codon_num} is longer than the protein.")
    
    codon_idx = codon_num - 1
    start_pos = int(start + codon_idx*3)
    
    # return the nucleotides of the codon and their genomic coordinates
    return gene_seq[int(codon_idx*3): int(codon_idx*3)+3], [start_pos, start_pos+1, start_pos+2]

       
        
        
def get_data_for_synthetic_VCF(df, sense):
    
    aa_to_codon_table = get_aa_to_codon_table()
    
    if "genome_index" in df.columns and "POS" not in df.columns:
        df.rename(columns={"genome_index": "POS"}, inplace=True)
        
    df[["gene", "variant"]] = df["mutation"].str.split("_", expand=True, n=1)
    
    if sense.upper() == "POS":
        
        for i, row in df.iterrows():

            # reference nucleotide at the site
            ref = h37Rv.seq[row["POS"] - 1]
            df.loc[i, "REF"] = ref

            if "p." not in row["variant"]:
                if ">" in row["variant"]:
                    df.loc[i, "ALT"] = row["variant"].split(">")[1]
                elif "ins" in row["variant"]:
                    df.loc[i, "ALT"] = ref + row["variant"].split("ins")[1]
                elif "del" in row["variant"]:
                    df.loc[i, "ALT"] = row["variant"].split("ins")[1].replace(ref, "")
            else:
                clean_var = row["variant"].replace("p.", "")

                # separate the mutation before, after, and position
                # get the indices of the position (numeric characters), then everything before or after is the mutation
                num_idx = []
                for k, char in enumerate(clean_var):
                    if char.isdigit():
                        num_idx.append(k)

                aa1 = clean_var[:num_idx[0]]
                aa2 = clean_var[num_idx[-1]+1:]

                if aa1 in Bio.SeqUtils.IUPACData.protein_letters_3to1.keys():

                    aa_pos = int(clean_var[num_idx[0]:num_idx[-1]+1])

                    start, end = h37Rv_genes.query(f"Symbol=='{row['gene']}'")[["Start", "End"]].values[0]
                    codon, codon_pos = get_codon_from_seq(h37Rv.seq, aa_pos, start, end)

                    # double checking methods. Check that the codon retrieved from the reference sequence is the same as the variant
                    assert Bio.SeqUtils.IUPACData.protein_letters_1to3[codon.translate()] == aa1

                    if aa2 in Bio.SeqUtils.IUPACData.protein_letters_3to1.keys():

                        # list of possible codons
                        possible_new_codons = aa_to_codon_table.query("AA==@aa2").Codon.values

                        # the index to replace. This looks for genome_index within the 3 positions of the codon values
                        idx_to_replace = codon_pos.index(row["POS"])

                        assert row["POS"] in codon_pos
                        df.loc[i, "ALT"] = np.random.choice(possible_new_codons)[idx_to_replace]

                    # insertion or deletion
                    else:
                        if "dup" in clean_var:
                            # get position, then randomly add one of the possible codons
                            df.loc[i, "ALT"] = ref + np.random.choice(possible_new_codons)

                #else:
                    # TODO: NEED AN EXAMPLE FIRST THO TO HELP FIGURE THIS ONE OUT
                    
    #else:
        # TODO: ALSO NEED AN EXAMPLE FOR NEGATIVE SENSE CASE BEFORE WRITING
        
    return df
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, sys, vcf

from Bio import SeqIO
from Bio.Seq import Seq
import Bio.SeqUtils
import Bio.Data
import warnings
warnings.filterwarnings("ignore")


h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/mycobrowser_h37rv_genes_v4.csv")



def split_multisite_mutations_dataframe(df):
    
    drop_mut = []
    add_df = pd.DataFrame(columns=df.columns)

    for i, row in df.iterrows():

        # convert to integers for reading/writing to VCF files and comparing to the saliency dataframe
        # in the saliency dataframe, all positions are converted to integers/floats, except for the indel rows
        if "," not in row["genome_index"]:
            df.loc[i, "genome_index"] = int(row["genome_index"])
        else:
            split_sites = row["genome_index"].split(",")

            for site in split_sites:
                add_df = pd.concat([add_df, pd.DataFrame({"drug": df["drug"].unique(),
                                                          "genome_index": int(site), 
                                                          "confidence": df["confidence"].unique(),
                                                          "mutation": row["mutation"]
                                                         }, index=[-1])], axis=0)


            drop_mut.append(row["mutation"])

    return pd.concat([df.query("mutation not in @drop_mut"), add_df], axis=0).reset_index(drop=True)




def get_dict_WHO_mutations_sites(who_variants_df, drug_abbr, gene=None):
    '''
    This function returns 2 dictionaries:
    
        1. one mapping an integer (corresponding to a WHO confidence category, i.e. 1-5) to a dataframe of unique mutations and their sites.
        2. one mapping an integer (corresponding to a WHO confidence category, i.e. 1-5) an array of all the unique nucleotide sites
        
    Arguments:
    
        1. who_variants_df: Dataframe of all WHO mutations across all drugs and categories
    
    There will be duplicate sites in the dataframe because multiple different mutations occur at the same nucleotide. 
    There will also be duplicate mutations if a mutation (usually an AA substitution) requires multiple SNVs in the same codon. Each SNV will be a new row, and they will have the same mutation field
    
    Splitting is necessary because the REF and ALT alleles for each nucleotide need to be filled in with another function.
    '''
    who_variants_single_drug = who_variants_df.query("drug == @drug_abbr")
    
    if gene is not None:
        who_variants_single_drug = who_variants_single_drug.query("mutation.str.contains(@gene)")
    
    sites_dict = {}
    dfs_dict = {}
    
    for num in range(1, 6):
        
        dfs_dict[num] = split_multisite_mutations_dataframe(who_variants_single_drug.loc[who_variants_single_drug["confidence"].str.contains(str(num))]).reset_index(drop=True)
        sites_dict[num] = np.unique(dfs_dict[num]["genome_index"])
        
    return sites_dict, dfs_dict





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
        raise ValueError(f"Protein length is {int(len(genome_seq)/3)}. {codon_num} is longer than the protein.")
    
    codon_idx = codon_num - 1
    start_pos = int(start + codon_idx*3)
    
    # return the nucleotides of the codon and their genomic coordinates
    return gene_seq[int(codon_idx*3): int(codon_idx*3)+3], [start_pos, start_pos+1, start_pos+2]

       
    
    
def is_list_consecutive(lst):
    '''
    Returns a boolean denoting whether the argument list contains ALL consecutive numbers or not
    '''
    return sorted(lst) == list(range(min(lst), max(lst)+1))



def split_consecutive_lists(lst):
    result = []
    start = 0
    end = 0
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]+1:
            end = i-1
            result.append(lst[start:end+1])
            start = i
    result.append(lst[start:])
    return result


        
def make_noncoding_mutation(df, row, idx, genome_seq):
    
    ref = str(genome_seq[row["POS"] - 1])
    df.loc[idx, "REF"] = ref

    if ">" in row["variant"]:
        df.loc[idx, "ALT"] = row["variant"].split(">")[1]
    elif "ins" in row["variant"]:
        df.loc[idx, "ALT"] = ref + row["variant"].split("ins")[1]
    elif "del" in row["variant"]:
        df.loc[idx, "ALT"] = row["variant"].split("del")[1].replace(ref, "")
        
        
        
        
def insert_nuc(clean_var, aa_to_codon_table, df, idx, pos, genome_seq):
    '''
    When one or more amino acids have to be inserted, extract the 3-letter AA abbreviations (from the mutation name) and get codons to insert.
    
    The codons are chosen randomly if there are multiple possibilities (due to the wobble effect).
    
    This is because the genome_index does't always line up. 
    '''
    
    insert_aa = clean_var.split("ins")[-1]

    # 3 letter abbreviations
    num_aa = int(len(insert_aa) / 3)
    add_nuc = ""

    for k in range(num_aa):
        aa = insert_aa[k*3:k*3+3]
        add_nuc += np.random.choice(aa_to_codon_table.query("AA==@aa").Codon.values)
        
    prev_nuc = str(genome_seq[pos-1])
    df.loc[idx, ["POS", "REF", "ALT"]] = [pos, prev_nuc, prev_nuc + add_nuc]
    
    
    
        
def get_data_for_synthetic_VCF(df, sense):
    
    aa_to_codon_table = get_aa_to_codon_table()
    
    df.rename(columns={"genome_index": "WHO_genome_index"}, inplace=True)
    df["POS"] = df.loc[:, 'WHO_genome_index']
        
    df = df.reset_index(drop=True)
    df[["gene", "variant"]] = df["mutation"].str.split("_", expand=True, n=1)
    
    if sense.upper() == "POS":
        
        for i, row in df.iterrows():
            
            # get the start and end coordinates of the gene where the mutation lies
            start, end = h37Rv_genes.query(f"Symbol=='{row['gene']}'")[["Start", "End"]].values[0]

            # mutations in noncoding regions. THESE ARE THE EASIEST CHANGES TO MAKE
            if "p." not in row["variant"]:
                make_noncoding_mutation(df, row, i, h37Rv.seq)
            
            # mutations in protein-coding regions
            else:
                clean_var = row["variant"].replace("p.", "")

                # separate the mutation before, after, and position
                # get the indices of the position (numeric characters), then everything before or after is the mutation
                num_idx = []
                for k, char in enumerate(clean_var):
                    if char.isdigit():
                        num_idx.append(k)
                        
                # SINGLE AMINO ACID CHANGES
                # if there are only consecutive numbers, that means that there's only a single number
                if is_list_consecutive(num_idx):

                    aa1 = clean_var[:num_idx[0]]
                    aa2 = clean_var[num_idx[-1]+1:]

                    aa_pos = int(clean_var[num_idx[0]:num_idx[-1]+1])                    
                    codon, codon_pos = get_codon_from_seq(h37Rv.seq, aa_pos, start, end)

                    # double checking methods. Check that the codon retrieved from the reference sequence is the same as the variant
                    assert Bio.SeqUtils.IUPACData.protein_letters_1to3[codon.translate()] == aa1

                    # INSERT SINGLE AMINO ACID
                    if aa2 in Bio.SeqUtils.IUPACData.protein_letters_3to1.keys():

                        # list of possible codons
                        possible_new_codons = aa_to_codon_table.query("AA==@aa2").Codon.values

                        # the index to replace. This looks for genome_index within the 3 positions of the codon values
                        idx_to_replace = codon_pos.index(row["POS"])

                        assert row["POS"] in codon_pos
                        df.loc[i, "REF"] = h37Rv.seq[row["POS"] - 1]
                        df.loc[i, "ALT"] = np.random.choice(possible_new_codons)[idx_to_replace]

                    # INSERT OR DELETE SINGLE AMINO ACID
                    else:
                        if "dup" in clean_var:
                            # get position, then randomly add one of the possible codons
                            df.loc[i, "REF"] = h37Rv.seq[row["POS"] - 1]
                            df.loc[i, "ALT"] = str(h37Rv.seq[row["POS"] - 1]) + np.random.choice(possible_new_codons)

                        elif "del" in clean_var:
                            # reference is the previous nucleotide AND the codon that will be deleted
                            df.loc[i, "REF"] = str(h37Rv.seq[row["POS"] - 1:row["POS"]+3])
                            # alternative is just the previous nucleotide
                            df.loc[i, "ALT"] = str(h37Rv.seq[row["POS"] - 1])
                
                # MULTI-AMINO ACID CHANGES
                else:
                    # this function returns all the consecutive lists available in num_idx. Each consecutive list is an AA
                    single_AA_site_lsts = split_consecutive_lists(num_idx)
                    
                    if len(single_AA_site_lsts) > 2:
                        raise ValueError("More than 2 amino acid coordinates", row["variant"].values)
                    else:
                        start_site = int(''.join(clean_var[k] for k in single_AA_site_lsts[0]))
                        end_site = int(''.join(clean_var[k] for k in single_AA_site_lsts[1]))

                    # don't need the actual codons here, just the nucleotides of the start and end
                    _, start_codon_sites = get_codon_from_seq(h37Rv.seq, start_site, start, end)
                    _, end_coord = get_codon_from_seq(h37Rv.seq, end_site, start, end)
                    end_coord = end_coord[-1]

                    # this is used in both the deletion and duplication cases. Get the nucleotides (inclusive) that need to be deleted or duplicated
                    intermediate_nucleotides = str(h37Rv.seq[start_codon_sites[0]-1:end_coord])
                    
                    # nucleotide coordinates to remove
                    if "del" in clean_var:

                        # use the previous nucleotide as the reference site, then the deletion comes right after it
                        prev_nuc = str(h37Rv.seq[start_codon_sites[0]-2])

                        if "ins" in clean_var:
                            insert_nuc(clean_var, aa_to_codon_table, df, i, start_codon_sites[0]-1, h37Rv.seq)
                        else:
                            df.loc[i, ["POS", "REF", "ALT"]] = [start_codon_sites[0]-1, prev_nuc + intermediate_nucleotides, prev_nuc]
                    
                    elif "ins" in clean_var:
                        insert_nuc(clean_var, aa_to_codon_table, df, i, start_codon_sites[-1], h37Rv.seq)
                        
                    elif "dup" in clean_var:
                        df.loc[i, ["POS", "REF", "ALT"]] = [end_coord, str(h37Rv.seq[end_coord]), str(h37Rv.seq[end_coord]) + intermediate_nucleotides] 
                    else:
                        print("Protein-coding, no indels", row["variant"])
    #else:
        # TODO: NEED AN EXAMPLE FOR NEGATIVE SENSE CASE
        
    return df




def create_synthetic_VCF_files(df, out_fName, vcf_dir="/n/scratch3/users/s/sak0914/annotated_VCF"):

    # create a header section
    header = '##fileformat=VCFv4.1\n'
    header += "##contig=<ID=NC_000962.3,length=4411532>\n"
    header += '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample\n'

    # text file for the list of mutation files (isolate name only) to pass into SNP concatenator later
    with open(out_fName, "w+") as out_file:

        # ITERATE through each mutation. N_mutations = N_files to be created. 
        # There can be multiple single site variants to make for a given mutation
        for mutation in df["mutation"].unique():
            
            # write the mutation name to the out text file    
            out_file.write(f'{mutation}\n')

            variants_to_add = []

            # iterate through each single variant for a given mutation
            for i, row in df.query("mutation==@mutation").reset_index(drop=True).iterrows():
                
                variants_to_add.append(['NC_000962.3', row["POS"], '.', row["REF"], row["ALT"], '.', 'PASS', '.', 'GT', '1/1'])

                # create a VCF file for the mutation
                with open(f'{vcf_dir}/{mutation}.vcf', 'w+') as vcf_file:

                    # write VCF file header
                    vcf_file.write(header)

                    # iterate through all the variants and add them
                    for variant in variants_to_add:
                        vcf_file.write('\t'.join(str(x) for x in variant) + '\n')
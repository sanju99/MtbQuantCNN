import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, itertools, sys, vcf, sparse

from Bio import SeqIO
from Bio.Seq import Seq
import Bio.SeqUtils
import Bio.Data
import warnings
warnings.filterwarnings("ignore")
from data_utils import *

h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")



def reverse_complement(seq):
    
    comp_dict = {'A': 'T', 
                 'C': 'G', 
                 'G': 'C', 
                 'T': 'A', 
                 'N': 'N', 
                 '-': '-'
                }
    
    # this is to turn it into a list where each element is of length 1
    seq = list("".join(seq))
    
    if len(np.unique(seq)) > 6:
        raise ValueError(f"More than 6 types of characters in the sequence!")

    if "X" in np.unique(seq):
        raise ValueError(f"There are Xs in the sequence!")
        
    seq = [comp_dict[base] for base in seq] 
    
    # reverse the sequence and return as a list
    return "".join(seq[::-1])


    

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
        
        if type(gene) is not list:
            gene = list(gene)
        
        who_variants_single_drug = who_variants_single_drug.loc[who_variants_single_drug["mutation"].str.contains("|".join(gene))]
    
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
        


def get_codon_from_seq(genome_seq, codon_num, start, end, sense):
    '''
    Get a codon of interest from a genome sequence. Required arguments:
    
        genome_seq: full sequence of the genome
        codon_num: 1-indexed number of the desired codon to return. Uses 1-indexing because that's the standard numbering convention. i.e. 10th codon would be the 9th indexed codon, at indices 27-29
        start: gene start (inclusive)
        end: gene end (inclusive)
    
    Returns the codon nucleotides and their coordinates in H37Rv
    '''

    if sense not in ["+", "-"]:
        raise ValueError(f"{sense} is not a valid strand sense!")
    
    # genome_seq is the full H37Rv sequence
    gene_seq = genome_seq[start-1:end]

    if sense == "-":
        gene_seq = reverse_complement(gene_seq)
    
    if codon_num < 1:
        raise ValueError(f"Codon must be a natural number. {codon_num} is not")
    elif codon_num > len(gene_seq) / 3:
        raise ValueError(f"Protein length is {int(len(gene_seq)/3)}. {codon_num} is longer than the protein.")
    
    codon_idx = codon_num - 1

    if sense == "+":
        start_pos = int(start + codon_idx*3)
        codon_pos = [start_pos, start_pos+1, start_pos+2]
    else:
        end_pos = int(end - codon_idx*3)
        codon_pos = [end_pos, end_pos-1, end_pos-2]
    
    # return the nucleotides of the codon and their genomic coordinates
    return gene_seq[int(codon_idx*3): int(codon_idx*3)+3], codon_pos




def within_same_codon(pos_lst):
    
    all_pairs = list(itertools.product(pos_lst, pos_lst))

    pos_to_fix = []

    for x, y in all_pairs:

        if np.abs(x - y) <= 2 and x != y:
            pos_to_fix.append(x)
        
    return np.unique(pos_to_fix)


    
    
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


        

def make_noncoding_mutation(df, row, idx, genome_seq, gene_start, gene_end, gene_sense):
    
    assert gene_sense in ["+", "-"]
    variant = row["variant"]

    # not an indel
    if ">" in variant:

        ref = str(genome_seq[row["POS"] - 1])
        df.loc[idx, "REF"] = ref # transforming coords into positive sense
        
        if gene_sense == '-':
            df.loc[idx, "ALT"] = reverse_complement(variant.split(">")[1])
        else:
            df.loc[idx, "ALT"] = variant.split(">")[1]
        
            
    elif "ins" in variant:

        ref = str(genome_seq[row["POS"] - 1])
        df.loc[idx, "REF"] = ref # genome_seq is positive sense

        # don't reverse complement the REF nucleotide because we want to reverse complement the inserted nucleotides to convert to positive sense coords
        if gene_sense == '-':
            df.loc[idx, "ALT"] = ref + reverse_complement(variant.split("ins")[1])
        else:
            df.loc[idx, "ALT"] = ref + variant.split("ins")[1]
    
    elif "del" in variant:

        dist = np.abs(int(variant.split("del")[0].split("c.")[1]))
        
        if gene_sense == '-':
            # subtract 1 from the position because this is a deletion
            new_pos = gene_end + dist - 1
            
            # upstream of a negative sense gene is downstream the positive sen
            ref = str(genome_seq[new_pos])
            del_nuc = reverse_complement(variant.split("del")[1])
        
            # ref is in positive sense
            assert ref == del_nuc
        
            # add the previous nucleotide to the reference, so that ref will be length N + 1, and alt will be length 1, where N is the length of the deletion
            ref = str(genome_seq[new_pos - 1: new_pos + 1])
            alt = ref.replace(del_nuc, "")
        else:
            # subtract 1 from the position because this is a deletion
            new_pos = gene_start - dist - 1
            
            # upstream of positive sense
            ref = str(genome_seq[new_pos])
            del_nuc = variant.split("del")[1]
            assert ref == del_nuc

            # add the previous nucleotide to the reference, so that ref will be length N + 1, and alt will be length 1, where N is the length of the deletion
            ref = str(genome_seq[new_pos - 1: new_pos + 1])
            alt = ref.replace(del_nuc, "")

        df.loc[idx, ["POS", "REF", "ALT"]] = [new_pos, ref, alt]

    return df
    
        
        
def aa_to_nuc(aa_lst, aa_to_codon_table, sense):
    '''
    Given a list string of 3-letter amino acid abbreviations, return a possible nucleotide sequence that would code for the peptide. 
    
    When one or more amino acids have to be inserted, extract the 3-letter AA abbreviations (from the mutation name) and get codons to insert.
    
    The codons are chosen randomly if there are multiple possibilities (due to the wobble effect).
    '''

    # 3 letter abbreviations
    assert len(aa_lst) % 3 == 0
    add_nuc = ""

    for k in range(0, len(aa_lst), 3):

        # get the 3-letter amino acid abbreviation
        single_aa = aa_lst[k:k+3] 
        add_nuc += np.random.choice(aa_to_codon_table.query("AA==@single_aa").Codon.values)

    if sense == "-":
        return reverse_complement(add_nuc)
    
    return add_nuc
    
    
    
        
def get_data_for_synthetic_VCF(df):
    '''
    For negative sense genes, the REF and ALT alleles must be converted to the positive sense coordinate system

    In make_MSA.py, variants are inputted into the reference sequence, then the entire sequence is reverse complemented for negative sense genes.

    Ignore frameshift variants because there are many possible variants that could cause frameshifts, and we have no way of knowing which ones are most likely for each gene without a master table
    '''
    
    aa_to_codon_table = get_aa_to_codon_table()
    
    df.rename(columns={"genome_index": "WHO_genome_index"}, inplace=True)
    df["POS"] = df.loc[:, 'WHO_genome_index'].astype(int)
        
    # exclude mutations on the last codon (usually a stop lost mutation) because we don't know what the actual mutation is and how much longer the protein goes on
    # exclude start lost mutations because again, we do not know what the identity of the codon is (it could literally be any of the other 19 amino acids)
    # also exclude frameshift variants because there are many possible variants that could cause frameshifts, and we have no way of knowing which ones should be added
    df = df.loc[~(df["mutation"].str.contains('|'.join(['Ter', 'fs']))) & ~(df["mutation"].str.endswith('Met1?'))].reset_index(drop=True)
    df[["gene", "variant"]] = df["mutation"].str.split("_", expand=True, n=1)
    
    for i, row in df.iterrows():
        
        # get the start and end coordinates of the gene where the mutation lies
        start, end, sense = h37Rv_genes.query(f"Symbol=='{row['gene']}'")[["Start", "End", "Strand"]].values[0]

        # mutations in noncoding regions. THESE ARE THE EASIEST CHANGES TO MAKE
        if "p." not in row["variant"]:
            df = make_noncoding_mutation(df, row, i, h37Rv.seq, start, end, sense)
        
        # mutations in protein-coding regions
        else:
            clean_var = row["variant"].replace("p.", "")

            # separate the mutation before, after, and position
            # get the indices of the position (numeric characters), then everything before or after is the mutation
            num_idx = []
            for k, char in enumerate(clean_var):
                if char.isdigit():
                    num_idx.append(k)

            # A SINGLE AMINO ACID MUST BE ALTERED, BUT MULTIPLE AMINO ACIDS COULD BE ADDED IN (i.e. delins)
            # if there are only consecutive numbers, that means that there's only a single number
            if is_list_consecutive(num_idx):

                aa1 = clean_var[:num_idx[0]]
                aa2 = clean_var[num_idx[-1]+1:]

                aa_pos = int(clean_var[num_idx[0]:num_idx[-1]+1])                    

                # for negative sense genes, codon_pos will be in decreasing order, but codon is the correct AA position 
                codon, codon_pos = get_codon_from_seq(h37Rv.seq, aa_pos, start, end, sense)

                # double checking methods. Check that the codon retrieved from the reference sequence is the same as the variant
                assert Bio.SeqUtils.IUPACData.protein_letters_1to3[Seq(codon).translate()] == aa1

                # INSERT SINGLE AMINO ACID
                if aa2 in aa_to_codon_table.AA.unique() or aa2 == "=": #Bio.SeqUtils.IUPACData.protein_letters_3to1.keys():

                    # list of possible codons
                    if aa2 in aa_to_codon_table.AA.unique():
                        possible_new_codons = aa_to_codon_table.query("AA==@aa2").Codon.values
                    # alternative start codons: most frequently Valine, but sometimes Leucine and Isoleucine
                    elif aa2 == "=":
                        possible_new_codons = aa_to_codon_table.query("AA in ['Val', 'Ile', 'Leu']").Codon.values

                    # check that the listed position is one of the positions determined from the genome sequence (this is just a sanity check)
                    assert row["POS"] in codon_pos

                    # the index to replace. This looks for genome_index within the 3 positions of the codon values
                    idx_to_replace = codon_pos.index(row["POS"])
                    idx_must_match = list(set(range(3)) - set([idx_to_replace]))

                    # find a codon that matches the other two nucleotides in the original codon. ONLY WANT TO CHANGE THE NUCLEOTIDE AT THE SPECIFIED POSITION
                    for new_codon in possible_new_codons:

                        # once it is found, break out of the loop
                        if new_codon[idx_must_match[0]] == codon[idx_must_match[0]] and new_codon[idx_must_match[1]] == codon[idx_must_match[1]]:
                            break

                    df.loc[i, "REF"] = h37Rv.seq[row["POS"] - 1]
                    
                    if sense == "+":
                        df.loc[i, "ALT"] = new_codon[idx_to_replace]
                    # for negative sense mutations, the reference is already in positive sense above, then simply reverse complement the new codon to get the alternative
                    else:
                        df.loc[i, "ALT"] = reverse_complement(new_codon[idx_to_replace])

                # INSERT (duplicate) OR DELETE SINGLE AMINO ACID, WHICH IS AA1
                else:
                    
                    # use the previous nucleotide as the reference site, then the deletion comes right after it. pos is the coordinate, but need pos - 1 to index the correct nucleotide
                    if sense == "+":

                        # this is the position in coordinate space of the previous nucleotide
                        pos = codon_pos[0]-1
                        prev_nuc = str(h37Rv.seq[pos-1]) # to get the actual nucleotide, subtract 1 to go into 0-indexed Python space

                        # Get the nucleotides (inclusive) that need to be deleted or duplicated
                        intermediate_nuc = str(h37Rv.seq[pos:codon_pos[-1]])
                        
                    else:
                        # for negative sense, have to take the downstream (lower coordinate number) nucleotide as the REF to support deletions as well
                        pos = codon_pos[-1]-1
                        prev_nuc = str(h37Rv.seq[pos-1]) # to get the actual nucleotide, subtract 1 to go into 0-indexed Python space

                        # for negative sense, reorder so that they are decreasing. codon_pos[-1] is the smallest position chronologically, up to pos
                        # intermediate_nuc = str(h37Rv.seq[codon_pos[-1]-1:pos])
                        intermediate_nuc = str(h37Rv.seq[pos:codon_pos[0]])
                                            
                    if "dup" in clean_var:
                        df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc, prev_nuc + intermediate_nuc]

                    elif "del" in clean_var:
                        # delete a single amino acid
                        if 'ins' not in clean_var:
                            df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc + intermediate_nuc, prev_nuc]
                        # delete the amino acid and add in an aribtrary number of amino acids
                        else:
                            # intermediate_nuc will be removed in the variant
                            # remove additional strings from the variant string on the right side of the position
                            aa2 = aa2.replace("del", "").replace("ins", "")
                            add_nuc = aa_to_nuc(aa2, aa_to_codon_table, sense)
                            
                            # the length of the additional nucleotides (3 per codon) should match the length of aa2, which as 3 letters per codon
                            assert len(add_nuc) == len(aa2)                                
                            df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc + intermediate_nuc, prev_nuc + add_nuc]
            
            # MULTIPLE AMINO ACID CHANGES ARE AFFECTED, SO NEED TO ITERATE THROUGH THEM
            else:
                # this function returns all the consecutive lists available in num_idx. Each consecutive list is an AA
                single_AA_site_lsts = split_consecutive_lists(num_idx)
                
                if len(single_AA_site_lsts) > 2:
                    raise ValueError("More than 2 amino acid coordinates", row["variant"].values)
                else:
                    start_site = int(''.join(clean_var[k] for k in single_AA_site_lsts[0]))
                    end_site = int(''.join(clean_var[k] for k in single_AA_site_lsts[1]))

                # don't need the actual codons here, just the nucleotides of the start and end
                _, start_codon_sites = get_codon_from_seq(h37Rv.seq, start_site, start, end, sense)
                _, end_coord = get_codon_from_seq(h37Rv.seq, end_site, start, end, sense)
                end_coord = end_coord[-1]

                # this is used in both the deletion and duplication cases. Get the nucleotides (inclusive) that need to be deleted or duplicated
                if sense == "+":
                    intermediate_nuc = str(h37Rv.seq[start_codon_sites[0]-1:end_coord])
                else:
                    intermediate_nuc = str(h37Rv.seq[end_coord-1:start_codon_sites[0]])
                
                # nucleotide coordinates to remove
                if "del" in clean_var:

                    if sense == "+":
                        # use the previous nucleotide as the reference site, then the deletion comes right after it. pos is the coordinate in 1-indexed space
                        pos = start_codon_sites[0]-1
                    else:
                        pos = end_coord-1
                    
                    prev_nuc = str(h37Rv.seq[pos-1]) # to get the actual nucleotide, subtract 1 to go into 0-indexed Python space
                    df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc + intermediate_nuc, prev_nuc]

                    # if there is an insertion, then update the alternative allele to have it
                    if "ins" in clean_var:
                        add_nuc = aa_to_nuc(clean_var.split("ins")[-1], aa_to_codon_table, sense)
                        df.loc[i, "ALT"] = prev_nuc + add_nuc
                
                elif "ins" in clean_var:
                    add_nuc = aa_to_nuc(clean_var.split("ins")[-1], aa_to_codon_table, sense)

                    if sense == "+":
                        # use the end of the start codon as the reference site, then insert nucleotides after it
                        pos = start_codon_sites[-1]
                    else:
                        # for negative sense, 
                        pos = start_codon_sites[-1]-1
                        
                    df.loc[i, ["POS", "REF", "ALT"]] = [pos, str(h37Rv.seq[pos-1]), str(h37Rv.seq[pos-1]) + add_nuc]
                
                elif "dup" in clean_var:
                    df.loc[i, ["POS", "REF", "ALT"]] = [end_coord, str(h37Rv.seq[end_coord]), str(h37Rv.seq[end_coord]) + intermediate_nuc] 
                
                else:
                    print("Protein-coding, no indels", row["variant"])

    # length checks
    df["REF_len"] = [len(val) for val in df["REF"].values]
    df["ALT_len"] = [len(val) for val in df["ALT"].values]
    
    assert len(df.query("REF_len == ALT_len & variant.str.contains('|'.join(['del', 'ins', 'dup']))")) == 0
    assert len(df.query("REF_len != ALT_len & ~variant.str.contains('|'.join(['del', 'ins', 'dup']))")) == 0

    del df["REF_len"]
    del df["ALT_len"]
    del df["WHO_genome_index"]
    del df["drug"]

    assert len(df.loc[pd.isnull(df['POS'])]) == 0
    assert len(df.loc[pd.isnull(df['REF'])]) == 0
    assert len(df.loc[pd.isnull(df['ALT'])]) == 0
    
    return df
    


def create_synthetic_VCF_files(df, out_fName, vcf_dir="/n/scratch3/users/s/sak0914/synthetic_VCF"):

    if not os.path.isdir(vcf_dir):
        os.makedirs(vcf_dir)
        
    if not os.path.isdir(os.path.dirname(out_fName)):
        os.makedirs(os.path.dirname(out_fName))
    
    # create a header section
    header = '##fileformat=VCFv4.1\n'
    header += "##contig=<ID=NC_000962.3,length=4411532>\n"
    header += '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample\n'

    print(f"Creating synthetic VCF files for {len(df['mutation'].unique())} mutations")
    
    # text file for the list of mutation files (isolate name only) to pass into SNP concatenator later
    with open(out_fName, "w+") as out_file:

        # ITERATE through each mutation. N_mutations = N_files to be created. 
        # There can be multiple single site variants to make for a given mutation
        for mutation in df["mutation"].unique():
            
            # absolute VCF file path
            vcf_fName = f'{vcf_dir}/{mutation}.vcf'
            
            # write the absolute path of the VCF file name to the out text file. Need this text file to pass into 06_make_MSA.py
            out_file.write(f'{vcf_fName}\n')

            variants_to_add = []

            # iterate through each single variant for a given mutation
            for i, row in df.query("mutation==@mutation").reset_index(drop=True).iterrows():
                
                variants_to_add.append(['NC_000962.3', row["POS"], '.', row["REF"], row["ALT"], '.', 'PASS', '.', 'GT', '1/1'])

                # create a VCF file for the mutation
                with open(vcf_fName, 'w+') as vcf_file:

                    # write VCF file header
                    vcf_file.write(header)

                    # iterate through all the variants and add them
                    for variant in variants_to_add:
                        vcf_file.write('\t'.join(str(x) for x in variant) + '\n')



def get_VCF_file_information(drug, seq_df_fName, genes_to_analyze, sense, START, END):
    '''
    genes_to_analyze can be a list or a string
    
    START is EXCLUSIVE and END is inclusive to be consistent with make_MSA.py
    '''
    
    seq_df = pd.read_csv(seq_df_fName).set_index("Isolate")
    seq_df.columns = [int(float(col)) if "." in col else col for col in seq_df.columns]

    # get dictionaries mapping WHO categories to unique sites and dataframes
    sites_dict, dfs_dict = get_dict_WHO_mutations_sites(who_variants, drug, genes_to_analyze)

    # all mutations merged with the saliency dataframe
    genes_ALL = pd.concat([dfs_dict[num] for num in range(1, 6)], axis=0)
    print(genes_ALL.shape)
    
    # get information to write VCF files
    # CURRENTLY, THIS ONLY WORKS IF ALL GENES IN THE LIST ARE THE SAME SENSE. IF NOT, PLEASE SPLIT THEM UP BEFORE RUNNING THIS FUNCTION
    genes_ALL = get_data_for_synthetic_VCF(genes_ALL)
    print(genes_ALL.shape)

    # keep only sites in our alignment
    genes_ALL["POS"] = genes_ALL["POS"].astype(int)
    
    genes_ALL = genes_ALL.query("POS > @START & POS <= @END")
    print(genes_ALL.shape)
    return genes_ALL




def get_peptide_lengths_WHO_mutations(drug, loci_with_mutations, locus_list, data_dir, WHO_mutations_df):
    
    peptide_lengths_df = make_H37Rv_CDS_length_df(locus_list, os.path.join(data_dir, drug, "fastas"))
    
    # initialize dataframe of peptide lengths, index = WHO mutations for insilico mutagenesis
    WHO_mutations_peptide_lengths = pd.DataFrame(columns=[f"{locus}_length" for locus in locus_list], index=list(WHO_mutations_df["mutation"].values) + ["MT_H37Rv"])

    for locus in locus_list:
    
        # sum of the protein lengths of all genes in the locus
        full_locus_protein_lengths = peptide_lengths_df.query("Locus==@locus").Length.sum()

        # for all mutations, the loci in which they don't occur in are all the same length
        if locus not in loci_with_mutations:
            
            WHO_mutations_peptide_lengths[f"{locus}_length"] = full_locus_protein_lengths

        else:
            
            # H37Rv wild-type
            WHO_mutations_peptide_lengths.loc["MT_H37Rv",  f"{locus}_length"] = full_locus_protein_lengths
            
            for i, row in WHO_mutations_df.iterrows():
            
                gene = row["gene"]
                
                if "*" in row['mutation']:
                    # single gene protein product length
                    full_protein_length = peptide_lengths_df.query("Gene==@gene").Length.values[0]
                    
                    AA_pos = int(''.join([val for val in row['mutation'] if val.isnumeric()]))
            
                    # the AA_pos (1-indexed) is the one that has become a stop codon, so (AA_pos - 1) amino acids remain
                    num_AA_remove = full_protein_length - (AA_pos - 1)
                    WHO_mutations_peptide_lengths.loc[row['mutation'], f"{locus}_length"] = full_locus_protein_lengths - num_AA_remove
            
                else:
                    # nothing to remove, full peptide length
                    WHO_mutations_peptide_lengths.loc[row['mutation'], f"{locus}_length"] = full_locus_protein_lengths
    
    return WHO_mutations_peptide_lengths
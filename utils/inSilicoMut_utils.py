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
h37Rv_regions = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_v4.csv")

drug_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")

who_variants_V2 = pd.read_csv("./data_processing/data_utils/WHO_catalog_V2.csv", header=[2]).reset_index(drop=True)
who_variants_V1 = pd.read_csv("./data_processing/data_utils/WHO_catalog_V1.csv")

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETO",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMK",
                  "Pretomanid": "PMD",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LFX",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

abbr_drug_dict = {value: key for key, value in drug_abbr_dict.items()}


def compute_codon_edit_distance(codon1, codon2):
    '''
    Use this function to determine which of the possible codons to use. Preferentially use the one that is closest in distance space to th  reference codon.
    '''
    # double check that codon lengths match
    assert len(codon1) == len(codon2)
    
    # Count the number of differing positions
    return sum(1 for a, b in zip(codon1, codon2) if a != b)
    


def select_new_codon_from_list(ref_codon, codon_lst):

    # only 1 possible codon
    if len(codon_lst) == 1:
        return codon_lst[0]
        
    codons_select_from_df = pd.DataFrame({'codon': codon_lst})
    
    for i, row in codons_select_from_df.iterrows():
        codons_select_from_df.loc[i, 'ref_distance'] = compute_codon_edit_distance(ref_codon, row['codon'])

    # take the codon with the smallest distance from the reference
    return codons_select_from_df.sort_values("ref_distance", ascending=True)['codon'].values[0]



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




def get_WHO_mutations(drug, V1=True, genes_lst=None):
    '''
    This function returns 2 dictionaries:
    
        1. one mapping an integer (corresponding to a WHO confidence category, i.e. 1-5) to a dataframe of unique mutations and their sites.
        2. one mapping an integer (corresponding to a WHO confidence category, i.e. 1-5) an array of all the unique nucleotide sites
        
    Arguments:
    
        1. who_variants: Dataframe of all WHO mutations across all drugs and categories
    
    There will be duplicate sites in the dataframe because multiple different mutations occur at the same nucleotide. 
    There will also be duplicate mutations if a mutation (usually an AA substitution) requires multiple SNVs in the same codon. Each SNV will be a new row, and they will have the same mutation field
    
    Splitting is necessary because the REF and ALT alleles for each nucleotide need to be filled in with another function.
    '''

    if V1:
        who_variants_single_drug = who_variants_V1.query("drug == @drug")
        
    else:
        drug = abbr_drug_dict[drug]

        # keep only Tier 1 for V2 of the catalog
        who_variants_single_drug = who_variants_V2.query("drug == @drug")

    if genes_lst is not None:
        
        if type(genes_lst) is not list:
            genes_lst = list(genes_lst)

        return who_variants_single_drug.loc[who_variants_single_drug["gene"].isin(genes_lst)]

    else:
        return who_variants_single_drug


    




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
        



def get_variant_number(variant, gene_sense, return_start_and_end=False):
    '''
    This function takes a variant (no gene name) and returns the numerical part. If a variant has multiple numbers because it spans multiple sites, it returns the first one.

    It works on protein-coding, synonymous, and non-protein-coding variants

    Returns the end position for negative sense genes and the start position for positive sense to return the MOST UPSTREAM

    If return_start_and_end: then both start and end are returned. 
    
        By default (False), the earliest site in a protein is returned, which is the most upstream position. For negative sense genes,
        that would be later in the protein actually. 
    '''
    
    num_idx = []
    for k, char in enumerate(variant):
        if char.isdigit():
            num_idx.append(k)
    
    if is_list_consecutive(num_idx):
        variant_number = int(variant[num_idx[0]:num_idx[-1]+1]) 
    
    else:
        # this function returns all the consecutive lists available in num_idx. Each consecutive list is an AA
        single_site_lsts = split_consecutive_lists(num_idx)
        
        if len(single_site_lsts) > 2:
            raise ValueError("More than 2 amino acid coordinates", variant)
        else:
            start_site = int(''.join(variant[k] for k in single_site_lsts[0]))
            end_site = int(''.join(variant[k] for k in single_site_lsts[1]))
            
            # return both, but first need to determine if they are negative
            if return_start_and_end:

                if variant.replace('p.', '').replace('c.', '').replace('n.', '')[0] == '-':
                    start_site *= -1
                    
                if variant.replace('p.', '').replace('c.', '').replace('n.', '').split('_')[-1][0] == '-':
                    end_site *= -1
                    
                return start_site, end_site

            else:
                if gene_sense == '-':
                    variant_number = end_site
                else:
                    variant_number = start_site

    # make it negative if the variant is upstream of the gene
    if variant.replace('p.', '').replace('c.', '').replace('n.', '')[0] == '-':
        variant_number *= -1

    return variant_number



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
    '''
    Everything should be in positive sense in the VCF (because that's what make_MSA.py assumes), so coordinates are transformed using reverse_complement for negative sense genes
    '''
    
    assert gene_sense in ["+", "-"]
    variant = row["mutation"] # without the gene prefix
    gene = row['gene']

    # this is the numerical part of the variant. If there are multiple, then it returns the smallest position (most upstream)
    # Remove the gene name so that if there are numbers in the gene name, they are not included
    variant_number = get_variant_number(variant.replace(gene, ''), gene_sense, return_start_and_end=False)

    # the start of the gene is 0, so if the variant number is negative, then simply add that number (negative) from the start
    # if the variant number is positive, then add variant_number and subtract 1
    if gene_sense == '+':
        if variant_number < 0:
            variant_pos = gene_start + variant_number
        else:
            variant_pos = gene_start + variant_number - 1
    else:
        # subtract negative to go farther upstream of the gene
        if variant_number < 0:
            variant_pos = gene_end - variant_number
        # same as above, but inverse the signs
        else:
            variant_pos = gene_end - variant_number + 1

    # not an indel
    if ">" in variant:

        # subtract 1 for proper Python 0-indexing
        ref = str(genome_seq[variant_pos - 1])

        # position is the exact position above. For indels, take one position upstream to make the variants easier to create
        df.loc[idx, ["POS", "REF"]] = [variant_pos, ref]
        
        if gene_sense == '-':
            df.loc[idx, "ALT"] = reverse_complement(variant.split(">")[1])
        else:
            df.loc[idx, "ALT"] = variant.split(">")[1]

    # not necessarily indels. MNPs could be here in the form delNNNinsNNN. This is how some synonymous variants are encoded
    else:

        # position will be 1 upstream of the current position (which is where the actual deletion or insertion occurs) UNLESS IT'S AN MNP, WHICH IS ENCODED AS DELINS
        new_pos = variant_pos - 1

        # duplications always occur alone, never with del or ins
        # same process as for insertions alone?
        if "dup" in variant:

            # if insertion occurs at position 10, then the variant in the synthetic VCF file should be at position 10. i.e. 10insAT --> POS = 10, REF = C, ALT = CAT
            # subtract 1 because of 0-indexing in Python 
            ref = str(genome_seq[variant_pos - 1])

            df.loc[idx, ["POS", "REF"]] = [new_pos, ref] # genome_seq is positive sense
    
            # don't reverse complement the REF nucleotide because we want to reverse complement the inserted nucleotides to convert to positive sense coords
            if gene_sense == '-':
                df.loc[idx, "ALT"] = ref + reverse_complement(variant.split("dup")[1])
            else:
                df.loc[idx, "ALT"] = ref + variant.split("dup")[1]
        
        elif "ins" in variant:
    
            # if insertion occurs at position 10, then the variant in the synthetic VCF file should be at position 10. i.e. 10insAT --> POS = 10, REF = C, ALT = CAT
            # subtract 1 because of 0-indexing in Python 
            ref = str(genome_seq[variant_pos - 1])
    
            if "del" in variant:

                # the nucleotides in the name will be in the same sense as the gene. So need to reverse complement if negative sense
                # note that the POS column is already the first position of the codon substitution, because of the function above run on the who_variants file for the V2 catalog results
                if gene_sense == '-':
                    df.loc[idx, ['POS', 'REF', 'ALT']] = [variant_pos, reverse_complement(variant.split("del")[1].split("ins")[0]), reverse_complement(variant.split("ins")[1])]
                else:
                    df.loc[idx, ['POS', 'REF', 'ALT']] = [variant_pos, variant.split("del")[1].split("ins")[0], variant.split("ins")[1]]
    
            # nucleotide insertion ONLY
            else:
                df.loc[idx, ["POS", "REF"]] = [new_pos, ref] # genome_seq is positive sense
        
                # don't reverse complement the REF nucleotide because we want to reverse complement the inserted nucleotides to convert to positive sense coords
                if gene_sense == '-':
                    df.loc[idx, "ALT"] = ref + reverse_complement(variant.split("ins")[1])
                else:
                    df.loc[idx, "ALT"] = ref + variant.split("ins")[1]
    
        # nucleotide deletion ONLY
        elif "del" in variant:
    
            if gene_sense == '+':
    
                # upstream of positive sense
                # if deletion occurs at position 10, then the variant in the synthetic VCF file should be at position 9. i.e. 10delAT --> POS = 9, REF = CAT, ALT = C
                # new_pos = gene_start - variant_number - 1
                alt = str(genome_seq[new_pos - 1]) # because deletion, the alternative allele is the nucleotide in the upstream position. The reference will include the deleted nucleotides
                del_nuc = variant.split("del")[1]
    
                # ref is not necessarily equal to del_nuc because multiple nucleotides may have been removed
                # add the previous nucleotide to the reference, so that ref will be length N + 1, and alt will be length 1, where N is the length of the deletion
                ref = alt + del_nuc
                
            else:
                # upstream of a negative sense gene is downstream the positive sense
                # subtract 1 like above to get the nucleotide BEFORE the deletion, then you have to subtract 1 again from new_pos because of 0-indexing in Python
                # new_pos = gene_end + variant_number - 1
                alt = str(genome_seq[new_pos - 1]) # because deletion, the alternative allele is the nucleotide in the upstream position. The reference will include the deleted nucleotides
                del_nuc = reverse_complement(variant.split("del")[1])

                # for extremely long deletions (longer than 100 nucleotides I think), the deletion string is not written out, so need to find it from the reference genome
                if len(del_nuc) == 0:

                    # get the start and end of the deletion, by default above, we only return the earliest
                    # both of these will be 1-indexed, and if they are negative, it will be reflected
                    deletion_start, deletion_end = get_variant_number(variant.replace(gene, ''), gene_sense, return_start_and_end=True)

                    # I think this works for both positive and negative variant numbers
                    deletion_length = deletion_end - deletion_start + 1

                    if gene_sense == '+':
                        del_nuc = str(h37Rv.seq[variant_pos-1: variant_pos-1+deletion_length])
                    else:
                        del_nuc = reverse_complement(str(h37Rv.seq[variant_pos-1: variant_pos-1+deletion_length]))
                    
                # add the previous nucleotide to the reference, so that ref will be length N + 1, and alt will be length 1, where N is the length of the deletion
                ref = alt + del_nuc

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

        # randomly select a codon. Because this is for insertions, there is no reference codon to compare to, so pick a codon at random
        add_nuc += np.random.choice(aa_to_codon_table.query("AA==@single_aa").Codon.values)

    if sense == "-":
        return reverse_complement(add_nuc)
    
    return add_nuc
    
    
    
        
def get_data_for_synthetic_VCF(df):
    '''
    For negative sense genes, the REF and ALT alleles must be converted to the positive sense coordinate system

    In make_MSA.py, variants are inputted into the reference sequence, then the entire sequence is reverse complemented for negative sense genes.

    Ignore variants for which we can't deduce an exact nucleotide change from the identity of the variant itself. i.e. stop lost, start lost, and frameshift mutations are named in such a way that we don't know what the nucleotide change is, and it is such a big change that it could be anything. For single amino acid changes, even though we pick a codon at random (which may not necessarily be a naturally-occurring codon), it's not such a big change. 
    '''
    
    aa_to_codon_table = get_aa_to_codon_table()

    alternative_start_codons = ['ATG', 'CTG', 'GTG', 'TTG', 'ATA', 'ATC', 'ATT']
    stop_codons = aa_to_codon_table.query("AA=='*'").Codon.values
    
    # exclude the following:
    # 1. stop lost mutations because we don't know what the actual mutation is and how much longer the protein goes on
    # 2. frameshift variants because there are many possible variants that could cause frameshifts, and we have no way of knowing which ones should be added    
    # 3. stop_lost mutations because we don't know how much longer the sequence would continue before the enzymes falling off
    # 4. pooled LoF and feature ablation variants

    # this is only for in silico mutagenesis. For SSM, all mutations are missense or nonsense, so this doesn't apply. Don't need to remove any mutations before generating variant calls
    if 'effect' in df.columns:
        # start_lost and stop_lost both end in '?'. stop lost is encoded with ext*? start_lost is p.Met1? or p.Val1?
        df = df.query("effect not in ['stop_lost', 'frameshift', 'LoF', 'feature_ablation']").reset_index(drop=True)
        
    for i, row in df.iterrows():

        gene = row['gene']
        
        # get the start and end coordinates of the gene where the mutation lies
        # this can be done for both coding and non-coding transcripts, so use the full table, not the genes-only table
        start, end, sense = h37Rv_regions.query(f"Name==@gene")[['Start', 'Stop', 'Strand']].values[0]
        
        # this is for checking if it's the first codon. Remove the gene name so that if there are numbers in the gene name, they are not included
        variant_number = get_variant_number(row['mutation'].replace(gene, ''), sense)
    
        df.loc[i, ['sense', 'variant_number']] = [sense, variant_number]
    
        # mutations in noncoding regions. THESE ARE THE EASIEST CHANGES TO MAKE
        if "p." not in row["mutation"]:
            df = make_noncoding_mutation(df, row, i, h37Rv.seq, start, end, sense)
        
        # mutations in protein-coding regions
        else:            
            clean_var = row["mutation"].replace("p.", "").replace(gene, "")
    
            # separate the mutation before, after, and position
            # get the indices of the position (numeric characters), then everything before or after is the mutation
            num_idx = [k for k, char in enumerate(clean_var) if char.isdigit()]
    
            # A SINGLE AMINO ACID MUST BE ALTERED, BUT MULTIPLE AMINO ACIDS COULD BE ADDED IN (i.e. delins)
            # if there are only consecutive numbers, that means that there's only a single number
            if is_list_consecutive(num_idx):
    
                aa1 = clean_var[:num_idx[0]]
                aa2 = clean_var[num_idx[-1]+1:]
    
                # this is a bug in the WHO catalog -- an early stop codon is recorded as Rv0565c_p.TrpLeu266*, instead of Rv0565c_p.Trp266*
                if aa2 == '*' and len(aa1) > 3:
                    aa1 = aa1[:3]
    
                aa_pos = int(clean_var[num_idx[0]:num_idx[-1]+1])                    
    
                # for negative sense genes, codon_pos will be in decreasing order, but codon is in the order of translation (5' - 3')
                codon, codon_pos = get_codon_from_seq(h37Rv.seq, aa_pos, start, end, sense)
    
                # double checking methods. Check that the codon retrieved from the reference sequence is the same as the variant
                if variant_number == 1:
                    # alternative start codons are possible
                    assert Bio.SeqUtils.IUPACData.protein_letters_1to3[Seq.Seq(codon).translate()] in ['Met', 'Val', 'Ile', 'Leu']
                else:
                    assert Bio.SeqUtils.IUPACData.protein_letters_1to3[Seq.Seq(codon).translate()] == aa1

                # start lost encoded in a different way
                if variant_number == 1 and row['variant'].endswith('?'):
    
                    # alternative allele is randomly chosen from the codons that are not alternative start codons nor stop codons (excluding stop codons isn't really necessary, but do it anyway)
                    codon_search_lst = list(set(aa_to_codon_table.Codon.values) - set(alternative_start_codons) - set(stop_codons))
                    new_codon = select_new_codon_from_list(codon, codon_search_lst)

                    # convert all to positive sense system.
                    if sense == '+':
                        df.loc[i, ['POS', 'REF', 'ALT']] = [np.min(codon_pos), str(codon), str(new_codon)]
                    else:
                        df.loc[i, ['POS', 'REF', 'ALT']] = [np.min(codon_pos), str(reverse_complement(codon)), str(reverse_complement(new_codon))]
    
                else:
                    
                    # INSERT SINGLE AMINO ACID
                    if aa2 in aa_to_codon_table.AA.unique():
    
                        # list of possible codons
                        if aa2 not in aa_to_codon_table.AA.unique():
                            raise ValueError(f"{aa2} is not a valid amino acid")
    
                        # for the first codon, if it's one of the AAs encoded by alternative start codons, keep only those codons
                        if aa_pos == 1 and aa2 in ['Val', 'Leu', 'Ile', 'Met']:
    
                            # pick a new codon at random intersecting with the alternative start codons
                            codon_search_lst = list(set(aa_to_codon_table.query("AA==@aa2").Codon.values).intersection(alternative_start_codons))
                            new_codon = select_new_codon_from_list(codon, codon_search_lst)
    
                        else:
                            # pick a new codon at random
                            codon_search_lst = list(set(aa_to_codon_table.query("AA==@aa2").Codon.values))
                            new_codon = select_new_codon_from_list(codon, codon_search_lst)
                            
                        # reference = original codon.
                        # put the earliest (most upstream) position in the codon as the position
                        df.loc[i, 'POS'] = np.min(codon_pos)
    
                        # alternative = new codon
                        if sense == "+":
                            df.loc[i, ['REF', 'ALT']] = [str(codon), new_codon]
                        elif sense == '-':
                            df.loc[i, ['REF', 'ALT']] = [str(reverse_complement(codon)), reverse_complement(new_codon)]
                        else:
                            raise ValueError(f"{sense} is not a valid strand sense")
    
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
                            # delete the amino acid and add in the inserted nucleotides
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
    
                    if sense == "+":
                        # use the previous nucleotide as the reference site, then the deletion comes right after it. pos is the coordinate in 1-indexed space
                        pos = start_codon_sites[0]-1
                        # the first position in start_codon_sites will be the largest number, then add 1
                    else:
                        pos = end_coord-1
                    
                    df.loc[i, ["POS", "REF", "ALT"]] = [pos, str(h37Rv.seq[pos-1]), str(h37Rv.seq[pos-1]) + intermediate_nuc] 
                
                else:
                    print("Protein-coding, no indels", row["variant"])
    
    # check that there are no NaNs
    assert len(df[['POS', 'REF', 'ALT']].dropna()) == len(df)

    # length checks
    df["REF_len"] = [len(val) for val in df["REF"].values]
    df["ALT_len"] = [len(val) for val in df["ALT"].values]
    df["POS"] = df["POS"].astype(int)

    # for synonymous variants, there are many with a del ins annotation because one codon is removed and another inserted. These will not pass, so only consider the p and n variants
    # I *think* _n. variants will pass because it just denotes that gene = rrs / rrl
    assert len(df.query("~variant.str.contains('_c.') & REF_len == ALT_len & variant.str.contains('|'.join(['del', 'ins', 'dup']))")) == 0
    assert len(df.query("~variant.str.contains('_c.') & REF_len != ALT_len & ~variant.str.contains('|'.join(['del', 'ins', 'dup']))")) == 0

    del df["REF_len"]
    del df["ALT_len"]
    
    if 'drug' in df.columns:
        del df["drug"]
    
    return df
    


def in_ranges(value, ranges):
    '''
    Function to check if a value falls within any of the given ranges. This is to be able to subset a list of variants and keep only those with positions within the model loci
    '''
    
    return any(start < value <= end for start, end in ranges)



def get_VCF_file_information(drug, tier2=False, V1=False, genes_to_analyze=None):
    '''
    genes_to_analyze can be a list or a string
    
    START is EXCLUSIVE and END is inclusive to be consistent with make_MSA.py
    '''

    # get the list of loci to get in silico mutations for from the config file
    kwargs = yaml.safe_load(open(f"config_files/config_{drug.lower()}.yaml"))

    if tier2:
        locus_list = kwargs['tier1_loci'] + kwargs['tier2_loci']
    else:
        locus_list = kwargs['tier1_loci']
        
    # get all WHO catalog mutations from the V1 OR V2 editions
    who_variants_single_drug = get_WHO_mutations(drug, V1=V1, genes_lst=genes_to_analyze)

    # make the POS, REF, and ALT columns
    who_variants_single_drug = get_data_for_synthetic_VCF(who_variants_single_drug)
    
    # keep only variants that lie in the range of nucleotides that the model was trained on. Start is 0-indexed, End is 1-indexed
    who_variants_single_drug = who_variants_single_drug[who_variants_single_drug['POS'].apply(lambda x: in_ranges(x, drug_loci.loc[drug_loci["Locus"].isin(locus_list)][['Start', 'End']].values))]
    
    return who_variants_single_drug



def create_synthetic_VCF_files(df, out_fName, vcf_dir):

    assert len(df) == df.variant.nunique()
    
    # create a header section
    header = '##fileformat=VCFv4.1\n'
    header += "##contig=<ID=Chromosome,length=4411532>\n"
    header += '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample\n'

    print(f"Creating synthetic VCF files for {df['mutation'].nunique()} mutations")

    with open(out_fName, 'w+') as out_file:
         
        # ITERATE through each mutation. N_mutations = N_files to be created. 
        # There can be multiple single site variants to make for a given mutation
        for _, row in df.iterrows():

            # need to remove special characters
            mutation_str = row['variant'].replace('.', '_').replace('*', '+')

            # absolute VCF file path
            vcf_fName = f"{vcf_dir}/{mutation_str}.vcf"

            out_file.write(vcf_fName + "\n")
                            
            # create a VCF file for the mutation
            with open(vcf_fName, 'w+') as vcf_file:

                # write VCF file header
                vcf_file.write(header)

                # write the variant that causes the mutation
                vcf_file.write('\t'.join(['Chromosome', str(row["POS"]), '.', row["REF"], row["ALT"], '.', 'PASS', '.', 'GT', '1/1']) + '\n')




def check_annotation_matches_fName(fName):
    '''
    The goal of this function is to check that the mutations made in the synthetic VCF files are correct. So we check that the snpEff annotation matches the name of the file (the intende mutation).

    Simple because only one variant per mutation (i.e. each VCF file has a single line)
    '''

    # remove all file extensions and directories
    variant_fName = os.path.basename(fName).split(".")[0]
        
    with open(fName, "r") as file:
        lines = file.readlines()
    
    # remove VCF header lines
    lines = [line for line in lines if line[0] != '#']
    
    for chars in lines[0].split('\t'):
        if 'ANN' in chars:
            annotation = chars

    # include both nucleotide change and AA change
    vcf_effects = []
    
    for annot in annotation.split('|'):
        if 'p.' in annot or 'c.' in annot or 'n.' in annot:
            vcf_effects.append(annot)
            
    return vcf_effects    
    


def make_H37Rv_CDS_length_df(locus_list, fasta_dir):
    '''
    IMPORTANT: Saliency functions must be run before using this function to make seqDict_fName

    Two indices: Locus and Gene because there are multiple genes in a single locus.

    The goal of this function is to make a dataframe of the H37Rv protein lengths of all genes in all loci in a model. 
    Then for insilico mutagenesis, you can get the position of the early stop codon and update the peptide length by subtracting the number of AAs that would be truncated from the H37Rv length

    NOTE: This is not supported for WHO catalog frameshift mutations that cause early stop codons. HOWEVER, in the PZA models that this is being used for, there are no such frameshift mutations because they increase the size of the alignment so they were removed from this analysis step. 
    '''

    # get the dataframe of start and end coordinates from mycobrowser
    h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

    gene_peptide_lengths_df = pd.DataFrame(columns=["Locus", "Gene", "Length"]).set_index(["Locus", "Gene"])
    locus_gene_dict = {}

    for locus in locus_list:

        with open(os.path.join(fasta_dir, f"{locus}.sh"), "r") as file:
            for line in file.readlines():
                if ".py" in line and ".fasta" in line:
                    lines = line.split(" ")
        
        start_end_lst = []
        
        for arg in lines:
            if arg.isdigit():
                start_end_lst.append(int(arg))
        
        assert len(start_end_lst) == 2
        locus_start, locus_end = start_end_lst[0] + 1, start_end_lst[-1]

        # all the genes in a single locus
        genes_lst = h37Rv_genes.query("Start >= @locus_start & End <= @locus_end").Symbol.values

        # keep track of the component genes in a single locus
        locus_gene_dict[locus] = genes_lst
        
        single_locus_peptide_lengths_df = pd.DataFrame(columns=[f"{gene}_length" for gene in genes_lst])

        # sum the lengths of all the genes within the locus
        WT_protein_length = 0

        for gene in genes_lst:

            sense = h37Rv_genes.query("Symbol==@gene")['Strand'].values[0]

            # reverse start and end for negative sense genes because the seqDict dataframes are in translated order (which is easy for translating the nucleotide sequences to AAs)
            if sense == "+":
                start, end = h37Rv_genes.query("Symbol==@gene")[['Start', 'End']].values[0]
            else:
                start, end = h37Rv_genes.query("Symbol==@gene")[['End', 'Start']].values[0]

            # subtract 1 because of the stop character
            gene_peptide_lengths_df.loc[(locus, gene), :] =  int((np.abs(end - start) + 1) / 3) - 1

    return gene_peptide_lengths_df.reset_index(), locus_gene_dict




def get_peptide_lengths_WHO_mutations(drug, locus_list, loci_with_mutations, data_dir, WHO_mutations_df):
    
    peptide_lengths_df, locus_gene_dict = make_H37Rv_CDS_length_df(locus_list, os.path.join(data_dir, drug, "fastas"))

    # WHO_mutations_peptide_lengths = pd.DataFrame(columns=[f"{locus}_length" for locus in locus_list], index=list(WHO_mutations_df["mutation"].values) + ["MT_H37Rv"])
    WHO_mutations_peptide_lengths = pd.DataFrame(columns=[f"{gene}_length" for gene in peptide_lengths_df.Gene.unique()], index=list(WHO_mutations_df["mutation"].values) + ["MT_H37Rv"])

    for locus in locus_list:

        # get all the genes in the locus
        genes_lst = locus_gene_dict[locus]

        for gene in genes_lst:
    
            # sum of the protein lengths of all genes in the locus
            # full_locus_protein_lengths = peptide_lengths_df.query("Locus==@locus").Length.sum()
            assert len(peptide_lengths_df.query("Gene==@gene")) == 1
            full_gene_protein_length = peptide_lengths_df.query("Gene==@gene").Length.values[0]
    
            # for all mutations, the loci in which they don't occur in are all the same length
            if locus not in loci_with_mutations or gene not in WHO_mutations_df.gene.unique():
                WHO_mutations_peptide_lengths[f"{gene}_length"] = full_gene_protein_length
    
            else:
                
                # H37Rv wild-type
                WHO_mutations_peptide_lengths.loc["MT_H37Rv",  f"{gene}_length"] = full_gene_protein_length
                
                for i, row in WHO_mutations_df.iterrows():
                
                    # we're not encoding frameshift variants because they they are in the catalog, it is not clear what the variant is, just that it causes a frameshift
                    # also exclude variants where the protein is extended because again don't have a good way to get the exact sequence. Just know that the protein has been extended
                    if "*" in row['mutation'] and "?" not in row['mutation']:
                        
                        AA_pos = int(''.join([val for val in row['mutation'] if val.isnumeric()]))
                
                        # the AA_pos (1-indexed) is the one that has become a stop codon, so (AA_pos - 1) amino acids remain
                        # num_AA_remove = full_gene_protein_length - (AA_pos - 1)
                        WHO_mutations_peptide_lengths.loc[row['mutation'], f"{gene}_length"] = AA_pos - 1 #full_gene_protein_length - num_AA_remove
                
                    else:
                        # nothing to remove, full peptide length
                        WHO_mutations_peptide_lengths.loc[row['mutation'], f"{gene}_length"] = full_gene_protein_length
    
    return WHO_mutations_peptide_lengths
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, itertools, sys, vcf, sparse
from evcouplings.align import Alignment
from sklearn.preprocessing import StandardScaler

from Bio import SeqIO
from Bio.Seq import Seq
import Bio.SeqUtils
import Bio.Data
import warnings
warnings.filterwarnings("ignore")


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


        
def make_noncoding_mutation(df, row, idx, genome_seq):
    
    ref = str(genome_seq[row["POS"] - 1])
    df.loc[idx, "REF"] = ref

    if ">" in row["variant"]:
        df.loc[idx, "ALT"] = row["variant"].split(">")[1]
    elif "ins" in row["variant"]:
        df.loc[idx, "ALT"] = ref + row["variant"].split("ins")[1]
    elif "del" in row["variant"]:
        df.loc[idx, "ALT"] = row["variant"].split("del")[1].replace(ref, "")
        
        
        
        
def insert_nuc(clean_var, aa_to_codon_table):
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
        
    return add_nuc
    
    
    
        
def get_data_for_synthetic_VCF(df, sense):
    
    aa_to_codon_table = get_aa_to_codon_table()
    
    df.rename(columns={"genome_index": "WHO_genome_index"}, inplace=True)
    df["POS"] = df.loc[:, 'WHO_genome_index']
        
    # exclude mutations on the last codon (usually a stop lost mutation) because we don't know what the actual mutation is and how much longer the protein goes on
    df = df.query("~mutation.str.contains('Ter')").reset_index(drop=True)
    df[["gene", "variant"]] = df["mutation"].str.split("_", expand=True, n=1)
    
    if sense.lower() == "pos":
        
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

                        # check that the listed position is one of the positions determined from the genome sequence (this is just a sanity check)
                        assert row["POS"] in codon_pos

                        # list of possible codons
                        possible_new_codons = aa_to_codon_table.query("AA==@aa2").Codon.values

                        # the index to replace. This looks for genome_index within the 3 positions of the codon values
                        idx_to_replace = codon_pos.index(row["POS"])
                        idx_must_match = list(set(range(3)) - set([idx_to_replace]))

                        # find a codon that matches the other two nucleotides in the original codon. ONLY WANT TO CHANGE THE NUCLEOTIDE AT THE SPECIFIED POSITION
                        for new_codon in possible_new_codons:

                            # once it is found, break out of the loop
                            if new_codon[idx_must_match[0]] == codon[idx_must_match[0]] and new_codon[idx_must_match[1]] == codon[idx_must_match[1]]:
                                break

                        df.loc[i, "REF"] = h37Rv.seq[row["POS"] - 1]
                        df.loc[i, "ALT"] = new_codon[idx_to_replace]

                    # INSERT OR DELETE SINGLE AMINO ACID, WHICH IS AA1
                    else:
                        
                        # use the previous nucleotide as the reference site, then the deletion comes right after it. pos is the coordinate, but need pos - 1 to index the correct nucleotide
                        pos = codon_pos[0]-1
                        prev_nuc = str(h37Rv.seq[pos-1])
                        
                        # Get the nucleotides (inclusive) that need to be deleted or duplicated
                        intermediate_nuc = str(h37Rv.seq[pos:codon_pos[-1]])
                    
                        if "dup" in clean_var:
                            df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc, prev_nuc + intermediate_nuc]

                        elif "del" in clean_var:
                            df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc + intermediate_nuc, prev_nuc]
                
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
                    intermediate_nuc = str(h37Rv.seq[start_codon_sites[0]-1:end_coord])
                    
                    # nucleotide coordinates to remove
                    if "del" in clean_var:

                        # use the previous nucleotide as the reference site, then the deletion comes right after it. pos is the coordinate, but need pos - 1 to index the correct nucleotide
                        pos = start_codon_sites[0]-1
                        prev_nuc = str(h37Rv.seq[pos-1])
                        
                        df.loc[i, ["POS", "REF", "ALT"]] = [pos, prev_nuc + intermediate_nuc, prev_nuc]

                        # if there is an insertion, then update the alternative allele to have it
                        if "ins" in clean_var:
                            add_nuc = insert_nuc(clean_var, aa_to_codon_table)
                            df.loc[i, "ALT"] = prev_nuc + add_nuc
                    
                    elif "ins" in clean_var:
                        add_nuc = insert_nuc(clean_var, aa_to_codon_table)
                        pos = start_codon_sites[-1]
                        df.loc[i, ["POS", "REF", "ALT"]] = [pos, str(h37Rv.seq[pos-1]), str(h37Rv.seq[pos-1]) + add_nuc]
                    
                    elif "dup" in clean_var:
                        df.loc[i, ["POS", "REF", "ALT"]] = [end_coord, str(h37Rv.seq[end_coord]), str(h37Rv.seq[end_coord]) + intermediate_nuc] 
                    
                    else:
                        print("Protein-coding, no indels", row["variant"])
    #else:
        # TODO: NEED AN EXAMPLE FOR NEGATIVE SENSE CASE
        
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






def get_train_test_val_lineages(df_train, df_test, df_val, lineage_fName="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv"):
    
    lineages = pd.read_csv(lineage_fName, index_col=[0])
    assert len(np.unique(lineages.values)) == 2

    train_lineages = lineages.loc[df_train["ROLLINGDB_ID"].values]
    assert sum(train_lineages.index.values != df_train["ROLLINGDB_ID"].values) == 0

    test_lineages = lineages.loc[df_test["ROLLINGDB_ID"].values]
    assert sum(test_lineages.index.values != df_test["ROLLINGDB_ID"].values) == 0

    val_lineages = lineages.loc[df_val["ROLLINGDB_ID"].values]
    assert sum(val_lineages.index.values != df_val["ROLLINGDB_ID"].values) == 0
    
    return train_lineages, test_lineages, val_lineages


def get_combined_model_inputs(X, lineages_matrix, model_type, include_lineage):
    
    if model_type == "CNN":
        
        if include_lineage:
            model_inputs = [X, lineages_matrix.values, np.zeros(len(X)), np.zeros(len(X))]
        else:
            model_inputs = [X, np.zeros(len(X)), np.zeros(len(X))]
        
    elif model_type == "Regression":
        
        scaler = StandardScaler()
        
        if include_lineage:
            model_inputs = scaler.fit_transform(np.concatenate([X, lineages_matrix.values], axis=1))
        else:
            model_inputs = scaler.fit_transform(X)
        
    else:
        raise ValueError(f"{model_type} is not a valid model type!")
        
    return model_inputs



###################### CATALOG BASED CLASSIFICATION ######################


def mutation_catalog_with_bootstrapping(df, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats=["Sensitivity", "Specificity", "AUC", "Accuracy", "Balanced_Acc"]):
    
    df = df.rename(columns={"ROLLINGDB_ID": "Isolate"}).reset_index(drop=True)
    cat1_mutations = who_variants_df.query("drug == @drug & confidence=='1) Assoc w R'").mutation.values
    isolates_R = isolate_variants_df.query("mutation in @cat1_mutations & FILTER == 'PASS' & Isolate in @df.Isolate.values").Isolate.values
        
    df_pred_catalog = df[["Isolate", f"{drug}_midpoint"]]
    df_pred_catalog["y_test"] = (df[f"{drug}_midpoint"] > binary_thresh).astype(int)
    df_pred_catalog["y_pred"] = df_pred_catalog["Isolate"].map(dict(zip(isolates_R, np.ones(len(isolates_R))))).fillna(0).astype(int)
    
    df_stats = compute_binary_metrics(df_pred_catalog["y_test"], df_pred_catalog["y_pred"], binary_thresh, binarize=False)[return_stats]
    df_stats["CV"] = 0
    bs_lst = []
    
    # perform bootstrapping with 10 replicates
    for i in range(10):
        
        bs_sample_idx = np.random.choice(df.index.values, size=len(df), replace=True)
        bs_df = df.iloc[bs_sample_idx, :]
        bs_isolates_R = isolate_variants_df.query("mutation in @cat1_mutations & FILTER == 'PASS' & Isolate in @bs_df.Isolate.values").Isolate.values
        
        bs_pred_catalog = bs_df[["Isolate", f"{drug}_midpoint"]]
        bs_pred_catalog["y_test"] = (bs_df[f"{drug}_midpoint"] > binary_thresh).astype(int)
        bs_pred_catalog["y_pred"] = bs_pred_catalog["Isolate"].map(dict(zip(bs_isolates_R, np.ones(len(bs_isolates_R))))).fillna(0).astype(int)
        
        bs_df_stats = compute_binary_metrics(bs_pred_catalog["y_test"], bs_pred_catalog["y_pred"], binary_thresh, binarize=False)[return_stats]
        bs_df_stats["CV"] = i + 1
        bs_lst.append(bs_df_stats)

    df_return = pd.concat([df_stats, pd.concat(bs_lst, axis=0)], axis=0).reset_index(drop=True)
    df_return["Model"] = "Catalog"
    return df_return



def classify_using_mutation_catalog(drug, data_path, who_variants_df, isolate_variants_df, binary_thresh, return_stats=["Sensitivity", "Specificity", "AUC", "Accuracy", "Balanced_Acc"]):

    df_train = pd.read_csv(os.path.join(data_path, drug, "data_for_model.csv")).query("category=='original_train_set'")
    df_test = pd.read_csv(os.path.join(data_path, drug, "data_for_model.csv")).query("category=='original_test_set'")
    df_val = pd.read_csv(os.path.join(data_path, drug, "validation_data_for_model.csv"))
        
    df_train = mutation_catalog_with_bootstrapping(df_train, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats)
    df_train["Dataset"] = "Train"
    
    df_test = mutation_catalog_with_bootstrapping(df_test, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats)
    df_test["Dataset"] = "Test"
    
    df_val = mutation_catalog_with_bootstrapping(df_val, drug, who_variants_df, isolate_variants_df, binary_thresh, return_stats)
    df_val["Dataset"] = "Validation"
    
    return pd.concat([df_train, df_test, df_val], axis=0).reset_index(drop=True)





def get_new_aln_for_regression(isolate_order,
                               locus_list,
                               results_dir,
                               fasta_dir
                              ):
                                  
    # just need these to get the full alignment length
    reduced_fastas = glob.glob(os.path.join(results_dir, "ridge", "*_reduced.fasta"))
    
    # Compute the total number of sites in our model by summing the length of all the alignment
    total_sites = 0
    
    for file in reduced_fastas:
        aln = Alignment.from_file(open(file))    
        total_sites += aln.L
        del aln
    
    total_seqs = len(isolate_order)
    print(f"Concatenation of {total_sites} nucleotides across {total_seqs} sequences")

    # Matrix to store the data for learning
    X = np.zeros((total_seqs, total_sites), dtype=np.int8)
    
    current_index = 0
    
    for locus in locus_list:

        if not os.path.isfile(os.path.join(fasta_dir, f"{locus}.fasta")):
            raise ValueError(f"{os.path.join(fasta_dir, f'{locus}.fasta')} does not exist!")
        
        aln = Alignment.from_file(open(os.path.join(fasta_dir, f"{locus}.fasta")), alphabet='-ACGT')
        indices_to_keep = np.load(os.path.join(results_dir, f"ridge/{locus}_indices.npy"))
        
        # only use sequence alignments with sites for the model. Otherwise get a vectorize error
        if aln.L != 0:

            # the fasta files contain sequences for all isolates. Keep only the isolates in the phenotypes file
            # need indices for splitting alignment in evcouplings
            # remove all possible file extensions
            keep_idx = [i for i, name in enumerate(aln.ids) if os.path.basename(name).replace(".eff", "").replace(".vcf", "") in isolate_order]
            assert len(keep_idx) == len(isolate_order)

            subset_alignment = aln.select(columns=indices_to_keep, sequences=keep_idx) 
            subset_alignment.ids = [os.path.basename(x).replace(".eff", "").replace(".vcf", "") for x in subset_alignment.ids]
            subset_alignment.id_to_index = {x:idx for idx,x in enumerate(subset_alignment.ids)}
            
            # Get the indices that would correctly reorder the alignment to match isolate_order
            reorder_index = [subset_alignment.id_to_index[x] for x in isolate_order if x in list(subset_alignment.id_to_index.keys())]

            if len(reorder_index) != len(list(subset_alignment.id_to_index.keys())):
                raise ValueError()
        
            # Reorder based on reorder_index
            subset_alignment.ids = np.array(subset_alignment.ids)[reorder_index]
            assert sum(isolate_order != subset_alignment.ids) == 0

            subset_alignment.matrix = subset_alignment.matrix[reorder_index, :]

            # Tells you which character is the most frequent in each site
            who_is_max = np.argmax(subset_alignment.frequencies, axis=1)
    
            # Major allele is encoded as 0, minor allele(s) as 1
            major_minor = subset_alignment.matrix_mapped != who_is_max
    
            # Add the encoding to the X matrix
            X[:, current_index:(current_index + major_minor.shape[1])] = major_minor
    
            # Keep track of how many sites in X we have filled in
            current_index = current_index + major_minor.shape[1]

    return X



def get_new_aln_for_CNN(df,
                        locus_list,
                        fasta_dir
                       ):
    
    # argument = directory that contains the fasta file
    df_genos = make_genotype_df(locus_list, fasta_dir)
    df_genos.index = [name.replace("-", "_").split(".")[0] for name in df_genos.index]
    
    df["ROLLINGDB_ID"] = [name.replace("-", "_") for name in df["ROLLINGDB_ID"]]
    
    # the additional new strains to predict MICs for
    df_genos = df_genos.loc[df["ROLLINGDB_ID"].values]
    
    assert len(df_genos) == len(df)

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[locus + "_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))
        
    return create_X(df_genos)




def get_inputs_for_regression(drug,
                              config_file,
                             ):

    kwargs = yaml.safe_load(open(config_file, "r"))
    
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    results_dir = kwargs["output_path"]
    fasta_dir = kwargs["genotype_input_directory"]
    include_lineage = kwargs["include_lineage"]
    
    df_train = pd.read_csv(kwargs["phenotype_file"]).query("category=='original_train_set'")    
    df_test = pd.read_csv(kwargs["phenotype_file"]).query("category=='original_test_set'")    
    df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))
    
    X = np.load(os.path.join(results_dir, "ridge", "combined_X.npy"))
    X_train = X[df_train.index.values, :]
    X_test = X[df_test.index.values, :]
    
    X_val = get_new_aln_for_regression(df_val["ROLLINGDB_ID"].values,
                                       locus_list,
                                       results_dir,
                                       fasta_dir
                                      )
        
    scaler = StandardScaler()
        
    train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)
    
    X_train = get_combined_model_inputs(X_train, train_lineages, "Regression", include_lineage)
    X_test = get_combined_model_inputs(X_test, test_lineages, "Regression", include_lineage)
    X_val = get_combined_model_inputs(X_val, val_lineages, "Regression", include_lineage)
        
    return X_train, X_test, X_val, df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val.reset_index(drop=True)




def get_inputs_for_CNN(drug,
                       config_file,
                      ):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    results_dir = kwargs["output_path"]
    fasta_dir = kwargs["genotype_input_directory"]
    include_lineage = kwargs["include_lineage"]

    binary_thresh = kwargs["binary_thresh"]
    loss_type = kwargs["loss_type"]
    binary = kwargs["binary"]
    bounded_loss = kwargs["bounded_loss"]
    
    df_train = pd.read_csv(os.path.join(data_dir, "data_for_model.csv")).query("category=='original_train_set'").reset_index(drop=True)    
    df_test = pd.read_csv(os.path.join(data_dir, "data_for_model.csv")).query("category=='original_test_set'").reset_index(drop=True)    
    df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))

    X_train = sparse.load_npz(os.path.join(results_dir, "pkl_sparse_train.npz")).todense()
    X_test = sparse.load_npz(os.path.join(results_dir, "pkl_sparse_test.npz")).todense()
    
    if not os.path.isfile(os.path.join(results_dir, "pkl_sparse_val.npz")):
        
        X_val = get_new_aln_for_CNN(df_val,
                                    locus_list,
                                    fasta_dir
                                   )
        sparse.save_npz(os.path.join(results_dir, "pkl_sparse_val.npz"), sparse.COO(X_val))
        
    else:
        X_val = sparse.load_npz(os.path.join(results_dir, "pkl_sparse_val.npz")).todense()

    train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)
    
    X_train = get_combined_model_inputs(X_train, train_lineages, "CNN", include_lineage)
    X_test = get_combined_model_inputs(X_test, test_lineages, "CNN", include_lineage)
    X_val = get_combined_model_inputs(X_val, val_lineages, "CNN", include_lineage)
        
    return X_train, X_test, X_val, df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val.reset_index(drop=True)
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

# BOTH ARE INCLUSIVE!!!
START = 1
END = 4411532

isolate_variants_df = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/isolate_variants.tsv", sep="\t")
output_file = f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/isolate_variants_fixed_annot.csv"

vcf_dir = "/n/scratch3/users/s/sak0914/annotated_VCF"

h37Rv_seq = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")

h37Rv_coords = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/h37Rv_coords_to_gene.csv")
h37Rv_coords.columns = ["POS", "Region"]
protein_coding_regions = h37Rv_coords.query("~Region.str.contains('NC_')").reset_index(drop=True)

h37Rv_genes_annot = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

def get_variants_each_gene(sample_id, isolate_variants_file):
    
    variants_in_each_gene = {key: [] for key in protein_coding_regions.query("POS >= @START & POS <= @END")["Region"].unique()}

    single_isolate_variants = isolate_variants_file.query("Isolate == @sample_id").reset_index(drop=True)
    
    for _, row in single_isolate_variants.iterrows():

        pos = int(row["POS"])

        if pos >= START and pos <= END and len(row["REF"]) == 1 and len(row["ALT"]) == 1:

            # check if the variant is in a protein-coding region
            if pos in protein_coding_regions["POS"].values:

                # get the corresponding gene
                assert len(protein_coding_regions.query("POS==@pos")["Region"].values) == 1
                gene = protein_coding_regions.query("POS==@pos")["Region"].values[0]

                variants_in_each_gene[gene].append(pos)
                
    return variants_in_each_gene




def fix_mutations_single_sample(sample_id, isolate_variants_df):
    
    # get list of variants in each gene for a given sample
    variants_in_each_gene = get_variants_each_gene(sample_id, isolate_variants_df)    
    
    single_sample_variants = isolate_variants_df.query("Isolate == @sample_id").reset_index(drop=True)

    no_AA_change = ['synonymous_variant', 'intergenic_region', 'intragenic_variant', 'upstream_gene_variant', 'downstream_gene_variant']

    for i, row in single_sample_variants.iterrows():
        if row["EFFECT"] in no_AA_change:
            single_sample_variants.loc[i, "mutation"] = row["GENE"] + "_" + row["NUC"]
        else:
            single_sample_variants.loc[i, "mutation"] = row["GENE"] + "_" + row["PROT"]
    
    for gene in variants_in_each_gene.keys():
        pos_to_fix = within_same_codon(variants_in_each_gene[gene])

        if len(pos_to_fix) > 0:
            
            # list to keep track of which codons have been done
            fixed_pos = np.array([])

            for _, row in single_sample_variants.iterrows():

                pos = row["POS"]
                
                # only change SNPs, not indels
                if row["FILTER"] == "PASS" and len(row["REF"]) == 1 and len(row["ALT"]) == 1:

                    if pos in pos_to_fix:

                        if pos not in fixed_pos:

                            gene = row["GENE"]
                            gene_start, gene_end, gene_sense = h37Rv_genes_annot.query("Symbol==@gene")[["Start", "End", "Strand"]].values[0]
                            
                            codon_num = int(((pos - gene_start) // 3) + 1)
                            codon_seq, codon_pos = get_codon_from_seq(h37Rv_seq.seq, codon_num, gene_start, gene_end, gene_sense)
                            ref_codon = str(codon_seq)
                            codon_seq = list(str(codon_seq))

                            mutated_codon_df = single_sample_variants.query("POS in @codon_pos")[["POS", "REF", "ALT"]].reset_index(drop=True)

                            for k in range(len(codon_pos)):
                                if codon_pos[k] in mutated_codon_df.POS.values:
                                    ref, alt = mutated_codon_df.loc[mutated_codon_df["POS"]==codon_pos[k], ["REF", "ALT"]].values[0]
                                    
                                    if gene_sense == "-":
                                        ref = reverse_complement(ref)
                                        alt = reverse_complement(alt)
                                    
                                    # assert ref == codon_seq[k]
                                    if ref != codon_seq[k]:
                                        print(sample_id, row["POS"], ref, codon_seq[k])

                                    codon_seq[k] = alt

                            mutated_codon = "".join(codon_seq)

                            for pos in codon_pos:
                                if pos in single_sample_variants["POS"].values:

                                    # only make the updates for SNPs. Ignore indels that occur on the same codon as other SNPs
                                    if len(single_sample_variants.query("POS==@pos")["REF"].values[0]) == 1 and len(single_sample_variants.query("POS==@pos")["ALT"].values[0]) == 1:
                                    
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

print(f"Fixing annotations for {len(isolate_variants_df['Isolate'].unique())} samples in the range [{START},{END}] and saving to {output_file}")
for i, sample_id in enumerate(isolate_variants_df["Isolate"].unique()):
    
    if len(isolate_variants_df.query("Isolate==@sample_id").loc[~pd.isnull(isolate_variants_df["POS"])]) == 0:

        nan_df = pd.DataFrame(pd.Series([np.nan]*(len(isolate_variants_df.columns)-1) + [sample_id])).T
        nan_df.columns = isolate_variants_df.columns
        df_new_lst.append(nan_df)

    else:
        df_new_lst.append(fix_mutations_single_sample(sample_id, isolate_variants_df))
    
    if i % 500 == 0:
        print(i)
        pd.concat(df_new_lst, axis=0).to_csv(output_file, index=False)

pd.concat(df_new_lst, axis=0).to_csv(output_file, index=False)
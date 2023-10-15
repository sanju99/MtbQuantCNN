import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle

from Bio import SeqIO, Seq

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
from inSilicoMut_utils import *

h37Rv_seq = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/mycobrowser_h37rv_genes_v4.csv")


_, drug = sys.argv


df = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/isolate_variants.csv.gz", compression="gzip")
print(df.shape)

ref_isolates = df.loc[pd.isnull(df["POS"])].Isolate.unique()

# remove isolates with no variants in them
df = df.loc[~pd.isnull(df["POS"])]

df.loc[df["FILTER"]=="PASS", "QC"] = 1
df.loc[df["FILTER"]=="Amb", "QC"] = df.loc[df["FILTER"]=="Amb"]["AF"].values.astype(float)

# the actual value will not be -1, it will be NaN. But when the df is pivoted, the NaNs need to be filled with 0 (meaning not present)
# then, the -1 values will be converted to NaN to mean missing. 
df["QC"] = df["QC"].fillna(-1)

df["REF_len"] = [len(val) for val in df["REF"].values]
df["ALT_len"] = [len(val) for val in df["ALT"].values]

df_SNP = df.query("REF_len == 1 & ALT_len==1").reset_index(drop=True)
df_indel = df.query("REF_len != 1 | ALT_len != 1").reset_index(drop=True)


def add_mutation_column(df):

    # combine GENE with NUC for these. for others, combine GENE with PROT
    nonmissense_types = ["synonymous_variant", "intergenic_region", "initiator_codon_variant", "start_lost", "stop_gained"]

    df.loc[(df["EFFECT"].isin(nonmissense_types)) | (df["PROT"]=="."), "mutation"] = df.loc[(df["EFFECT"].isin(nonmissense_types)) | (df["PROT"]==".")]["GENE"] + "_" + df.loc[(df["EFFECT"].isin(nonmissense_types)) | (df["PROT"]==".")]["NUC"]

    df.loc[pd.isnull(df["mutation"]), "mutation"] = df.loc[pd.isnull(df["mutation"])]["GENE"] + "_" + df.loc[pd.isnull(df["mutation"])]["PROT"]

    return df


df_SNP = add_mutation_column(df_SNP)
df_indel = add_mutation_column(df_indel)


# check for duplicates that would make the pivot function fail
df_needs_fix = df_SNP.iloc[df_SNP.index.values[df_SNP.duplicated(["Isolate", "mutation", "QC"], keep=False)]].reset_index(drop=True)
df_no_fix = df_SNP.iloc[df_SNP.index.values[~df_SNP.duplicated(["Isolate", "mutation", "QC"], keep=False)]].reset_index(drop=True)
df_needs_fix.shape, df_no_fix.shape
print(f"Fixing {len(df_needs_fix)} SNPEff annotations across {len(df_needs_fix.Isolate.unique())} isolates")



def fix_cooccurring_SNPs(single_sample_variants):

    sample_id = single_sample_variants.Isolate.unique()
    pos_to_fix = within_same_codon(single_sample_variants.POS.values)

    if len(pos_to_fix) > 0:
        
        # list to keep track of which codons have been done
        fixed_pos = np.array([])
    
        for _, row in single_sample_variants.iterrows():
    
            pos = row["POS"]
            
            if pos in pos_to_fix:
    
                if pos not in fixed_pos:
    
                    gene = row["GENE"]
                    gene_start, gene_end, gene_sense = h37Rv_genes.query("Symbol==@gene")[["Start", "End", "Strand"]].values[0]
    
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
                            
                            assert ref == codon_seq[k]
    
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



# fix SNPEff annotations for SNPs only
df_fixed = []
for isolate in df_needs_fix.Isolate.unique():

    single_sample_variants = df_needs_fix.loc[df_needs_fix["Isolate"]==isolate]

    single_sample_variants_fixed = fix_cooccurring_SNPs(single_sample_variants)

    df_fixed.append(single_sample_variants_fixed.drop_duplicates(["Isolate", "QC", "mutation"]))


# combine the fixed df with the df of SNPs that didn't need to be fixed and the indels dataframe and save to a new dataframe
df_SNP_fixed = pd.concat([df_no_fix] + df_fixed, axis=0)
assert len(set(df_SNP_fixed["Isolate"]).symmetric_difference(df_SNP["Isolate"])) == 0

df_final = pd.concat([df_indel, df_SNP_fixed])
print(df_final.shape)
df_final.to_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/isolate_variants_FULL_fixed.tsv", sep="\t", index=False)
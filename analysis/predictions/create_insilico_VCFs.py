import pandas as pd
import numpy as np
import glob, os, yaml, itertools, subprocess, sys, argparse, shutil

import Bio.SeqUtils
import Bio.Data
from Bio import SeqIO
from Bio.Seq import Seq
import warnings, pickle
warnings.filterwarnings("ignore")

# load all utils functions
who_variants_V2 = pd.read_csv("./data_processing/data_utils/WHO_catalog_V2.csv", header=[2]).reset_index(drop=True)
who_variants_V1 = pd.read_csv("./data_processing/data_utils/WHO_catalog_V1.csv")

sys.path.append("utils")
from data_utils import *
from inSilicoMut_utils import *

drug_loci = pd.read_csv("./data_processing/data_utils/drug_loci.csv")

results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"
data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

coll_2014 = pd.read_csv("/home/sak0914/who-analysis/data/coll2014_SNP_scheme.tsv", sep="\t")
coll_2014["lineage"] = coll_2014["#lineage"].str.replace("lineage", "")
del coll_2014["#lineage"]

lineages_matrix = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv", index_col=[0])
amino_acid_biophysical_properties = pd.read_csv("./data_processing/protein_seqs/biophysical_properties_AA.csv", index_col=[0])

h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
h37Rv_regions = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_v4.csv")
h37Rv_genes = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")
h37Rv_coords_to_gene = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/h37Rv_coords_to_gene.csv")
h37Rv_coords_to_gene_dict = dict(zip(h37Rv_coords_to_gene['pos'], h37Rv_coords_to_gene['region']))

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


parser = argparse.ArgumentParser()

# Add a required string argument for the drug
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

# boolean argument for including tier 2 loci (also encoded as NT sequences), default value False. If you include the flag, it is considered True
parser.add_argument('--tier2', action='store_true', help='Flag to add tier 2 loci to in silico mutagenesis (not used for saturation mutagenesis)')

# Add an optional argument for site-saturation mutagenesis
parser.add_argument('--saturation-muts', dest='saturation_muts', action='store_true', help='Get MIC predictions for insilico mutations')

# Add an optional string argument for the locus for which to perform site-saturation mutagenesis. Because there are so many variants, don't do it by default for all genes
parser.add_argument("--gene", type=str, help='Specify the gene (not locus) for which to perform site-saturation mutagenesis. This is only used if --saturation-muts is specified')

parser.add_argument('--model_suffix', dest='model_suffix', help='If specified, use the {drug}_{model_suffix} directory. Must be one of "binary" or "augment" if specified.')

parser.add_argument('--nonsense', dest='nonsense_only', action='store_true', help='If True, only generate synthetic sequences for nonsense mutations')


cmd_line_args = parser.parse_args()
config_file = cmd_line_args.config_file
include_tier2 = cmd_line_args.tier2
gene = cmd_line_args.gene
saturation_muts = cmd_line_args.saturation_muts
model_suffix = cmd_line_args.model_suffix
assert model_suffix in ['binary', 'augment', None]
nonsense_only = cmd_line_args.nonsense_only



def create_WHO_catalog_insilico_files(drug, out_dir, include_tier2=False):

    # get variants from only the 2nd version of the catalog
    drug_full_name = abbr_drug_dict[drug]
    who_variants = who_variants_V2.copy().query("drug==@drug_full_name")
        
    df_variants = get_VCF_file_information(drug,
                                           tier2=include_tier2,
                                           V1=False, # V1 (True) or V2 (False)
                                          )

    # remove extra columns from the V2 catalog to reduce clutter
    df_variants = df_variants[['variant', 'gene', 'mutation', 'effect', 'FINAL CONFIDENCE GRADING', 'POS', 'REF', 'ALT']].rename(columns={'FINAL CONFIDENCE GRADING': 'confidence'})

    # then write them to new VCF files
    # this also creates a file of the mutation names
    create_synthetic_VCF_files(df_variants, 
                               os.path.join(out_dir, "WHO_mutations.txt"),
                               vcf_dir
                          )
    
    # save a dataframe of the variants and the POS, REF, and ALT fields
    df_variants.to_csv(os.path.join(out_dir, "WHO_nucleotide_variants.csv"), index=False)
    print(f"{len(df_variants)} total WHO catalog mutations")

    # then run this for annotation
    print(f"\nPlease run\n    snpEff eff Mycobacterium_tuberculosis_h37rv -noStats -fileList -no-downstream -no-upstream {os.path.join(out_dir, 'WHO_mutations.txt')}\n")

    # subprocess.run(f"snpEff eff Mycobacterium_tuberculosis_h37rv -noStats -fileList -no-downstream -no-upstream {os.path.join(out_dir, 'WHO_mutations.txt')}", shell=True, executable='/bin/bash')
    
    return df_variants




def create_synthetic_VCF_saturation_mutagenesis(drug, gene, out_dir, nonsense_only=False):
    '''
    This creates synthetic VCF files for all possible amino acid substitutions at each site, including eacrly stop codons
    '''

    if nonsense_only:
        print(f"Creating nonsense mutations at every codon in {gene}")
    else:
        print(f"Creating all possible amino acid substitutions in {gene}")

    # this can only be done for protein-coding genes, so use the genes-only table
    gene_start, gene_end, gene_sense = h37Rv_genes.query("Symbol==@gene")[['Start', 'End', 'Strand']].values[0]

    if gene_sense == '+':
        protein_seq = h37Rv.seq[gene_start-1:gene_end].translate()
    else:
        protein_seq = h37Rv.seq[gene_start-1:gene_end].reverse_complement().translate()
    
    assert protein_seq[0] in ['M', 'I', 'V', 'L']
    assert protein_seq[-1] == '*'
    assert list(str(protein_seq)).index('*') == len(protein_seq) - 1
    
    # remove stop codon
    protein_seq = str(protein_seq)[:-1]
    
    aa_to_codon_table = get_aa_to_codon_table()
    
    # needs variant and mutation columns. variant = {gene}_{mutation}
    df_site_saturation_mutagenesis = pd.DataFrame(columns=['gene', 'mutation'])
    
    i = 0
    
    for aa_pos, aa in enumerate(protein_seq):
    
        aa_three_letter = Bio.SeqUtils.IUPACData.protein_letters_1to3[aa]
    
        # iterate through the other 19 amino acids and the stop character at each site
        remaining_aa = list(set(aa_to_codon_table.AA.unique()) - set([aa_three_letter]))
        assert len(remaining_aa) == 20
        
        for mut_aa in remaining_aa:

            if nonsense_only:
                if mut_aa == '*':
                    df_site_saturation_mutagenesis.loc[i, :] = [gene, f"p.{aa_three_letter}{aa_pos+1}{mut_aa}"]
                    i += 1
            else:
                df_site_saturation_mutagenesis.loc[i, :] = [gene, f"p.{aa_three_letter}{aa_pos+1}{mut_aa}"]
                i += 1

    if nonsense_only:
        assert len(df_site_saturation_mutagenesis) == len(protein_seq)
    else:
        assert len(df_site_saturation_mutagenesis) == 20*len(protein_seq)
        
    df_site_saturation_mutagenesis['variant'] = df_site_saturation_mutagenesis['gene'] + '_' + df_site_saturation_mutagenesis['mutation']
    df_site_saturation_mutagenesis['drug'] = drug
    
    df_site_saturation_mutagenesis_variants = get_data_for_synthetic_VCF(df_site_saturation_mutagenesis)

    # save
    if not os.path.isdir(f"{out_dir}/{gene}"):
        os.mkdir(f"{out_dir}/{gene}")
        
    df_site_saturation_mutagenesis_variants.to_csv(f"{out_dir}/{gene}_nucleotide_variants.csv", index=False)
    
    create_synthetic_VCF_files(df_site_saturation_mutagenesis_variants, 
                               f"{out_dir}/{gene}/{gene}_mutations.txt",
                               vcf_dir,
                              )

    print(f"\nPlease run\n    snpEff eff Mycobacterium_tuberculosis_h37rv -noStats -fileList -no-downstream -no-upstream {out_dir}/{gene}/{gene}_mutations.txt\n")

    return df_site_saturation_mutagenesis_variants




def remove_mutations_to_preserve_aln(model_aln_df, full_aln_df, START, vcf_df):
    
    print(f"Original aln insertions: {model_aln_df['len_insertion'].sum()}")
    print(f"In silico aln insertions: {full_aln_df['len_insertion'].sum()}")
    
    model_aln_df["name"] = "old"
    full_aln_df["name"] = "new"

    insertions_combined = model_aln_df.merge(full_aln_df, on="aln_idx", how="outer")

    # idx + START + 1 = record.POS. Add 1 to START because it is 0-indexed
    insertions_combined["POS"] = insertions_combined["aln_idx"] + START + 1
    
    # x = old, y = new
    search_df = insertions_combined.loc[insertions_combined['len_insertion_x'] != insertions_combined['len_insertion_y']]#insertions_combined.loc[pd.isnull(insertions_combined["name_x"])]

    vcf_df["REF_len"] = [len(val) for val in vcf_df["REF"]]
    vcf_df["ALT_len"] = [len(val) for val in vcf_df["ALT"]]

    drop_mutations = vcf_df.query("POS in @search_df.POS & ALT_len > REF_len")[["confidence", "mutation", "POS", "REF", "ALT"]].mutation.values
    print(drop_mutations)
    print(f"Dropped {len(drop_mutations)} mutations to preserve the alignment")

    vcf_df_cleaned = vcf_df.query("mutation not in @drop_mutations")
    
    return vcf_df_cleaned, len(drop_mutations)
    
    
    
def remove_mutations_to_preserve_aln_create_new_files(drug, locus, out_dir):

    START = drug_loci.query("Locus==@locus")['Start'].values[0]

    vcf_files_fName = f"{out_dir}/{locus}/WHO_mutations.txt"
    vcf_files_fName_original = f"{out_dir}/{locus}/WHO_mutations_original.txt"

    # all nucleotide variants are one level up because this is a master dataframe for all of them
    nucleotide_vars_fName = f"{out_dir}/WHO_nucleotide_variants.csv"
    nucleotide_vars_fName_original = f"{out_dir}/WHO_nucleotide_variants_original.csv"

    # this is the name of the final file, but currently it is the original file. It will be renamed
    insertion_sites_fName = f"{fasta_dir}/{locus}_insertion_sites.csv"

    # original file when all variants were included
    insertion_sites_fName_original = f"{fasta_dir}/{locus}_insertion_sites_original.csv"
    
    # original dataframe of insertions when the original data were aligned before training
    model_aln_fName = os.path.join(drug_data_dir, "fastas", f"{locus}_insertion_sites.csv")

    print(f"insertion_sites_fName: {insertion_sites_fName}")
    print(f"insertion_sites_fName_original: {insertion_sites_fName_original}")
    print(f"insertion_sites_fName_original_training: {model_aln_fName}")

    model_aln_df = pd.read_csv(model_aln_fName)

    # isolates above + WHO mutations for insilico validation
    # insilico_aln_df = pd.read_csv(os.path.join(data_dir, drug, "inSilico_analysis", locus, "fastas", f"{locus}_insertion_sites.csv"))
    insilico_aln_df = pd.read_csv(insertion_sites_fName)

    # change the name of the insertion sites file so that the new one will appear when make_MSA.py is rerun
    if os.path.isfile(insertion_sites_fName) and not os.path.isfile(insertion_sites_fName_original):
        os.rename(insertion_sites_fName, insertion_sites_fName_original)

    variants_df = pd.read_csv(nucleotide_vars_fName)
    vcf_df_cleaned, num_dropped_mutations = remove_mutations_to_preserve_aln(model_aln_df, insilico_aln_df, START, variants_df)

    # change the file name to keep the original
    if os.path.isfile(nucleotide_vars_fName) and not os.path.isfile(nucleotide_vars_fName_original):
        os.rename(nucleotide_vars_fName, nucleotide_vars_fName_original)

    # save the cleaned file to the original file name
    vcf_df_cleaned.to_csv(nucleotide_vars_fName, index=False)

    fNames_lst = pd.read_csv(vcf_files_fName, sep="\t", header=None).rename(columns={0: 'fName'})

    # change the file name to keep the original
    if os.path.isfile(vcf_files_fName) and not os.path.isfile(vcf_files_fName_original):
        os.rename(vcf_files_fName, vcf_files_fName_original)

    # revert to original names, which is what's in vcf_df_cleaned. The different names are due to special character problems for file names
    fNames_lst["mutation"] = [os.path.basename(val.split(".")[0].replace('_p_', '_p.').replace('_c_', '_c.').replace('_n_', '_n.').replace('+', '*')) for val in fNames_lst['fName'].values]
    
    # save the full paths to the new file so that it can be re-aligned with the training data
    fNames_lst.loc[fNames_lst["mutation"].isin(vcf_df_cleaned["variant"].values)]['fName'].to_csv(vcf_files_fName, index=False, header=None, sep="\t")

    # keep only those in the WHO nucleotide variants dataframe
    print(f"Kept {len(fNames_lst.loc[fNames_lst['mutation'].isin(vcf_df_cleaned['variant'].values)])} mutations for {drug}")

    # only rerun the alignment if some mutations needed to be dropped
    # if num_dropped_mutations > 0:

    # actually it's unreliable sometimes (not sure what the bug is), so to be safe, re-align the sequences after dropping mutations that increase the aln size, which is occurring correctly
    subprocess.run(f"bash {out_dir}/bash_scripts/{locus}.sh", shell=True)
    



def write_ref_seqs_for_constant_loci(drug, variable_locus, constant_loci_list, out_dir):

    # these aren't genes, so nothing will be returned
    if 'rrs' in variable_locus or 'rrl' in variable_locus:
        variable_genes_lst = [variable_locus]
    else:
        # get the individual genes in the locus. This is because the in silico variants are named at the gene level
        # i.e. gyrA_p.Asp94Gly rather than gyrBA_p.Asp94Gly
        variable_genes_lst = get_genes_lst([variable_locus])
    
    # insilico mutated sequences
    seq_with_variants = [seq.id for seq in SeqIO.parse(f"{out_dir}/{variable_locus}.fasta", "fasta")]

    for constant_locus in constant_loci_list:
    
        # get the reference sequence for the locus
        original_fasta_fName = f"{drug_data_dir}/fastas/{constant_locus}.fasta"
        print(f"original_fasta_fName: {original_fasta_fName}")
            
        locus_seq = [(seq.id, seq.seq) for seq in SeqIO.parse(original_fasta_fName, "fasta")]
        ref_seq = str(locus_seq[-1][1])
        print(constant_locus, len(ref_seq))
        
        with open(f"{out_dir}/{constant_locus}.fasta", "w+") as file:
            
            for seq in seq_with_variants:

                found_variant_name = False

                for variable_gene in variable_genes_lst:

                    # only write the sequences that have a gene name in the id
                    # for insilico mutations, the id is like gyrA_p_Asp94Gly, whereas isolates will be i.e. SAMN...
                    # edge case: because the regions for Rv0678 and mmpS5-mmpL5 overlap (they are separate loci because they have different strand sense), some variants will be in both regions because they are in the promoter region, which is the same for both
                    if variable_gene in seq or 'mmp' in seq or 'Rv0678' in seq:
                        file.write(">" + seq + "\n")
                        file.write(ref_seq + "\n")
                        break

                    # only write the sequences that have a gene name in the id
                    # for insilico mutations, the id is like gyrA_p_Asp94Gly, whereas isolates will be i.e. SAMN...
                    # edge cases for ethA-ethR and mmpS5-mmpL5-Rv0678: 
                    # because the regions for Rv0678 and mmpS5-mmpL5 overlap (they are separate loci because they have different strand sense), some variants will be in both regions because they are in the promoter region, which is the same for both
                    overlapping_regions_to_check = ['mmpL5', 'mmpS5', 'Rv0678', 'ethA', 'ethR']
                    
                    if variable_gene in seq:
                        found_variant_name = True
                        break
                    else:
                        for gene_name in overlapping_regions_to_check:
                            if gene_name in seq:
                                found_variant_name = True
                                break

                # also include MT_H37Rv because need it for getting locus lengths
                if found_variant_name or "MT_H37Rv" in seq:
                    file.write(">" + seq + "\n")
                    file.write(ref_seq + "\n")



# need to figure out which loci need to be re-aligned (i.e. which loci have in silico mutations in them)
kwargs = yaml.safe_load(open(config_file, "r"))

drug = kwargs['drug']

if model_suffix is not None:
    out_dir = f"{data_dir}/{drug}_{model_suffix}"
else:
    out_dir = f"{data_dir}/{drug}"

# this is for the original data used for training, not the source of the additional predictions files
drug_data_dir = out_dir
print(f"drug_data_dir: {drug_data_dir}")

if saturation_muts:
    out_dir = f"{out_dir}/inSilico_analysis/saturation_mutagenesis"
    assert gene is not None # must be specified
else:
    out_dir = f"{out_dir}/inSilico_analysis"

    # if it exists, delete and make a new one. But only for insilico-muts because all loci are done together.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    for fName in ['WHO_mutations.txt', 'WHO_nucleotide_variants.csv', 'WHO_nucleotide_variants_original.csv']:
        if os.path.isfile(f"{drug_data_dir}/inSilico_analysis/{fName}"):
            print(f"Deleting {drug_data_dir}/inSilico_analysis/{fName}")
            os.remove(f"{drug_data_dir}/inSilico_analysis/{fName}")

vcf_dir = f"{out_dir}/synthetic_VCF"

# this will make all subdirectories
if not os.path.isdir(vcf_dir):
    os.makedirs(vcf_dir)


##################################################### STEP 1: CREATE VCF FILES FOR ALL VARIANTS FROM BOTH V1 AND V2 EDITIONS OF THE WHO MUTATION CATALOG #####################################################

    
# WHO catalog mutagenesis
if saturation_muts:

    if gene not in drug_loci.Locus.values:
        
        locus_list = drug_loci.query("Locus.str.contains(@gene)").Locus.values

        # if the string search above fails, assign locus list manually
        if len(locus_list) == 0:
            if gene in ['mmpS5', 'mmpL5']:
                locus_list = ['mmpLS5']

            elif gene in ['gyrA', 'gyrB']:
                locus_list = ['gyrBA']

            elif gene in ['rpoB', 'rpoC']:
                locus_list = ['rpoBC']
                
        assert len(locus_list) == 1
    else:
        locus_list = [gene]

    # this will print a command to run snpEff on the VCF files to check annotations
    if locus_list[0] == 'mmpLS5':
        df_variants = create_synthetic_VCF_saturation_mutagenesis(drug, 'mmpS5', locus_list[0], out_dir, nonsense_only=nonsense_only)
        df_variants = create_synthetic_VCF_saturation_mutagenesis(drug, 'mmpL5', locus_list[0], out_dir, nonsense_only=nonsense_only)
    else:
        df_variants = create_synthetic_VCF_saturation_mutagenesis(drug, gene, out_dir, nonsense_only=nonsense_only)

    df_variants = pd.read_csv(f"{out_dir}/{gene}_nucleotide_variants.csv")

else:
    # this will print a command to run snpEff on the VCF files to check annotations
    df_variants = create_WHO_catalog_insilico_files(drug, out_dir, include_tier2=include_tier2)

    locus_list = kwargs['tier1_loci']

    if include_tier2:
        locus_list += kwargs['tier2_loci']

    print(f"Making in silico mutations for {','.join(locus_list)}")

    df_variants = pd.read_csv(f"{out_dir}/WHO_nucleotide_variants.csv")
        
    
# check that everything actually is a variant. If REF = ALT, then it's a new case that I haven't considered
if len(df_variants.query("REF == ALT")) > 0:
    print(df_variants.query("REF == ALT"))
    raise ValueError("Some variants do not have a different allele in the ALT column!")

proceed = input("Please press Enter when you have finished running snpEff to annotate the synthetic VCF files")

if proceed.lower() == "":

    if saturation_muts:
        annotated_VCFs = glob.glob(f"{vcf_dir}/{gene}*.eff.vcf")
        nonannotated_VCFs = glob.glob(f"{vcf_dir}/{gene}*.vcf")
    else:
        annotated_VCFs = glob.glob(f"{vcf_dir}/*.eff.vcf")
        nonannotated_VCFs = glob.glob(f"{vcf_dir}/*.vcf")
        
    delete_VCFs = list(set(nonannotated_VCFs)-set(annotated_VCFs))
    
    if len(annotated_VCFs) * 2 != len(nonannotated_VCFs) or len(annotated_VCFs) == 0:
        raise ValueError("Please run snpEff annotations first!")
    
    # remove the unannotated VCFs to save space
    for fName in delete_VCFs:
        os.remove(fName)

    # create a text file of full paths to the synthetic VCF files
    if saturation_muts:
        pd.Series(annotated_VCFs).to_csv(os.path.join(out_dir, gene, f'{gene}_mutations.txt'), index=False, header=None)
    else:
        pd.Series(annotated_VCFs).to_csv(os.path.join(out_dir, 'WHO_mutations.txt'), index=False, header=None)            
        
    
##################################################### STEP 2: INSPECT THE ANNOTATED FILES TO MAKE SURE THE ANNOTATIONS MAKE SENSE #####################################################
        

if saturation_muts:
    df_variants = pd.read_csv(f"{out_dir}/{gene}_nucleotide_variants.csv")
else:
    df_variants = pd.read_csv(f"{out_dir}/WHO_nucleotide_variants.csv")

non_matching_variants = []

for fName in glob.glob(f"{vcf_dir}/*.eff.vcf"):

    # returns all the annotations, both in nucleotide and amino acid space
    results = check_annotation_matches_fName(fName)
    found = False
    
    if '_p_' in fName:
        for possibility in results:
            # check that the variant name in the file name is one of the annotations
            # for start lost mutations, if it's an alternative start codon, then it will say p.Val1? or p.Ile? even though the first codon codes for a Met
            if '_'.join(os.path.basename(fName).split(".")[0].split('_')[1:]) in [possibility.replace('.', '_').replace('*', '+'), 'p_Val1?', 'p_Ile1?', 'p_Leu1?']:
                found = True

        if not found:
            non_matching_variants.append(os.path.basename(fName).split(".")[0].replace('_p_', '_p.').replace('+', '*'))

# should be POS = 763969. This variant is 597C>T
if len(non_matching_variants) > 0:
    print(df_variants.query("variant in @non_matching_variants"))
    print("You can individually check the VCF files and their annotations using the check_annotation_matches_fName function, which takes the full path to the VCF file as the only argument")
else:
    print("There are no mutations whose snpEff annotations do not match their file names")
        

##################################################### STEP 3: ALIGN THE ANNOTATED VCFS TO THE TRAINING DATA TO GET NUCLEOTIDE ALNS #####################################################


# only get alignments for loci that have in silico mutations in them
for locus in locus_list:

    locus_start, locus_end = drug_loci.query("Locus==@locus")[['Start', 'End']].values[0]

    if saturation_muts:
        fasta_dir = f"{out_dir}/{gene}/fastas"    
    else:
        fasta_dir = f"{out_dir}/{locus}/fastas"

    print(f"fasta_dir: {fasta_dir}")

    # check if there are in silico mutations in the locus region
    if len(df_variants.query("POS > @locus_start & POS <= @locus_end")) > 0:

        # create separate text files for the individual loci for ease later. All variants will be in the master WHO_mutations.txt file though
        single_locus_mutations = df_variants.query("POS > @locus_start & POS <= @locus_end")

        # make a separate subdirectory for each locus because that one will be variable, and the rest of the loci in the model will be constant (H37Rv ref seq) for predictions
        if not os.path.isdir(fasta_dir):
            os.makedirs(fasta_dir)

        if saturation_muts:

            # all mutations will be aligned, but only those with variants in the region of interest will have any changes to the alignment (the others will just be MT_H37Rv ref seq)
            if model_suffix is not None:
                subprocess.run(f"python3 data_processing/03_get_seq_alns.py -c config_files/config_{drug.lower()}.yaml --saturation-muts --locus {locus} --gene {gene} --model_suffix {model_suffix}", shell=True)
            else:
                subprocess.run(f"python3 data_processing/03_get_seq_alns.py -c config_files/config_{drug.lower()}.yaml --saturation-muts --locus {locus} --gene {gene}", shell=True)
        
        else:
            
            with open(f"{out_dir}/{locus}/WHO_mutations.txt" , "w+") as file:
                
                for variant in single_locus_mutations.variant.values:
                    
                    fName = f"{vcf_dir}/{variant.replace('.', '_').replace('*', '+')}.eff.vcf"
                    assert os.path.isfile(fName)
                    file.write(f"{fName}\n")

            if model_suffix is not None:
                # all mutations will be aligned, but only those with variants in the region of interest will have any changes to the alignment (the others will just be MT_H37Rv ref seq)
                subprocess.run(f"python3 data_processing/03_get_seq_alns.py -c config_files/config_{drug.lower()}.yaml --insilico-muts --locus {locus} --model_suffix {model_suffix}", shell=True)
            else:
                subprocess.run(f"python3 data_processing/03_get_seq_alns.py -c config_files/config_{drug.lower()}.yaml --insilico-muts --locus {locus}", shell=True)


##################################################### STEP 4: REMOVE MUTATIONS THAT INCREASE THE ALIGNMENT LENGTH #####################################################


        # this is not needed for site-saturation mutagenesis because there are no indels
        if not saturation_muts:
            # this works iteratively on multiple loci
            remove_mutations_to_preserve_aln_create_new_files(drug, locus, out_dir)


##################################################### STEP 5: CREATE INPUT FILES (ALL SAME SEQUENCES) FOR THE REMAINING MODEL LOCI #####################################################

        
        constant_loci_list = list(set(kwargs['tier1_loci'] + kwargs['tier2_loci']) - set([locus]))

        print(locus, constant_loci_list)
        
        # write files for all other loci, both tiers in the directory for each variable locus
        write_ref_seqs_for_constant_loci(drug, locus, constant_loci_list, out_dir=fasta_dir)
import pandas as pd
import numpy as np
import glob, os, yaml, itertools, subprocess, sys, shutil, sparse, re, vcf, warnings, argparse
warnings.filterwarnings("ignore")

import Bio.SeqUtils
import Bio.Data
from Bio import Seq, SeqIO, Entrez

# dataframe of all regions in H37Rv and their coordinates. Includes genes and non-coding regions
h37Rv_full = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_v4.csv")

################################################################## NOTES ##################################################################

# initiator_codon_variant and stop_retained_variant are encoded differently in snpEff than in the catalog, but they're not Group 1-2, so ignore
# silent variants composed of MNVs are formatted like this in the catalog: katG_c.CTCinsGAT, whereas the combineCodons function outputs katG_c.CTC>GAT

# this includes LoF variants -- i.e. Rv0678_LoF is significantly associated with Bedaquiline resistance, so all component LoF mutations (frameshift, stop gained, start lost) are Group 2) Assoc w R - Interim
who_catalog_V2 = pd.read_csv("/home/sak0914/MtbQuantCNN/data_processing/data_utils/WHO_catalog_V2.csv", header=[2])
who_catalog_V2 = who_catalog_V2[['drug', 'variant', 'FINAL CONFIDENCE GRADING', 'Comment']].rename(columns={'FINAL CONFIDENCE GRADING': 'confidence'})

parser = argparse.ArgumentParser()

# Add a required string argument for the input file (annotations)
parser.add_argument("-i", "--input", dest='input_file', type=str, required=True)

cmd_line_args = parser.parse_args()
input_file = cmd_line_args.input_file


def split_annotations(df):
    '''
    Function to split snpEff annotations that affect multiple genes 
    '''
    
    df_annotation = df['ANN'].str.split(',', expand=True)

    # keep the other columns separate and concatenate later
    df_split_combine = df[[col for col in df.columns if col != 'ANN']]
    
    df_ann_combine = []
    
    for i, col in enumerate(df_annotation.columns):
        
        df_ann_sep = df_annotation[col].str.split('|', expand=True)
    
        df_ann_sep.columns = [
            "Allele",
            "EFFECT",
            "Annotation",
            "GENE",
            "GENEID",
            "FeatureType",
            "FeatureId",
            "Biotype",
            "Rank",
            "HGVS_C",
            "HGVS_P",
            "cDNA_len",
            "CDS_len",
            "Protein_len",
            "Distance",
            ] + [f"Warning{num+1}" for num in np.arange(df_ann_sep.shape[1] - 15)] # the rest of the columns are warning columns
    
        if i == 0:
            df_ann_sep = df_ann_sep[['EFFECT', 'GENE', 'HGVS_C', 'HGVS_P']]
    
        else:
            df_ann_sep = df_ann_sep[['EFFECT', 'GENE']]
    
            # rename all other columns, but keep the first GENE and EFFECT columns as the original
            df_ann_sep.rename(columns={'GENE': f"GENE_{i}", 'EFFECT': f"EFFECT_{i}"}, inplace=True)
                        
        df_ann_combine.append(df_ann_sep)
    
    df_ann_combine = pd.concat(df_ann_combine, axis=1)
    df_split_combine = pd.concat([df_split_combine, df_ann_combine], axis=1)

    assert len(df_split_combine) == len(df)
    
    return df_split_combine
    
    
    
def convert_H37Rv_intragenic_coordinate(pos, gene):
    '''
    Use this only if POS in HGVS_C (so not a protein-changing variant)
    '''

    start, end, sense = h37Rv_full.query("Name==@gene")[['Start', 'Stop', 'Strand']].values[0]

    if type(pos) == str:
        pos = int(pos)

    assert pos >= start
    assert pos <= end

    if sense == '+':
        intragenic_coord = pos - start + 1
    else:
        intragenic_coord = end - pos + 1

    assert intragenic_coord >= 0
    return intragenic_coord




def convert_H37Rv_upstream_coordinate(pos, gene):
    '''
    Use this only if POS in HGVS_C (so not a protein-changing variant)
    '''

    start, end, sense = h37Rv_full.query("Name==@gene")[['Start', 'Stop', 'Strand']].values[0]

    if type(pos) == str:
        pos = int(pos)

    # must be intergenic, so should be outside the gene bounds
    if sense == '+':
        upstream_dist = start - pos
    else:
        upstream_dist = pos - end

    assert upstream_dist > 0
    return -upstream_dist



def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(seq))



def reverse_complement_variant(variant):

    # separate the gene from the rest of the variant so that no letters in the gene name are altered
    gene, mutation = variant.split('_', maxsplit=1)
    
    # Split the mutation into components using a regular expression
    # This pattern assumes nucleotide sequences are in uppercase letters
    components = re.split(r'([ATGC]+)', mutation)
    
    # Iterate through the components and reverse complement the nucleotide sequences
    for i, component in enumerate(components):
        if re.match(r'^[ATGC]+$', component):
            components[i] = reverse_complement(component)
    
    # Join the components back together
    reversed_mutation = ''.join(components)
    return gene + '_' + reversed_mutation
    


def process_intragenic_variant_WHO_catalog_coord(row):
    '''
    c = coding. p = protein. n = noncoding
    '''
    
    pos = row['POS']
    ref = row['REF']
    alt = row['ALT']
    effect = row['EFFECT']
    gene = row['GENE']
    nt_change = row['HGVS_C']
    prot_change = row['HGVS_P']

    if 'ablation' in effect:

        # multiple genes could be affected, so split the variant across them all
        # return comma-separated string for compatibility with later function
        return ','.join([single_gene + '_deletion' for single_gene in gene.split('&')])

    # initiator_codon_variants are listed as p.Met1? in the HGVS_P column by snpEff, so change those to the nucleotide change (so silent variants)
    # p.Met1? is only used for start lost mutations
    if 'initiator_codon_variant' in effect:
        return gene + '_' + nt_change
    
    # account for the alternative start codons. But in the catalog, they are all encoded as p.Met1? because the first amino acid in a protein is changed to Met after translation
    if 'start_lost' in effect:
        if 'p.Met1' in prot_change or 'p.Val1' in prot_change or 'p.Ile1' in prot_change or 'p.Leu1' in prot_change:
            return gene + '_' + 'p.Met1?'
        else:
            # if it's a frameshift on the first codon that also causes loss of the start codon, the variant should be the frameshift
            if 'frameshift' in effect:
                return gene + '_' + prot_change
            else:
                raise ValueError(f"{input_file}: p.Met1 not in variant at POS = {pos}")

    # protein-coding changes are easy
    if not pd.isnull(prot_change) and prot_change != '' and effect != 'synonymous_variant':
        return gene + '_' + prot_change
    else:
        # this occurs if the coordinate in the HGVS_C column is already the intragenic coordinate, instead of the H37Rv coordinate
        if str(pos) not in nt_change:
            return gene + '_' + nt_change

        # get intragenic coordinate for non-protein changing variants
        intragenic_coord = convert_H37Rv_intragenic_coordinate(pos, gene)

        # sometimes inconsistencies here, so change the prefix just to be safe
        if gene in ['rrs', 'rrl']:
            nt_change = nt_change.replace('c.', 'n.')
    
        # replace the H37Rv coordinate with the intragenic coordinate
        return gene + '_' + nt_change.replace(str(pos), str(intragenic_coord))
        
        
        
        
def process_intergenic_variant_WHO_catalog_coord(row):

    pos = row['POS']
    ref = row['REF']
    alt = row['ALT']
    nt_change = row['HGVS_C']
    
    gene_1, gene_2 = row['GENE'].split('-')

    # sometimes inconsistencies here, so change the prefix just to be safe
    if gene_1 in ['rrs', 'rrl'] or gene_2 in ['rrs', 'rrl']:
        letter_prefix = 'n'
    else:
        letter_prefix = 'c'
        
    # gene_1 is the gene with smaller coordinates. gene_2 is larger
    gene_1 == h37Rv_full.query("Stop <= @pos")['Name'].values[-1]
    
    # this means that CHR_END is in the gene name
    if len(h37Rv_full.query("Stop >= @pos")) == 0:
        gene_2 = gene_1
    else:
        gene_2 == h37Rv_full.query("Stop >= @pos")['Name'].values[0]

    # if negative sense, then the downstream gene is the one that comes earlier in H37Rv coordinate space
    gene_1_data = h37Rv_full.query("Name==@gene_1 | Locus==@gene_1")
    
    if len(gene_1_data) > 1:
        gene_1_data = gene_1_data.query("Feature=='CDS'")
        assert len(gene_1_data) == 1
        
    gene_2_data = h37Rv_full.query("Name==@gene_2 | Locus==@gene_2")
    
    if len(gene_2_data) > 1:
        gene_2_data = gene_2_data.query("Feature=='CDS'")
        assert len(gene_2_data) == 1
    
    sense_1 = gene_1_data['Strand'].values[0]
    sense_2 = gene_2_data['Strand'].values[0]

    # downstream of both genes, so probably doesn't affect their transcription
    if sense_1 == '+' and sense_2 == '-':
        return np.nan

    # first remove the letter prefix. Shouldn't be p. in here, but do it anyway
    nt_change = nt_change.replace('n.', '').replace('c.', '').replace('p.', '')

    # remove the position from the nt_change variable
    # the only numbers should be the coordinate
    str_pos = ''.join([char for char in nt_change if char.isdigit()])

    # variant is on a single position
    if str_pos in nt_change:
        
        nt_change = nt_change.replace(str_pos, '')

        # genes are in the same sense direction, so return upstream of the later gene
        if sense_1 == sense_2:

            if sense_1 == '+':
                primary_gene = gene_2
            else:
                primary_gene = gene_1
                
            new_coord = convert_H37Rv_upstream_coordinate(str_pos, primary_gene)

            if sense_1 == '+':
                return f"{primary_gene}_{letter_prefix}.{new_coord}{nt_change}"
            else:
                return reverse_complement_variant(f"{primary_gene}_{letter_prefix}.{new_coord}{nt_change}")
    
        # upstream of both genes, so need to return the variant relative to both genes
        elif sense_1 == '-' and sense_2 == '+':

            new_coord_gene_1 = convert_H37Rv_upstream_coordinate(str_pos, gene_1)
            new_coord_gene_2 = convert_H37Rv_upstream_coordinate(str_pos, gene_2)
            
            # return a comma-separated string with both mutations -- will split them into two lines later
            # only reverse complement the first variant because it's negative sense
            final_variant_1 = reverse_complement_variant(f"{gene_1}_{letter_prefix}.{new_coord_gene_1}{nt_change}")
            
            return f"{final_variant_1},{gene_2}_{letter_prefix}.{new_coord_gene_2}{nt_change}"
            
    # this means that there is a range of positions, and the variant needs to be of the form rpoA_c.-11_-4delATGCATGC
    else:
        # this will be the form for insertions, deletions, and duplications
        assert '_' in nt_change
        assert len(nt_change.split('_')) == 2
    
        # note that the nucleotide change will be on the second coordinate
        pos_1, pos_2 = nt_change.split('_')

        # then remove that underscore spacer between the two positions. Already verified that there's only one by checking that length of the split is 2
        nt_change = nt_change.replace('_', '')
    
        str_pos_1 = ''.join([char for char in pos_1 if char.isdigit()])
        str_pos_2 = ''.join([char for char in pos_2 if char.isdigit()])
    
        assert str_pos_1 in nt_change
        assert str_pos_2 in nt_change
    
        # replace both strings with the empty string
        nt_change = nt_change.replace(str_pos_1, '').replace(str_pos_2, '')

        if sense_1 == sense_2:
            
            if sense_2 == '+':
                primary_gene = gene_2
            else:
                primary_gene = gene_1
        
            new_coord_pos_1 = convert_H37Rv_upstream_coordinate(str_pos_1, primary_gene)
            new_coord_pos_2 = convert_H37Rv_upstream_coordinate(str_pos_2, primary_gene)

            # order them in increasing order of the upstream coordinate (not the H37Rv coordinate), which for negative sense genes will be the opposite order
            if sense_2 == '+':
                return f"{primary_gene}_{letter_prefix}.{new_coord_pos_1}_{new_coord_pos_2}{nt_change}"
            else:
                return reverse_complement_variant(f"{primary_gene}_{letter_prefix}.{new_coord_pos_2}_{new_coord_pos_1}{nt_change}")
    
        # upstream of both genes, so need to return the variant relative to both genes
        elif sense_1 == '-' and sense_2 == '+':

            new_coord_pos_1_gene_1 = convert_H37Rv_upstream_coordinate(str_pos_1, gene_1)
            new_coord_pos_2_gene_1 = convert_H37Rv_upstream_coordinate(str_pos_2, gene_1)
            
            new_coord_pos_1_gene_2 = convert_H37Rv_upstream_coordinate(str_pos_1, gene_2)
            new_coord_pos_2_gene_2 = convert_H37Rv_upstream_coordinate(str_pos_2, gene_2)

            # return a comma-separated string with both mutations -- will split them into two lines later
            # order them in increasing order of the upstream coordinate (not the H37Rv coordinate), which for negative sense genes will be the opposite order
            if sense_2 == '+':
                return f"{gene_1}_{letter_prefix}.{new_coord_pos_1_gene_1}_{new_coord_pos_2_gene_1}{nt_change},{gene_2}_{letter_prefix}.{new_coord_pos_1_gene_2}_{new_coord_pos_2_gene_2}{nt_change}"

            # reverse complement
            else:
                final_variant_1 = reverse_complement_variant(f"{gene_1}_{letter_prefix}.{new_coord_pos_2_gene_1}_{new_coord_pos_1_gene_1}{nt_change}")
                final_variant_2 = reverse_complement_variant(f"{gene_2}_{letter_prefix}.{new_coord_pos_2_gene_2}_{new_coord_pos_1_gene_2}{nt_change}")
                
                return final_variant_1, final_variant_2
        


def get_variants_with_ablations(df):

    # get all feature ablations
    # Filter columns that match the pattern
    lof_strings = ['feature_ablation', 'transcript_ablation', 'start_lost', 'stop_gained']
    annotation_columns = [col for col in df.columns if col.startswith('EFFECT')]
    
    # Initialize an empty boolean Series to combine the filter results
    combined_filter = pd.Series([False] * len(df))
    
    # Apply the search string conditions for each relevant column and combine the results
    for col in annotation_columns:
        combined_filter = combined_filter | ((~pd.isnull(df[col])) & (df[col].str.contains('|'.join(lof_strings))))
    
    # Use the combined filter to select the matching rows
    df_ablation = df[combined_filter].reset_index(drop=True)

    if len(df_ablation) == 0:
        return None

    df_ablation_add = []
    
    for i, row in df_ablation.iterrows():
    
        for col in annotation_columns:

            if not pd.isnull(row[col]):

                # there can be multiple effects separated by &. But I think there should only be one corresponding gene
                # i.e. a frameshift variant can cause an early stop codon, so it may say 'frameshift&stop_gained'
                indiv_effects = row[col].split('&')

                for variant in indiv_effects:
                    
                    if variant in lof_strings:
                        
                        add_row = df_ablation.iloc[i, :]

                        # the format is gene_deletion or frameshift or start_lost or stop_gained
                        # there is no gene name on the front of the non-deletion variants
                        # subsequent columns (EFFECT_i/GENE_i)

                        # this is for columns named GENE_i
                        if '_' in col:
                            gene_num = col.split('_')[1]
                            gene_col = f'GENE_{gene_num}'
                        else:
                            gene_col = 'GENE'

                        if 'ablation' in variant:
                            add_row['variant'] = row[gene_col] + '_deletion'
                        
                        # already checked above that row[col] is in the lof_strings
                        elif variant == 'start_lost':
                            # this is the format of the start lost mutations
                            add_row['variant'] = row[gene_col] + '_p.Met1?'

                        # for stop_gained (the only other possible lof string) can leave the variant column because it will already be in the compatible form
                        # so just get the strings from the gene and HGVS_P columns and combine them
                        else:
                            add_row['variant'] = row[gene_col] + '_' + row['HGVS_P']

                        # same for all cases
                        add_row['GENE'] = row[gene_col]
                        add_row['EFFECT'] = variant
                        
                        df_ablation_add.append(pd.DataFrame(add_row).T)

    if len(df_ablation_add) > 0:
        df_ablation_add = pd.concat(df_ablation_add)

    # don't return GENE_i or EFFECT_i. Won't know which column has the deleted gene and it's not worth getting, so these columns will be NaN in the annotated variants dataframe
    # but keep the primary columns. For ablations that affect multiple genes, each gene is a separate row (for the same variant) with the effect corresponding to that gene
    return pd.concat([df_ablation, df_ablation_add]).dropna(subset='variant').reset_index(drop=True)[['POS', 'REF', 'ALT', 'FILTER', 'QUAL', 'AF', 'DP', 'BQ', 'MQ', 'GENE', 'EFFECT', 'variant']]
    


def add_variant_column(df):

    # separate function to handle deletions that are not reflected in the GENE column but only in the ANN column, which will be listed as {gene}_deletion
    df_ablations = get_variants_with_ablations(df)

    # then remove the extra columns to clean up the dataframe
    df = df[[col for col in df.columns if 'GENE_' not in col and 'EFFECT_' not in col]]
        
    df_add = []
    
    for i, row in df.iterrows():
        
        if '-' in row['GENE']:
            cleaned_mut = process_intergenic_variant_WHO_catalog_coord(row)
        else:
            cleaned_mut = process_intragenic_variant_WHO_catalog_coord(row)
            
        df.loc[i, 'variant'] = cleaned_mut
        
        if not pd.isnull(cleaned_mut):

            # variants found in multiple overlapping genes or in the upstream promoter regions of multiple genes and might be relevant to both -- duplicate the rows
            if ',' in cleaned_mut:
                
                multi_gene_variants = cleaned_mut.split(',')
                
                # add additional rows to the end of the dataframe
                for addl_variant in multi_gene_variants[1:]:

                    add_row = df.iloc[i, :]
                    add_row['variant'] = addl_variant
                    df_add.append(pd.DataFrame(add_row).T)
                
                # make this variant only the first one
                df.loc[i, 'variant'] = multi_gene_variants[0]
            
    if len(df_add) > 0:
        df = pd.concat([df, pd.concat(df_add)])
            
    # remove the intergenic variants that are downstream of both flanking genes (so not in upstream promoter regions)
    df = df.dropna(subset='variant')

    # combine with ablations and sort by position
    # df_ablations could be None if there are no ablations
    if df_ablations is not None:
        return pd.concat([df, df_ablations]).sort_values("POS").reset_index(drop=True)
    else:
        return df.sort_values("POS").reset_index(drop=True)


output_file = input_file.replace('.tsv', '_annot.tsv')

if os.path.isfile(output_file):
    exit()
    
# read in variants TSV file, remove imprecise variants from this
try:
    df = pd.read_csv(input_file, sep='\t')
except:
    print(f"{input_file} is empty, which means that are no variants in the WHO catalog regions")
    exit()
    
# remove imprecise variants because the variant caller is not confidence
df = df.query("IMPRECISE == False")

if len(df) == 0:
    print(f"There are only imprecise variants in the WHO catalog regions in {input_file}")
    exit()

# split the annotation field. This is for large variants that affect multiple genes. Need to separate them into individual gene components
df = split_annotations(df)

# there are no gene fusions in the mutations catalog, so remove them to avoid having to try to process them
# this is the column for the first variant effect
df = df.loc[(~pd.isnull(df['EFFECT'])) & (~df['EFFECT'].str.contains('fusion'))].dropna(subset='GENE').reset_index(drop=True)

if len(df) == 0:
    print(f"There are only gene fusion variants in the WHO catalog regions in {input_file}")
    exit()
        
# make a copy to merge with the cleaned variant column later
df_og = df.copy()

df = add_variant_column(df)

# some fabG1 and inhA variants are in both gene numbering systems, so make a mapping
alias_variants_df = who_catalog_V2.loc[(~pd.isnull(who_catalog_V2['Comment'])) & (who_catalog_V2['Comment'].str.contains('alias', case=False))].drop_duplicates('variant')[['variant', 'Comment']]

alias_variants = [row['Comment'].replace('Alias ', '') for _, row in alias_variants_df.iterrows()]
alias_variants_dict = dict(zip(alias_variants, alias_variants_df['variant']))
del who_catalog_V2['Comment']

df['variant'] = df['variant'].map(alias_variants_dict).fillna(df['variant'])

# one of the alias variants is weird: fabG1_p.Leu203Leu = inhA_c.-154G>A. Because it's synonymous in fabG1, the variant column will fabG1_c.609G>N
# so need to manually replace this
df.loc[(df['GENE']=='fabG1') & (df['HGVS_P']=='p.Leu203Leu') & (df['REF']=='G') & (df['ALT']=='A'), 'variant'] = 'inhA_c.-154G>A'

# combine this with the original df for later debugging if needed
df[['POS', 'REF', 'ALT', 'variant']].merge(df_og, on=['POS', 'REF', 'ALT'], how='outer')

# then combine with the WHO catalog variants and confidences for V2 only. If V1 is needed, it will be merged in when getting resistance predictions for V1. But don't save them because each file will take up more space, and it's fairly rare for someone to need the V1 predictions
df = df.merge(who_catalog_V2, on='variant', how='left')

# save the annotated variants file for resistance prediction
df.drop_duplicates(subset=['variant', 'drug', 'confidence']).to_csv(output_file, sep='\t', index=False)
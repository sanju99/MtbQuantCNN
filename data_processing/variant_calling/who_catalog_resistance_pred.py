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
who_catalog_V1 = pd.read_csv("/home/sak0914/MtbQuantCNN/data_processing/data_utils/WHO_catalog_V1.csv")
who_catalog_V2 = pd.read_csv("/home/sak0914/MtbQuantCNN/data_processing/data_utils/WHO_catalog_V2.csv", header=[2])

who_catalog_V1 = who_catalog_V1.rename(columns={'drug': 'drug_V1', 'variant_V2': 'variant', 'confidence': 'confidence_V1'})[['drug_V1', 'variant', 'confidence_V1']]
who_catalog_V2 = who_catalog_V2[['drug', 'variant', 'FINAL CONFIDENCE GRADING', 'Comment']].rename(columns={'drug': 'drug_V2', 'FINAL CONFIDENCE GRADING': 'confidence_V2'})

parser = argparse.ArgumentParser()

# Add a required string argument for the input file (annotations)
parser.add_argument("-i", "--input", dest='input_file', type=str, required=True)

# Add a required string argument for the output file (CSV of resistance predictions)
parser.add_argument("-o", "--output", dest='output_file', type=str, required=True)

parser.add_argument('--V1', dest='V1', action='store_true', help='If true, save predictions for V1 instead of V2. The default is to get V2 predictions')

parser.add_argument('--AF-thresh', type=float, dest='AF_thresh', default=0.75, help='Alternative allele frequency threshold (exclusive) to consider variants present. Default = 0.75')

cmd_line_args = parser.parse_args()

input_file = cmd_line_args.input_file
output_file = cmd_line_args.output_file
get_V1_pred = cmd_line_args.V1
AF_thresh = cmd_line_args.AF_thresh

# must be a float in the range [0, 1)
if AF_thresh > 1:
    AF_thresh /= 100



def get_all_drug_resistances_V2(df):

    resistance_dict = {}
    
    for drug in np.sort(who_catalog_V2['drug_V2'].unique()):

        # get only Group 1-2 resistance mutations
        R_assoc_variants = who_catalog_V2.loc[who_catalog_V2['confidence_V2'].str.contains('Assoc w R')].query("drug_V2==@drug").variant.values

        # for BDQ/CFZ, LoF in mmpL5 abrogates the effect of R-assoc Rv0678 mutations
        # for AMK/KAN, LoF in eis abrogates the effect of R-assoc eis promoter mutations
        # checked that the only mutations with "Abrogates" in the comments are these mutations
        negating_muts = who_catalog_V2.dropna(subset="Comment").query("drug_V2==@drug & Comment.str.contains('Abrogates')").variant.values

        if len(df.query("variant in @negating_muts")) > 0:
            resistance_dict[drug] = 'S'
        else:
            if len(df.query("variant in @R_assoc_variants")) > 0:
                resistance_dict[drug] = 'R'
            else:
                resistance_dict[drug] = 'S'
    
    resistance_df = pd.DataFrame(resistance_dict, index=[0]).T.reset_index()
    resistance_df.columns = ['Drug', 'Phenotype']
    return resistance_df
    
    
    
    
def get_all_drug_resistances_V1(df):

    resistance_dict = {}
    
    for drug in np.sort(who_catalog_V1['drug_V1'].unique()):

        # get only Group 1-2 resistance mutations
        R_assoc_variants = who_catalog_V1.loc[who_catalog_V1['confidence_V1'].str.contains('Assoc w R')].query("drug_V1==@drug").variant.values
        
        if len(df.query("variant in @R_assoc_variants")) > 0:
            resistance_dict[drug] = 'R'
        else:
            resistance_dict[drug] = 'S'
    
    resistance_df = pd.DataFrame(resistance_dict, index=[0]).T.reset_index()
    resistance_df.columns = ['Drug', 'Phenotype']
    return resistance_df
    



def make_susceptible_dataframe():
    '''
    Make a dataframe with S for all 15 drugs
    '''
    
    resistance_df = pd.DataFrame(columns=['Drug', 'Phenotype'])
    resistance_df['Drug'] = np.sort(who_catalog_V2['drug_V2'].unique())
    resistance_df['Phenotype'] = 'S'

    return resistance_df




# add a suffix for the AF threshold
output_file = '.'.join(output_file.split(".")[:-1]) + f"_AF_thresh_{int(AF_thresh*100)}.csv"

if get_V1_pred:
    # add a suffix for V1
    output_file = '.'.join(output_file.split(".")[:-1]) + "_V1.csv"

if os.path.isfile(output_file):
    exit()

if 'annot' not in input_file:
    input_file = input_file.replace('.tsv', '_annot.tsv')
    
# annotated variants file created from the previous script
if not os.path.isfile(input_file):
    print(f"{input_file} is all drug-susceptible")
    resistance_df = make_susceptible_dataframe()
    resistance_df.to_csv(output_file, index=False)
    exit()

# rename the V2 results columns to keep them separate from V1
df = pd.read_csv(input_file, sep='\t').rename(columns={'drug': 'drug_V2', 'confidence': 'confidence_V2'})

# if needing V1 variants, merge the V1 variants in. Otherwise, it's not worth increasing the size of the variants dataframe
if get_V1_pred:
    
    # then combine with the WHO catalog variants and confidences for both versions
    df = df.merge(who_catalog_V1, on='variant', how='left')

# some issues with indels: indels with AF in the range (0.25, 0.75] are often present at a high level, but reads that start or end near the indel position don't cover it sufficiently and won't be considered as supporting the indel. These will pull down the AF, even though the indel is present in many reads. So set the AF thresold at 0.25 for these and additionally check that at least 5 reads support the indel (IC or DC).
# these variants typically have high MQ and DP, so they are very likely real

# this is quite prevalent in ethA and to a lesser degree, other non-essential drug resistance genes like katG, pncA, Rv0678, mmpL5, and mmpS5

# check that the complement of what you want is not true because not all variant types have all fields filled in, so you will lose some if you do i.e. AF > 0.75
# for indels (including gene deletions), only check DP, AF, MQ, and FILTER because BQ and QUAL are often NaN for them
df_highConf_muts = pd.concat([df.loc[(df['REF'].str.len() > df['ALT'].str.len()) & 
                                       (~df['FILTER'].str.contains('|'.join(['Del', 'LowCov'])))
                                        ].query("~(AF <= 0.25) & ~(DP < 5) & ~(MQ < 30) & ~(DC < 5)"), # deletion
    
                                df.loc[(df['REF'].str.len() < df['ALT'].str.len()) & 
                                       (~df['FILTER'].str.contains('|'.join(['Del', 'LowCov'])))
                                        ].query("~(AF <= 0.25) & ~(DP < 5) & ~(MQ < 30) & ~(IC < 5)"), # insertion

                                # for SNPs/MNPs, check everything
                                df.loc[(df['REF'].str.len() == df['ALT'].str.len()) & 
                                       (~df['FILTER'].str.contains('|'.join(['Del', 'LowCov'])))
                                        ].query("~(QUAL < 10) & ~(AF <= @AF_thresh) & ~(DP < 5) & ~(BQ < 20) & ~(MQ < 30)") # SNP
                             ])

# which catalog version to use
if get_V1_pred:
    resistance_df = get_all_drug_resistances_V1(df_highConf_muts)
else:
    resistance_df = get_all_drug_resistances_V2(df_highConf_muts)

resistance_df.to_csv(output_file, index=False)
import numpy as np
import pandas as pd
import os

# downloaded from https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_reuse_table_20231208.csv
# this was the most recent version available at the time this script was written
cryptic_reuse = pd.read_csv("./MIC_data/CRyPTIC_reuse_table_20231208.csv")

# from Sacha Laurent, FIND
cc_df = pd.read_csv("./data_processing/data_utils/drug_CC.csv")

drug_abbr_dict = {'Amikacin': 'AMI',
                  'Bedaquiline': 'BDQ',
                  'Capreomycin': 'CAP',
                  'Clofazimine': 'CFZ',
                  'Delamanid': 'DLM',
                  'Ethambutol': 'EMB',
                  'Ethionamide': 'ETH',
                  'Isoniazid': 'INH',
                  'Kanamycin': 'KAN',
                  'Levofloxacin': 'LEV',
                  'Linezolid': 'LZD',
                  'Moxifloxacin': 'MXF',
                  'Ofloxacin': 'OFX',
                  'Prothionamide': 'PRO',
                  'Pyrazinamide': 'PZA',
                  'Rifampicin': 'RIF',
                  'Rifabutin': 'RFB',
                  'Streptomycin': 'STM'
                 }

abbr_drug_dict = {value: key for key, value in drug_abbr_dict.items()}


def determine_normalization_media(drug, cc_df):
    '''
    Accepts both full-length drug names and abbreviations (but must be in the dictionary above)
    '''
    if drug not in drug_abbr_dict.keys():
        if drug in abbr_drug_dict.keys():
            drug = abbr_drug_dict[drug]
        else:
            raise ValueError(f"{drug} is not in either of the drug dictionaries")

    # UKMYC5 and UKMYC6 CCs are the same, but check
    if cc_df.query("Drug==@drug & Medium.str.contains('UKMYC')")['Value'].nunique() > 1:
        raise ValueError(f"There is more than 1 critical concentration for {drug} in UKYMC plates")

    # pick one of the UKMYC plates, doesn't matter
    cryptic_cc = cc_df.query("Drug==@drug & Medium == 'UKMYC5'")['Value'].values[0]

    found_medium = False
    
    for medium in ['7H10', '7H11']:
        if len(cc_df.query("Drug == @drug & Medium == @medium")) == 1:
            medium_to_normalize_to = medium
            normalize_cc = cc_df.query("Drug == @drug & Medium == @medium")['Value'].values[0]
            found_medium = True
            break

    if not found_medium:
        medium_to_normalize_to = 'UKMYC5'
        normalize_cc = cryptic_cc

    return cryptic_cc, medium_to_normalize_to, normalize_cc



def split_cryptic_mic_ranges(drug, df):
    '''
    For CRyPTIC data, the recorded MIC becomes the upper bound. The lower bound is one dilution lower (typically 1/2 of the upper bound). The midpoint is the average.
    
    MICs at the upper extreme have the same value for the lower and upper bounds and the midpoint. MICs at the lower extreme have lower = 0, upper = MIC, and midpoint = average. 
    '''

    df = df.reset_index(drop=True)
    drug = drug.upper()
    print(f"Processing MICs for {len(df)} isolates for {drug}")
    
    unique_mics = list(np.sort(np.unique([float(num.strip(">").strip("<=")) for num in df[f"{drug}_MIC"]])))
    print(f"CRyPTIC breakpoints: {unique_mics}\n")

    for i, row in df.iterrows():

        raw_mic = row[f"{drug}_MIC"]
        
        if "<" in raw_mic:
            val = float(raw_mic.strip("<="))
            df.loc[i, f"{drug}_lower_bound"] = 0
            df.loc[i, f"{drug}_midpoint"] = val / 2
            df.loc[i, f"{drug}_upper_bound"] = val

        # for this case, put the value for everything because the error function should be computed normally
        elif ">" in raw_mic:
            val = float(raw_mic.strip(">"))
            df.loc[i, f"{drug}_lower_bound"] = val
            df.loc[i, f"{drug}_midpoint"] = val
            df.loc[i, f"{drug}_upper_bound"] = np.inf
                
        else:
            upper = float(raw_mic)

            # the lower bound is one dilution below
            if unique_mics.index(upper) > 0:
                lower = unique_mics[unique_mics.index(upper)-1]
            else:
                lower = 0

            df.loc[i, f"{drug}_upper_bound"] = upper
            df.loc[i, f"{drug}_lower_bound"] = lower
            df.loc[i, f"{drug}_midpoint"] = np.mean([lower, upper])
            
        i += 1

    # check that bounds make sense with the midpoints
    assert len(df.query(f"{drug}_lower_bound > {drug}_midpoint")) == 0
    assert len(df.query(f"{drug}_upper_bound < {drug}_midpoint")) == 0
                
    assert len(df.loc[pd.isnull(df[f'{drug}_midpoint'])]) == 0
    return df



def normalize_MICs(df, drug):

    cryptic_cc, medium_to_normalize_to, normalize_cc = determine_normalization_media(drug, cc_df)
    
    df[f'{drug}_MEDIA_NORM'] = medium_to_normalize_to

    for col in [f'{drug}_lower_bound', f'{drug}_midpoint', f'{drug}_upper_bound']:

        # normalize according to Farhat et al., Nat Comm, 2019
        # MIC_2 = MIC_1 x (CC_2 / CC_1)
        df[f'{col}_NORM'] = df[col] * normalize_cc / cryptic_cc

    return df
    

dfs_lst = []

print(f"Processing CRyPTIC MICs for {len(drug_abbr_dict)} drugs\n")

out_dir = "./CRyPTIC_single_drugs"

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for full_drug_name, drug in drug_abbr_dict.items():

    # the dictionary is exhaustive, but the CRyPTIC dataframe may not have data for all these drugs
    if f'{drug}_MIC' in cryptic_reuse.columns:

        # single drug dataframe
        single_drug_df = cryptic_reuse[['ENA_RUN', 'ENA_SAMPLE', f'{drug}_BINARY_PHENOTYPE', f'{drug}_PHENOTYPE_QUALITY', f'{drug}_MIC']].dropna()
    
        # split MICs into ranges, with lower bound, and upper bound, and midpoint
        single_drug_df_clean = split_cryptic_mic_ranges(drug, single_drug_df)
    
        # normalize all three columns
        single_drug_df_clean_norm = normalize_MICs(single_drug_df_clean, drug)
    
        # single_drug_df_clean_norm.to_csv(f"{out_dir}/{full_drug_name}.csv", index=False)
        dfs_lst.append(single_drug_df_clean_norm.set_index(['ENA_RUN', 'ENA_SAMPLE']))

# column merge the list of dataframes. By setting the sample IDs to be the indices, the dataframe will be concatenated properly
df_combined = pd.concat(dfs_lst, axis=1, join='outer')

# then reset index so that you have ENA_RUN and ENA_SAMPLE in the columns
df_combined = df_combined.reset_index()

# naming conventions: sort ascending run IDs and replace . with , for consistency with other dataframes
for i, row in df_combined.iterrows():
    if '.' in row['ENA_RUN']:
        df_combined.loc[i, 'ENA_RUN'] = ','.join(np.sort(row['ENA_RUN'].replace('.', ',').split(',')))

# combine with BioSample IDs
cryptic_metadata = pd.read_csv("./MIC_data/cryptic_WGS_metadata.csv")

# exclude ENA_RUN column, join on Sample, which is a unique sample identifier, like BioSample
df_combined = cryptic_metadata[['BioSample', 'Sample', 'Combined_Runs']].drop_duplicates().merge(df_combined.iloc[:, 1:].rename(columns={'ENA_SAMPLE': 'Sample'}), how='inner', on='Sample')

print(f"Final dataframe shape: {df_combined.shape}")
print(f"{df_combined.BioSample.nunique()} samples")

df_combined.to_csv("./MIC_data/CRyPTIC_normalized.csv", index=False)
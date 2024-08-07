import numpy as np
import pandas as pd
import os, sys, warnings
warnings.filterwarnings('ignore')

# copy of /n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/MIC_combined_data.csv, with the following changes:
# renamed drug columns for consistency with WHO catalog and removed _lower_bound, _upper_bound, and _midpoint columns to redo the normalization. Kept the original string drug column

# df_rollingDB['MEDIA'] = df_rollingDB['MEDIA'].replace('BACTEC MGIT 960', 'MGIT').replace('Middlebrook 7H10', '7H10').replace('Middlebrook 7H10 (except PZA)', '7H10')

# see /n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/MIC_combined_data_readme.txt
# df_rollingDB.loc[df_rollingDB['STUDY_NAME']=='Eldholm Nat Comm 2015', 'MEDIA'] = '7H9'

rollingDB_drugs = ['INH', 'RIF',
       'RFB', 'EMB', 'STM', 'ETH', 'CIP', 'CYS', 'CAP', 'KAN', 'OFX', 'PAS',
       'PZA', 'AMI', 'MXF', 'PRO', 'CFZ', 'LEV', 'CLAR', 'GATI', 'AMOXCLAV',
       'LZD']

# no_mic_isolates = []

# for i, row in df_rollingDB.iterrows():
#     if sum(~pd.isnull(row[rollingDB_drugs])) == 0:
#         no_mic_isolates.append(row['BioSample'])

# 40 samples
# print(f"Removed {len(no_mic_isolates)} samples with no MICs")
# df_rollingDB = df_rollingDB.query("BioSample not in @no_mic_isolates").reset_index(drop=True)

# 22 samples
# print(f"Removed {sum(pd.isnull(df_rollingDB['MEDIA']))} samples with no media listed")
# df_rollingDB = df_rollingDB.loc[~pd.isnull(df_rollingDB['MEDIA'])]

# contains multiple rows for the same biosample if they have multiple sequencing runs
df_rollingDB = pd.read_csv("./MIC_data/rollingDB_MIC_raw.csv").drop_duplicates('BioSample', keep='first')

df_alland = pd.read_csv("./MIC_data/Alland_Courtney_MICs.csv")

# not sure why there are duplicate IDs, but first remove duplicate rows that are identical everywhere (keep the first instance)
# then drop any remaining duplicate IDs that have different values (MICs) in the different rows. Don't keep these samples at all
df_MIC_ML = pd.read_csv("./MIC_data/MIC_ML.csv").drop_duplicates(keep='first').drop_duplicates(subset='ROLLINGDB_ID', keep=False).reset_index(drop=True)

df_cryptic = pd.read_csv("./MIC_data/CRyPTIC_normalized.csv")

# from Sacha Laurent, FIND
cc_df = pd.read_csv("~/who-analysis/data/drug_CC.csv")

# curated by the Farhat lab
cc_df_internal = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/critical_concentrations.csv", index_col=[0])

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


def split_mic_ranges(df, drug):
    '''
    For ROLLINGDB data, most of it is already in bound form, so the upper and lower bounds are extracted, and the average becomes the midpoint. Isolates with only a single MIC (no range) recorded are removed. 
    
    MICs at the upper extreme have the same value for the lower and upper bounds and the midpoint. MICs at the lower extreme have lower = 0, upper = MIC, and midpoint = average. 
    '''
    
    df = df.dropna(subset=drug).reset_index(drop=True)
    drug = drug.upper()

    for i, row in df.iterrows():

        raw_mic = row[f"{drug}"]
        
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
                
        elif "-" in raw_mic:
            lower = float(raw_mic.split("-")[0])
            upper = float(raw_mic.split("-")[1])
            df.loc[i, f"{drug}_lower_bound"] = lower
            df.loc[i, f"{drug}_midpoint"] = np.mean([lower, upper])
            df.loc[i, f"{drug}_upper_bound"] = upper

        # the rest will be NaN
        else:
            continue
            
        i += 1

    # check that bounds make sense with the midpoints
    assert len(df.query(f"{drug}_lower_bound > {drug}_midpoint")) == 0
    assert len(df.query(f"{drug}_upper_bound < {drug}_midpoint")) == 0

    # print(f"Dropped {sum(pd.isnull(df[f'{drug}_midpoint']))} isolates without MIC ranges")

    # some studies used different media for different drugs, so update the MEDIA column accordingly
    if drug in ['INH', 'RIF', 'STM', 'EMB']:
        df.loc[df['MEDIA']=='LJ (INH, RIF, STR, EMB), 7H11 (CIP, KAN, CAP, ETA, PAS, CYS)', 'MEDIA'] = 'LJ'

    if drug in ['CIP', 'KAN', 'CAP', 'ETH', 'PAS', 'CYS']:
        df.loc[df['MEDIA']=='LJ (INH, RIF, STR, EMB), 7H11 (CIP, KAN, CAP, ETA, PAS, CYS)', 'MEDIA'] = '7H11'

    # No 7H10 or 7H11 critical concentrations
    if drug == 'PZA':
        df['MEDIA'] = 'MGIT'

    for other_drug in rollingDB_drugs:
        if other_drug != drug:
            del df[other_drug]
    
    # there will be some NaNs (e.g. no MIC range, so drop them)
    return df.loc[~pd.isnull(df[f'{drug}_midpoint'])]



def normalize_MICs(df, drug, cc_df):
    '''
    Accepts both full-length drug names and abbreviations (but must be in the dictionary above)
    '''
    if drug not in drug_abbr_dict.keys():
        if drug in abbr_drug_dict.keys():
            drug_full_name = abbr_drug_dict[drug]
        else:
            raise ValueError(f"{drug} is not in either of the drug dictionaries")

    medium_cc_dict = dict(zip(cc_df.query("Drug==@drug_full_name")['Medium'], cc_df.query("Drug==@drug_full_name")['Value']))

    found_medium = False
    
    for medium in ['7H10', '7H11', 'MGIT']:
        if len(cc_df.query("Drug == @drug_full_name & Medium == @medium")) == 1:
            medium_to_normalize_to = medium
            normalize_cc = cc_df.query("Drug == @drug_full_name & Medium == @medium")['Value'].values[0]
            found_medium = True
            break

    if not found_medium:
        raise ValueError(f"Count not find a medium to normalize to for {drug}")

    print(f"Normalizing to {medium_to_normalize_to} medium")
    df[f'{drug}_MEDIA_NORM'] = medium_to_normalize_to
    df['NORM_CC'] = normalize_cc
    df['ORIG_CC'] = df['MEDIA'].map(medium_cc_dict)

    for col in [f'{drug}_lower_bound', f'{drug}_midpoint', f'{drug}_upper_bound']:

        # normalize according to Farhat et al., Nat Comm, 2019
        # MIC_2 = MIC_1 x (CC_2 / CC_1)
        df[f'{col}_NORM'] = df[col] * normalize_cc / df['ORIG_CC']

    # if they are NaN, it's because there is no critical concentration for that drug, so it was dropped
    num_no_cc = sum(pd.isnull(df[f'{drug}_midpoint_NORM']))
    print(f"Dropped {num_no_cc}/{len(df)} isolates with MICs tested in media without critical concentrations")
    
    return df.dropna(subset=f'{drug}_midpoint_NORM', how='any')



def process_bounds_MICML_data(df, drug):

    if drug not in df.columns:
        return pd.DataFrame()
    
    df_single_drug = df.loc[~pd.isnull(df[drug])]
    new_dfs = []
    
    for db in df_single_drug["DB_OF_ORIGIN"].unique():
        
        df_single_db = df_single_drug.query("DB_OF_ORIGIN==@db").reset_index(drop=True)
        db_bounds = list(np.sort(np.unique(df_single_db[drug]))) # make it a list so you can use the index method

        # check that each study only used one medium for each drug
        assert df_single_db['MEDIA'].nunique() == 1

        print(f"Study: {db}, Medium: {df_single_db['MEDIA'].unique()[0]}, Breakpoints: {db_bounds}")
        
        for i, row in df_single_db.iterrows():
                            
            # set the midpoint to the second smallest value because the MIC can't actually be 0
            if row[drug] == 0:
                midpoint_idx = 1
            else:
                midpoint_idx = db_bounds.index(row[drug])
            
            # the recorded value is the upper bound
            new_high = db_bounds[midpoint_idx]
            
            # if the upper bound is the smallest concentration, the lower bound should be 0
            if midpoint_idx == 0:
                new_low = 0 #db_bounds[midpoint_idx]
            # the lower bound should be 1 concentration below the upper bound
            else:
                new_low = db_bounds[midpoint_idx-1]

            # these MICs have already been normalized, so add the NORM suffix to the names
            df_single_db.loc[i, [f"{drug}_lower_bound_NORM", f"{drug}_upper_bound_NORM", f"{drug}_midpoint_NORM"]] = [new_low, new_high, np.mean([new_low, new_high])]
                
        new_dfs.append(df_single_db)
    
    df_final = pd.concat(new_dfs, axis=0)
    assert len(df_final) == len(df_single_drug)
    assert len(set(df_final["ROLLINGDB_ID"]).symmetric_difference(df_single_drug["ROLLINGDB_ID"])) == 0

    if drug in ['BDQ', 'DLM']:
        df_final[f'{drug}_MEDIA_NORM'] = '7H11'
    else:
        df_final[f'{drug}_MEDIA_NORM'] = '7H10'

    cols = ['ID', 'ROLLINGDB_ID', 'ISOLATION_LOCATION', 'DB_OF_ORIGIN', 'TESTING_LOCATION', 'MEDIA', f'{drug}_MEDIA_NORM'] + [f"{drug}_quality", f"{drug}_lower_bound_NORM", f"{drug}_midpoint_NORM", f"{drug}_upper_bound_NORM"]
    keep_cols = list(set(cols).intersection(df_final.columns))
    return df_final[keep_cols]
    


_, drug = sys.argv

output_dir = os.path.join("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs", drug.upper())

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if drug in df_rollingDB.columns:
    df_rollingDB_single_drug = split_mic_ranges(df_rollingDB, drug)
    df_rollingDB_single_drug = normalize_MICs(df_rollingDB_single_drug, drug, cc_df)
    df_rollingDB_single_drug = df_rollingDB_single_drug.rename(columns={'BioSample': 'ROLLINGDB_ID'})[['ROLLINGDB_ID', 'DB_OF_ORIGIN', 'MEDIA'] + list(df_rollingDB_single_drug.columns[df_rollingDB_single_drug.columns.str.contains(drug)])].dropna(subset=f"{drug}_midpoint_NORM", axis=0)
else:
    df_rollingDB_single_drug = pd.DataFrame()

# already split to midpoint/bounds and assume normalization
if drug in df_alland.columns:
    df_alland_single_drug = df_alland[['ROLLINGDB_ID', f"{drug}_lower_bound_NORM", f"{drug}_midpoint_NORM", f"{drug}_upper_bound_NORM"]]
    df_alland_single_drug['DB_OF_ORIGIN'] = 'Alland'
else:
    df_alland_single_drug = pd.DataFrame()
    
if drug in df_MIC_ML.columns:
    df_MIC_ML_single_drug = process_bounds_MICML_data(df_MIC_ML, drug)

    # MIC-ML data was already normalized, so leave as is
else:
    df_MIC_ML_single_drug = pd.DataFrame()

if f"{drug}_midpoint_NORM" in df_cryptic.columns:
    df_cryptic_single_drug = df_cryptic.dropna(subset=f"{drug}_midpoint_NORM", axis=0)

    df_cryptic_single_drug = df_cryptic_single_drug.rename(columns={'BioSample': 'ROLLINGDB_ID'})[['ROLLINGDB_ID'] + list(df_cryptic.columns[df_cryptic_single_drug.columns.str.contains(drug)])]
    df_cryptic_single_drug['MEDIA'] = 'UKMYC'
    df_cryptic_single_drug['DB_OF_ORIGIN'] = 'CRyPTIC'
else:
    df_cryptic_single_drug = pd.DataFrame()

# remove LOW quality phenotypes and combine the four datasets. Keep the first instance (for i.e. TDR isolates that are in rollingDB and Alland)
df_combined = pd.concat([df_cryptic_single_drug,
                         df_rollingDB_single_drug,
                         df_MIC_ML_single_drug,
                         df_alland_single_drug
                        ]).drop_duplicates('ROLLINGDB_ID', keep='first')

if f"{drug}_PHENOTYPE_QUALITY" in df_combined.columns:
    df_combined = df_combined.query(f"{drug}_PHENOTYPE_QUALITY != 'LOW'")

# remove extra metadata columns
del_cols = [f"{drug}_BINARY_PHENOTYPE", f"{drug}_MIC", 'NORM_CC', 'ORIG_CC', drug, 'ID', 'ISOLATION_LOCATION', 'TESTING_LOCATION']

for col in del_cols:
    if col in df_combined.columns:
        del df_combined[col]

print(f"{len(df_combined)} MICs for {drug}") 
print(df_combined['DB_OF_ORIGIN'].value_counts())
df_combined.rename(columns={f"{drug}_MEDIA_NORM": "MEDIA_NORM"}).to_csv(os.path.join(output_dir, "combined_MIC.csv"), index=False)
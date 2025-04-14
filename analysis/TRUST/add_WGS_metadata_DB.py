import numpy as np
import pandas as pd
import glob, os, argparse

parser = argparse.ArgumentParser()

# "/n/data1/hms/dbmi/farhat/rollingDB/TRUST/clinical_data/20240826_raw_data.csv"
parser.add_argument("-d", "--dir", dest='WGS_directory', type=str, default="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_processed", help='Directory where the processed WGS data for the TRUST isolates is stored')
parser.add_argument("-o", "--output", dest='out_fName', type=str, default="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_summary.csv", help='Full path to a CSV file for the WGS summary')

cmd_line_args = parser.parse_args()
WGS_directory = cmd_line_args.WGS_directory
out_fName = cmd_line_args.out_fName

# append to existing dataframe
if os.path.isfile(out_fName):
    df_WGS = pd.read_csv(out_fName)
else:
    df_WGS = pd.DataFrame(columns=['SampleID'])

all_WGS_samples = os.listdir(WGS_directory)
df_add = pd.DataFrame(columns = ['SampleID', 'Kraken_Unclassified_Percent', 'Mean_Depth', 'Median_Depth', 'Perc_Sites_10x', 'Perc_Sites_20x', 'F2']).set_index('SampleID')

df_add_flc = []

for i, name in enumerate(all_WGS_samples):

    if name not in df_WGS.SampleID.values:
        
        kraken_report = pd.read_csv(f"{WGS_directory}/{name}/{name}/kraken/kraken_report", sep='\t', header=None)
    
        if 'unclassified' in kraken_report[5].values:
            kraken_unclassified = float(kraken_report.loc[kraken_report[5]=='unclassified'][0].values[0])
        else:
            kraken_unclassified = 0
    
        if os.path.isfile(f"{WGS_directory}/{name}/bam/{name}.dedup.bam"):
    
            df_depth = pd.read_csv(f"{WGS_directory}/{name}/bam/{name}.depth.tsv.gz", compression='gzip', sep='\t', header=None)
            assert len(df_depth) == 4411532

            mean_depth = df_depth[2].mean()
            median_depth = df_depth[2].median()
            prop_10x = len(df_depth.loc[df_depth[2] >= 10]) / len(df_depth)
            prop_20x = len(df_depth.loc[df_depth[2] >= 20]) / len(df_depth)
    
        else:

            mean_depth = np.nan
            median_depth = np.nan
            prop_10x = np.nan
            prop_20x = np.nan
    
        if os.path.isfile(f"{WGS_directory}/{name}/lineage/F2_Coll2014.txt"):
            
            F2 = pd.read_csv(f"{WGS_directory}/{name}/lineage/F2_Coll2014.txt", sep='\t', header=None)[0].values[0]
    
            flc = pd.read_csv(f"{WGS_directory}/{name}/lineage/fast_lineage_caller_output.txt", sep='\t')
            flc.columns = ['SampleID', 'Coll2014', 'Freschi2020', 'Lipworth2019', 'Shitikov2017', 'Stucki2016']
            
            df_add_flc.append(flc)
    
        else:
            F2 = np.nan
    
        df_add.loc[name, :] = [kraken_unclassified, mean_depth, median_depth, prop_10x*100, prop_20x*100, F2]
    
    if i % 100 == 0:
        print(i)

# additional string cleaning
df_add_flc = pd.concat(df_add_flc)
df_add_flc['SampleID'] = df_add_flc['SampleID'].str.replace('_variants', '')
df_add_flc['Coll2014'] = df_add_flc['Coll2014'].str.replace('lineage', '')

df_final = df_add.merge(df_add_flc, on='SampleID', how='outer').reset_index(drop=True)

# combine with the existing samples
df_final = pd.concat([df_WGS, df_final])

# get the primary lineage from the Coll2014 column
for i, row in df_final.iterrows():

    if not pd.isnull(row['Coll2014']):
        primary_lineages = []

        for single_lineage in row['Coll2014'].split(','):
            if single_lineage[0].isnumeric():
                primary_lineages.append(single_lineage[0])
            else:
                primary_lineages.append(single_lineage) # in case there are bovis samples, which are string named, not numerically named

        primary_lineages = np.sort(np.unique(primary_lineages))

        if len(primary_lineages) == 1:
            df_final.loc[i, 'Lineage'] = str(primary_lineages[0])
        else:
            df_final.loc[i, 'Lineage'] = ','.join(primary_lineages)

    else:
        df_final.loc[i, 'Lineage'] = np.nan

print(df_final.Lineage.value_counts())

# checks
assert len(df_final.query("Kraken_Unclassified_Percent > 20").dropna(subset='Median_Depth')) == 0
assert len(df_final.loc[pd.isnull(df_final['F2'])].query("Kraken_Unclassified_Percent <= 20")) == 0

df_final[['Freschi_Lineage_1', 'Freschi_Lineage_2']] = df_final['Freschi2020'].str.split(',', expand=True)
df_final['Culture_Mixed_Infection'] = (df_final['Freschi2020'].str.contains(',') | df_final['Coll2014'].str.contains(',')).astype(int)

df_final.to_csv(out_fName, index=False)
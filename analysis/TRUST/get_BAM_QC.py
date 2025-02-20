import numpy as np
import pandas as pd
import os, glob


df_TRUST = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_summary.csv")

samples_dir = "/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_processed"

df_TRUST = df_TRUST[['SampleID']]
df_TRUST['RUN'] = df_TRUST['SampleID']


def compute_BAM_depth_metrics(df, sample_id_col, run_id_col):

    df_BAM_depths = pd.DataFrame(columns = [sample_id_col, run_id_col, 'Mean_Depth', 'Median_Depth', 'Prop_20x', 'Prop_10x'])
    idx = 0
    
    for i, name in enumerate(df[sample_id_col].values):

        run_ids = np.sort(df.query(f"{sample_id_col}==@name")[run_id_col].values)
    
        if os.path.isfile(f"{samples_dir}/{name}/bam/{name}.depth.tsv.gz"):
    
            df_depth = pd.read_csv(f"{samples_dir}/{name}/bam/{name}.depth.tsv.gz", compression='gzip', header=None, sep='\t')
        
            pass_props = []
            
            for k, col in enumerate(df_depth.columns[2:]):

                mean_depth = df_depth[col].mean()
                median_depth = df_depth[col].median()
                prop_20x = len(df_depth.loc[df_depth[col] >= 20]) / len(df_depth)
                prop_10x = len(df_depth.loc[df_depth[col] >= 10]) / len(df_depth)

                df_BAM_depths.loc[idx, :] = [name, run_ids[k], mean_depth, median_depth, prop_20x, prop_10x]
                idx += 1
    
        else:
            print(f"No depth file for {name}")

        if i % 100 == 0:
            df_BAM_depths.to_csv("/home/sak0914/MtbQuantCNN/TRUST_depth_summary.csv", index=False)
            print(i)
    
    return df_BAM_depths


df_BAM_depths = compute_BAM_depth_metrics(df_TRUST, 'SampleID', 'RUN')

df_BAM_depths.to_csv("/home/sak0914/MtbQuantCNN/TRUST_depth_summary.csv", index=False)
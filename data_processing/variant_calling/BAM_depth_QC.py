import numpy as np
import pandas as pd
import sys, os

_, sample_dir = sys.argv

sample_ID = os.path.basename(sample_dir)
depths = pd.read_csv(os.path.join(sample_dir, "bam", f"{sample_ID}.depth.tsv"), sep='\t', header=None)

# update run_IDs to contain ONLY samples that pass kraken filtration so that it is consistent with the number of BAM files that exist
# so get them from the run_IDs.txt file in each folder, which contains all sequencing runs with a BAM file
run_IDs = []

with open(f"{sample_dir}/bam/run_IDs.txt", 'r') as file:
    # each line is a BAM file with the absolute path
    for line in file:
        run_IDs.append(os.path.basename(line).split('.')[0])

if len(run_IDs) + 2 != depths.shape[1]:
    raise ValueError(f"Number of columns in {depth_fName} is not consistent with {len(run_IDs)} sequencing runs")

if depths.shape[1] < 3:
    raise ValueError(f"here should be at least 3 columns in {depth_fName}. There are only {depths.shape[1]}")

depths.columns = ['CHROM', 'POS'] + run_IDs

with open(f"{sample_dir}/bam/pass_run_IDs.txt", "w+") as file:
    
    for run_ID in run_IDs:

        # some MIC-ML sample names might be numerical, i.e. 20542
        run_ID = str(run_ID)
    
        # median depth across the entire H37Rv ref genome
        median = depths[run_ID].median()
    
        # proportion of sites with a coverage of at least 20. Round in case there are samples with 0.949 or something (saw one with 0.9498)
        prop_sites_depth_20 = np.round(len(depths.loc[depths[run_ID] >= 20]) / len(depths), 2)

        print(f"Depth: {prop_sites_depth_20}")

        # median depth must be greater than 15, and at least 95% of sites must have a coverage of at least 20
        if median > 15 and prop_sites_depth_20 >= 0.95:

            # if they are the same, there is no extra directory level
            if sample_ID == run_ID:
                fName = f"{sample_dir}/bam/{sample_ID}.dedup.bam"
            else:
                fName = f"{sample_dir}/{run_ID}/bam/{run_ID}.dedup.bam"
            
            if not os.path.isfile(fName):
                raise ValueError(f"{fName} doesn't exist")
                
            file.write(fName + "\n")
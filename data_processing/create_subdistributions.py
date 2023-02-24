import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
import warnings
warnings.filterwarnings("ignore")


# output_dir = /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
_, drug, output_dir, lower_bound, upper_bound = sys.argv


df_original = pd.read_csv(os.path.join(output_dir, drug.upper(), "data_for_model.csv"))


output_dir = os.path.join("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs", drug.upper())
cryptic_genomic_path = "/n/data1/hms/dbmi/farhat/rollingDB/cryptic_output"
rollingdb_genomic_path = "/n/data1/hms/dbmi/farhat/rollingDB/genomic_data"
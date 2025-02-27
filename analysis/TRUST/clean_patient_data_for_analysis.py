import pandas as pd
import numpy as np
import glob, os, yaml, sparse, itertools, subprocess, sys, pickle, re, collections, shutil, vcf
from functools import reduce
from itertools import product

import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")
from epi_utils import *
from datetime import datetime


# this is all 452 patients, no exclusions based on the availability of WGS or validity for the TCC analysis
df_full = pd.read_csv("raw_data/20240826_metadata_MIC_method_updates.csv")

# this is all WGS samples that we have matched with patients, not just WGS at baseline
df_full_WGS = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/TRUST/clinical_data/20241127_combined_patients_WGS_samples.csv")

df_TTP_smear, df_TCC, df_combined_culture_results = get_combined_culture_results(df_full)
print(f"{df_TCC.pid.nunique()} pids for the TCC analysis")
print(f"{df_TTP_smear.dropna(subset='culture_sample_num').query('culture_sample_num <= 5').pid.nunique()} pids with baseline TTPs within the first 5 weeks")


df_trust_patients, TRUST_phenos, df_pred_combined =  read_combine_all_TRUST_data("./processed_data/combined_patients_WGS_samples.csv", CNN_results_dir="/n/data1/hms/dbmi/farhat/Sanjana/CNN_results", baseline_only=True)

# also run on this so that the variables are in better encodings
df_trust_patients = process_patient_metadata_better_encodings(df_trust_patients, TRUST_phenos, df_TTP_smear=df_TTP_smear, include_TTP=True)

print(f"{df_trust_patients.merge(df_TCC, on='pid').pid.nunique()}/{df_trust_patients.pid.nunique()} patients with valid TCC, WGS, and MICs")
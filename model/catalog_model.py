import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sparse, sys, warnings, yaml, vcf, pickle, shutil, subprocess, argparse
import scipy.optimize

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETO",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMK",
                  "Pretomanid": "PTM",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LFX",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

abbr_drug_dict = {value: key for key, value in drug_abbr_dict.items()}

data_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"
results_dir = "/n/data1/hms/dbmi/farhat/Sanjana/CNN_results"

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)
parser.add_argument('--V1', dest='V1', action='store_true', help='If true, save predictions for V1 in addition to V2')
parser.add_argument('--binary', dest='binary', action='store_true', help='If true, save predictions for the binary, not quantitative CNN. This simply affects the test set of isolates used.')
parser.add_argument('--AF-thresh', type=float, dest='AF_thresh', default=0.75, help='Alternative allele frequency threshold (exclusive) to consider variants present')

cmd_line_args = parser.parse_args()

config_file = cmd_line_args.config_file
get_V1_pred = cmd_line_args.V1
binary = cmd_line_args.binary
AF_thresh = cmd_line_args.AF_thresh

if AF_thresh > 1:
    AF_thresh /= 100

kwargs = yaml.safe_load(open(config_file, "r"))
drug = kwargs["drug"]
cc = kwargs['binary_thresh']

# single model because there is no lineage or mino acid information in the catalog model
if binary:
    phenotypes_path = os.path.join(data_dir, f"{drug}_binary")
    output_path = f"{results_dir}/{drug}_binary"
else:
    phenotypes_path = os.path.join(data_dir, drug)
    output_path = f"{results_dir}/{drug}"
    
output_file = os.path.join(output_path, "catalog_test_predictions_V2.csv")

if get_V1_pred:
    output_file = output_file.replace('V2', 'V1')

if AF_thresh != 0.75:
    output_file = '.'.join(output_file.split(".")[:-1]) + f"_AF_thresh_{int(AF_thresh*100)}." + output_file.split(".")[-1]

print(f"Saving predictions to {output_file}")

# get test set isolates
df_test = pd.read_csv(os.path.join(phenotypes_path, "data_for_model.csv")).query("category=='test_set'").reset_index(drop=True)
    
def single_isolate_catalog_resistance_prediction(sample_id, drug, get_V1_pred=False, AF_thresh=0.75):

    drug_full_name = abbr_drug_dict[drug]

    fName = f"/n/data1/hms/dbmi/farhat/rollingDB/genomic_data/{sample_id}/WHO_resistance/{sample_id}_pred_AF_thresh_{int(AF_thresh*100)}.csv"

    if get_V1_pred:
        fName = '.'.join(fName.split(".")[:-1]) + '_V1.csv'
        
    df_resistance = pd.read_csv(fName, index_col=['Drug'])

    # returns 'R' or 'S'
    return df_resistance.loc[drug_full_name, 'Phenotype']

df_test['y_pred'] = df_test['ROLLINGDB_ID'].apply(single_isolate_catalog_resistance_prediction, args=(drug, get_V1_pred, AF_thresh))
df_test['y_pred'] = df_test['y_pred'].map({'R': 1, 'S': 0})

# should not be any NaNs
assert sum(pd.isnull(df_test['y_pred'])) == 0

df_test['y_pred'] = df_test['y_pred'].astype(int)

if binary:
    df_test['y_test'] = df_test['Binary'].values
    df_test[['ROLLINGDB_ID', 'y_pred', 'y_test']].to_csv(output_file, index=False)
else:
    # if LB == CC, then that means the MIC was (LB, UB], so it is resistance. All MICs are (LB, UB].
    df_test['y_test'] = (df_test[f'{drug}_lower_bound'] >= cc).astype(int)
    df_test[['ROLLINGDB_ID', 'y_pred', 'y_test', 'Span_CC']].to_csv(output_file, index=False)    
import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml, vcf
import warnings
warnings.filterwarnings("ignore")


# example: python3 -u analysis/get_highConfMut_isolates.py RIF /n/scratch3/users/s/sak0914/annotated_VCF
_, drug, vcf_dir = sys.argv
df_phenos = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/data_for_model.csv")
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")

who_high_conf = who_variants.loc[(who_variants["drug"] == drug) & (who_variants.confidence.str.contains("|".join(["1", "2"])))].reset_index(drop=True)

for _, row in who_high_conf.iterrows():
    if "," in row["genome_index"]:
        expanded_pos = row["genome_index"].split(",")
        
        for pos in expanded_pos:
            add_df = pd.DataFrame({"drug": drug, "genome_index": pos, "confidence": row["confidence"], "gene": row["gene"], "variant": row["variant"]}, index=[len(who_high_conf)])
            who_high_conf = pd.concat([who_high_conf, add_df])
          
who_high_conf = who_high_conf.loc[~who_high_conf.genome_index.str.contains(",")]
who_high_conf = who_high_conf.drop_duplicates().reset_index(drop=True)
who_high_conf["genome_index"] = who_high_conf["genome_index"].astype(int)

aa_code_dict = {'Val':'V', 'Ile':'I', 'Leu':'L', 'Glu':'E', 'Gln':'Q', \
'Asp':'D', 'Asn':'N', 'His':'H', 'Trp':'W', 'Phe':'F', 'Tyr':'Y',    \
'Arg':'R', 'Lys':'K', 'Ser':'S', 'Thr':'T', 'Met':'M', 'Ala':'A',    \
'Gly':'G', 'Pro':'P', 'Cys':'C'}

code_aa_dict = {val: key for key, val in aa_code_dict.items()}

# convert them to 3-letter amino acid codes, which is what the ANN field
for i, row in who_high_conf.iterrows():
    
    if len(row["variant"].split("_")) == 2:
        var = row["variant"].split("_")[1]
        expand_code = code_aa_dict[var[0]] + var[1:-1] + code_aa_dict[var[-1]]
        who_high_conf.loc[i, "ANN"] = expand_code
    else:
        who_high_conf.loc[i, "ANN"] = row["variant"]
        
  
# read in list of VCF files
vcf_files_list = glob.glob(f"{vcf_dir}/*.eff.vcf")
vcf_files_list = [val for val in vcf_files_list if os.path.basename(val).split(".")[0] in df_phenos.ROLLINGDB_ID.values]

assert len(vcf_files_list) == len(df_phenos)
highConf_isolates = []

for i, fName in enumerate(vcf_files_list):
    
    try:
        vcf_file = vcf.Reader(filename=fName)
    
        for record in vcf_file:

            # if FILTER == PASS, the FILTER field is an empty list, so the length is 0
            if record.POS in who_high_conf.genome_index.values and len(record.FILTER) == 0:

                variant_to_check = who_high_conf.loc[who_high_conf["genome_index"]==record.POS, "ANN"].values[0]

                if variant_to_check in ",".join(record.INFO['ANN']):
                    highConf_isolates.append(os.path.basename(fName).split(".")[0])
                    break
    except:
        print(fName)

    if i % 100 == 0:
        print(i)
            
            
print(f"{len(highConf_isolates)} isolates with Category 1 or 2 {drug} mutations")
            
with open(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/Cat12_isolates.txt", "w+") as file:
    for isolate in highConf_isolates:
        file.write(isolate + "\n")
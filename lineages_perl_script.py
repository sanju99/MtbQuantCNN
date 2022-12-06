import numpy as np
import pandas as pd
import glob, os, sys, itertools, yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# this script runs fast-lineage-caller on the 


def get_lineages(df):
        
    for i, row in df.iterrows():      
        
        fName = row["Path"].replace(".vcf", "") + ".vcf"
        
        if not os.path.isfile(fName):
            print(f"{fName} is not a file")
        else:
            try:
                proc = subprocess.Popen(f"fast-lineage-caller {fName} --noheader --count", shell=True, encoding='utf8', stdout=subprocess.PIPE)
                output = proc.communicate()[0]

                # the second value is the Freschi et al lineage
                df.loc[i, "Lineage"] = output.split("\t")[1].replace("lineage", "")
            except:
                print(f"Problem with {fName}")
            
    # split multiple lineages per isolate, put into a new dataframe
    split_lineages = df["Lineage"].str.split(",", expand=True)
    split_lineages.columns = [f"Lineage_{num}" for num in np.arange(len(split_lineages.columns))+1]

    # separate lineage and SNP count for each one. Create columns Lineage_N and Count_N for all lineages
    for col in split_lineages:
        count_col = f"Count_{col.split('_')[1]}"
        split_lineages[[col, count_col]] = split_lineages[col].str.split("(", expand=True)
        split_lineages[count_col] = split_lineages[count_col].str.strip(")")

    # delete original Lineage column and return
    df = pd.concat([df, split_lineages], axis=1)
    del df["Lineage"]
    return df


    
def create_new_perl_script(orig_perl, drug, output_dir):
    
    paths_file = os.path.join(output_dir, "paths.txt")
    
    # get the new perl script file name. Append the drug name
    new_perl = os.path.join(output_dir, os.path.basename(orig_perl).strip(".pl") + "_" + drug.upper() + ".pl")

    # open the old file
    with open(orig_perl, "r") as old_file:

        # open the new file
        with open(new_perl, "w+") as new_file:

            # only change the line where the input paths list is given
            for line in old_file:
                if "my @fileListRaw =&ReadInFile" in line:
                    new_file.write("my @fileListRaw =&ReadInFile('" + paths_file + "');\n")
                else:
                    new_file.write(line)
                    
    print(f"Created {new_perl}!")
                    
    
# # create a new perl script for aligning sequences later
# create_new_perl_script(ref_perl_script, drug, output_dir)
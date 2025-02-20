import numpy as np
import pandas as pd
import glob, os, subprocess, vcf, shutil, sparse, yaml, sys, pickle, itertools


# drug_loci = pd.read_csv("./data_utils/drug_loci.csv")

# # make a dictionary mapping each position to the locus (to be used later)
# all_model_pos = {}

# for i, row in drug_loci.drop_duplicates('Locus').iterrows():

#     # add 1 to the end because it is exclusive. The start and end positions in the dataframe are 1-indexed, inclusive
#     pos_lst = list(np.arange(row['Start'], row['End'] + 1))

#     for pos in pos_lst:
#         all_model_pos[pos] = row['Locus']


def get_indels(fName):
    
    vcf_reader = vcf.Reader(filename=fName)
    indels_lst = []
    
    for record in vcf_reader:

        alt_allele = "".join(np.array(record.ALT).astype(str))
        
        if len(record.REF) != len(alt_allele):
            indels_lst.append(record)
    
    return indels_lst



def get_indels_same_length_proximity(fName, proximity=100):

    indels_lst = get_indels(fName)
    indel_pairs = list(itertools.combinations(indels_lst, 2))
    
    suspicious_indels = []

    # even though this is slow and indel_pairs can be long, the computations are very fast, so it runs very quickly
    for (indel_1, indel_2) in indel_pairs:

        # check proximity first
        pos_diff = np.abs(indel_1.POS - indel_2.POS)
        
        if pos_diff <= proximity:

            alt_allele_1 = "".join(np.array(indel_1.ALT).astype(str))
            alt_allele_2 = "".join(np.array(indel_2.ALT).astype(str))

            # check that lengths match
            if len(indel_1.REF) == len(indel_2.REF) and len(alt_allele_1) == len(alt_allele_2):

                # if there are two of the same indel at the same position, then don't drop both of them. Need to keep one
                if pos_diff == 0:
                    print(fName, indel_1.POS)
                    # if 'SVTYPE' in indel_1.INFO.keys() and 'SVTYPE' not in indel_2.INFO.keys():
                    #     keep_indel = indel_1
                    
                    # elif 'SVTYPE' not in indel_1.INFO.keys() and 'SVTYPE' in indel_2.INFO.keys():
                    #     keep_indel = indel_2

                else:
                    # preferentially keep the structural variant and consider it present so that we don't get an artificially low AF from the other variant
                    # this is because these are very likely true variants, but they've been duplicated, and we don't want to double count
                    if 'SVTYPE' in indel_1.INFO.keys() and 'SVTYPE' not in indel_2.INFO.keys():
                        # these shouldn't be the case in these types of indels
                        assert 'IMPRECISE' not in indel_1.INFO.keys()
                        keep_pos = indel_1.POS
                        drop_pos = indel_2.POS
                        
                    elif 'SVTYPE' not in indel_1.INFO.keys() and 'SVTYPE' in indel_2.INFO.keys():
                        # these shouldn't be the case in these types of indels
                        assert 'IMPRECISE' not in indel_2.INFO.keys()
                        keep_pos = indel_2.POS
                        drop_pos = indel_1.POS
    
                    # randomly keep one
                    else:
                        keep_idx = np.random.choice([0, 1])
                        keep_pos = [indel_1.POS, indel_2.POS][keep_idx]
                        drop_pos = [indel_1.POS, indel_2.POS][1 - keep_idx]

                    # put the drop positions second so that they can easily be removed from the VCF later
                    suspicious_indels.append([keep_pos, drop_pos])

    # print(f"{len(suspicious_indels)}/{len(indel_pairs)} indel pairs need to be checked")
    suspicious_indels = pd.DataFrame(suspicious_indels)

    if len(suspicious_indels) > 0:
        suspicious_indels.columns = ['KEEP_POS', 'DROP_POS']
    
    return suspicious_indels


# df_samples = "./samples_pass_geno_QC.csv"
# new_dir = "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF_clean"
# out_file = "./VCFs_remove_duplicate_indels.csv"
_, df_samples, new_dir, out_file = sys.argv


duplicate_indels_df = []
fNames_lst = pd.read_csv(df_samples, sep='\t', header=None)[0].values
print(f"Deduplicating indels for {len(fNames_lst)} isolates")

for i, fName in enumerate(fNames_lst):

    suspicious_indels = get_indels_same_length_proximity(fName)
    suspicious_indels['VCF'] = fName
    suspicious_indels['Isolate'] = os.path.basename(fName).replace('_variants', '').replace('.vcf', '')

    duplicate_indels_df.append(suspicious_indels)

    if i % 1000 == 0:
        print(i)

duplicate_indels_df = pd.concat(duplicate_indels_df, axis=0).reset_index(drop=True)
duplicate_indels_df[['KEEP_POS', 'DROP_POS']] = duplicate_indels_df[['KEEP_POS', 'DROP_POS']].astype(int)
print(f"{duplicate_indels_df.Isolate.nunique()}/{len(fNames_lst)} isolates have suspicious indels")

# # determine which locus they all occur in
# duplicate_indels_df['Locus'] = duplicate_indels_df['KEEP_POS'].map(all_model_pos)
# duplicate_indels_df['Locus2'] = duplicate_indels_df['DROP_POS'].map(all_model_pos)

# # check that the loci are the same for both positions
# assert len(duplicate_indels_df.loc[(pd.isnull(duplicate_indels_df['Locus'])) | (pd.isnull(duplicate_indels_df['Locus2']))]) == 0
# assert len(duplicate_indels_df.query("Locus != Locus2"))==0
# del duplicate_indels_df['Locus2']

# no changes, so copy them to the new directory
if not os.path.isdir(new_dir):
    os.makedirs(new_dir)

fNames_no_change = list(set(fNames_lst) - set(duplicate_indels_df.VCF))
print(f"Copying {len(fNames_no_change)} VCFs that don't need to be altered to {new_dir}")

for fName in fNames_no_change:
    if not os.path.isfile(os.path.join(new_dir, os.path.basename(fName).replace('_variants', ''))):
        shutil.copy(fName, os.path.join(new_dir, os.path.basename(fName).replace('_variants', '')))

# iterate through duplicate_indels_df and remove the position to drop in each pair of duplicate indels
df_remove_pos_commands = pd.DataFrame(columns=['Isolate', 'VCF', 'Command'])

# make dataframe of commands. This is because for some isolates, multiple site need to be removed, so create the command in Python here then run it in bash in the next step
for i, fName in enumerate(duplicate_indels_df.VCF.unique()):

    cmd = ''
    remove_pos_lst = duplicate_indels_df.query("VCF == @fName")['DROP_POS'].values

    for pos in remove_pos_lst:
        cmd += f'POS != {pos} & '

    # remove trailing ampersand from above
    cmd = cmd.strip('& ')
    df_remove_pos_commands.loc[i, :] = [os.path.basename(fName).replace('.vcf', '').replace('_variants', ''), fName, cmd]

df_remove_pos_commands.to_csv(out_file, header=None, index=False)
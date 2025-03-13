import pandas as pd
import numpy as np
import glob, os, yaml, sparse, itertools, re, warnings
from functools import reduce
from itertools import product
import scipy.stats as st
warnings.filterwarnings("ignore")

# ordinal encoding from RedCap: bl_afbprog --> smear
smear_encoding_dict = {6: np.nan, 
                       5: np.nan, 
                       0: 0, 
                       4: 1,
                       1: 2,
                       2: 3,
                       3: 4, 
                      }

drugs_lst = ['RIF', 'INH', 'EMB', 'PZA']


def get_all_available_MICs_single_drug(df_trust_patients, drug, baseline_only=True):

    # Function to extract the numeric part of the column name
    def extract_number(col_name):
        return int(col_name.split('_')[-1])

    # restrict weeks 1 and 2
    if baseline_only:
    
        df_MIC_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID', f's_{drug.lower()}mic_sputum_specimen_1', f's_{drug.lower()}mic_sputum_specimen_2']]
    
        # get the MIC testing media too
        # PZA doesn't have a testing media field because it was all MGIT
        if drug != 'PZA':
            df_MIC_methods_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID', f's_{drug.lower()}micmeth_sputum_specimen_1', f's_{drug.lower()}micmeth_sputum_specimen_2']]
        else:
            df_MIC_methods_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID']]

    # get all available MICs, taking the earliest one
    else:

        df_MIC_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID'] +
                                                list(df_trust_patients.columns[df_trust_patients.columns.str.contains(f's_{drug.lower()}mic_sputum_specimen')])
        ]
    
        # get the MIC testing media too
        # PZA doesn't have a testing media field because it was all MGIT
        if drug != 'PZA':
            df_MIC_methods_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID'] +
                                                            list(df_trust_patients.columns[df_trust_patients.columns.str.contains(f's_{drug.lower()}micmeth')])
            ]
        else:
            df_MIC_methods_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID']]

    df_MIC_single_drug = df_MIC_single_drug.set_index(['Original_ID', 'pid', 'SampleID'])
    df_MIC_methods_single_drug = df_MIC_methods_single_drug.set_index(['Original_ID', 'pid', 'SampleID'])
    
    # Reorder the columns based on the numeric part of their names
    df_MIC_single_drug = df_MIC_single_drug.reindex(sorted(df_MIC_single_drug.columns, key=extract_number), axis=1)
    df_MIC_methods_single_drug = df_MIC_methods_single_drug.reindex(sorted(df_MIC_methods_single_drug.columns, key=extract_number), axis=1)
    
    # get the first column
    df_MIC_single_drug[drug] = df_MIC_single_drug.iloc[:, 0].values

    # iterate through the remaining and fill NaNs
    for col in df_MIC_single_drug.columns[1:]:
        df_MIC_single_drug[drug] = df_MIC_single_drug[drug].fillna(df_MIC_single_drug[col])
    
    # keep only patient IDs with a measured MIC
    df_MIC_single_drug = df_MIC_single_drug.dropna(subset=drug).reset_index()
    
    # PZA MICs were all measured in MGIT, so there are no method columns
    if drug != 'PZA':

        # get the first column
        df_MIC_methods_single_drug[f"{drug}_method_num"] = df_MIC_methods_single_drug.iloc[:, 0].values
        
        for col in df_MIC_methods_single_drug.columns[1:]:
            df_MIC_methods_single_drug[f"{drug}_method_num"] = df_MIC_methods_single_drug[f"{drug}_method_num"].fillna(df_MIC_methods_single_drug[col])

        # keep only patient IDs with an MIC method
        df_MIC_methods_single_drug = df_MIC_methods_single_drug.dropna(subset=f"{drug}_method_num").reset_index()
    
        # merge and add the testing method. Other = Agar proportion method, which used Middlebrook 7H11 media
        media_dict = {1: 'Microtiter_plate', 2: 'MGIT', 3: '7H11'}
        df_MIC_methods_single_drug[f"{drug}_method"] = df_MIC_methods_single_drug[f"{drug}_method_num"].map(media_dict)

        return df_MIC_single_drug.merge(df_MIC_methods_single_drug[["pid", f"{drug}_method_num", f"{drug}_method"]]).drop_duplicates()

    else:
        df_MIC_single_drug[f"{drug}_method"] = 'MGIT'

        return df_MIC_single_drug.drop_duplicates()




def read_combine_WGS_MICs(patient_WGS_data_fName, df_RedCap, baseline_only=True):
    '''
    This function keeps only measured MICs and WGS samples taken in the first two weeks of treatment because we are interested in associating baseline characteristics with outcome.
    '''

    ############################################# STEP 1: READ IN THE COMBINED PATIENT-WGS DATAFRAME #############################################

    
    df_trust_patients = pd.read_csv(patient_WGS_data_fName).merge(df_RedCap, on='pid')
    
    print(f"{df_trust_patients.pid.nunique()} patients with any WGS samples")

    # fix lineages. Sometimes the names got converted to integers for the single number lineages
    for i, row in df_trust_patients.iterrows():
        if not pd.isnull(row['Lineage']):
            if type(row['Lineage']) != str:
                df_trust_patients.loc[i, 'Lineage'] = str(int(row['Lineage']))
    
    df_trust_patients['Lineage'] = df_trust_patients['Lineage'].astype(str)
    df_trust_patients['Lineage'] = df_trust_patients['Lineage'].replace('nan', np.nan)

    # keep only WGS samples that were not contaminated. Low sequencing depth isn't an issue here, they were all sequenced to very high depths
    df_trust_patients = df_trust_patients.dropna(subset='F2').reset_index(drop=True)
    
    print(f"{df_trust_patients.pid.nunique()} patients with uncontaminated WGS samples")

    
    ######################################### STEP 2: KEEP ONLY SEQUENCES COLLECTED IN THE FIRST 2 WEEKS ##############################################


    # get the sample week
    df_trust_patients['sample_collection_week'] = df_trust_patients['Original_ID'].str.split('-').str[1]
    
    # replace month 5 with 20 for weeks
    df_trust_patients['sample_collection_week'] = df_trust_patients['sample_collection_week'].replace('01A', '01').replace('m5', '20')
    
    df_trust_patients['sample_collection_week'] = df_trust_patients['sample_collection_week'].astype(int)
    
    # keep only WGS samples collected in the first 2 weeks
    if baseline_only:
        df_trust_patients = df_trust_patients.query("sample_collection_week <= 2").reset_index(drop=True)
    
        print(f"{df_trust_patients.pid.nunique()} patients with uncontaminated WGS samples taken in the first 2 weeks\n")

    
    ###################################################### STEP 3: REMOVE PATIENTS WITH SEQUENCING AT THE SAME TIMEPOINT WITH DIFFERENT LINEAGES ###########################
    

    pids_multiple_sequences_same_timepoint = df_trust_patients.iloc[df_trust_patients.index.values[df_trust_patients.duplicated('Original_ID', keep=False)]].pid.unique()

    pids_multiple_sequences_same_timepoint_discordant_lineages = pd.DataFrame(df_trust_patients.groupby(['pid', 'Original_ID'])['Coll2014'].nunique()).query("Coll2014 > 1").reset_index().pid.values
    
    print(f"{len(pids_multiple_sequences_same_timepoint)} patients: {pids_multiple_sequences_same_timepoint} have multiple sequences at the same timepoint")
    print(f"Removing {len(pids_multiple_sequences_same_timepoint_discordant_lineages)} patients: {pids_multiple_sequences_same_timepoint_discordant_lineages} because there are multiple WGS samples at the same timepoint with different lineages")

    df_trust_patients = df_trust_patients.query("pid not in @pids_multiple_sequences_same_timepoint_discordant_lineages")
    
    
    ###################################################### STEP 4: READ IN ALL AVAILABLE MEASURED MICS ######################################################

    
    TRUST_phenos = []
    
    for drug in drugs_lst:
        
        # this contains all WGS runs, so keep only the unique pids for counting/plotting purposes
        # all MICs here were converted to MGIT. The only drug for which that makes any difference is INH, whose MGIT (0.1) and 7H10 (0.2) critical concentrations are different.
        df_single_drug = get_all_available_MICs_single_drug(df_trust_patients, drug, baseline_only=baseline_only)
        df_single_drug = df_single_drug[['pid', drug, f"{drug}_method"]]
        
        print(f"{len(df_single_drug)} patients have MICs for {drug}")
        TRUST_phenos.append(df_single_drug)
    
    TRUST_phenos = reduce(lambda left, right: pd.merge(left, right, on='pid', how='outer'), TRUST_phenos).drop_duplicates()
    
    if baseline_only:
        print(f"{TRUST_phenos.pid.nunique()} patients have measured MICs in the first 2 weeks\n")
    else:
        print(f"{TRUST_phenos.pid.nunique()} patients have measured MICs\n")

    return df_trust_patients, TRUST_phenos

    

def get_combined_smear_and_culture_results_single_pid(df_trust_patients, pid):

    ########################################## STEP 1: CULTURE POSITIVITIY ########################################## 

    # get all sputum culture results for a single pid 
    single_pid_culture_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + list(df_trust_patients.columns[df_trust_patients.columns.str.contains('culture_conversion')])].set_index('pid').loc[pid]).reset_index()
    
    single_pid_culture_results.columns = ['column', 'result']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_culture_results['sample_num'] = single_pid_culture_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_culture_results['column'] # original column name, don't need anymore
    single_pid_culture_results = single_pid_culture_results.sort_values('sample_num').reset_index(drop=True)

    ########################################## STEP 2: TIME TO CULTURE POSITIVITY ########################################## 
    
    # get all TTP culture results for a single pid 
    single_pid_TTP_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('ttp')) & (~df_trust_patients.columns.str.contains('|'.join(['analysis', 'hour', 'day'])))])].set_index('pid').loc[pid]).reset_index()
    
    # BMC Group combined TTP in days with TTP hours (so days + 24 * hours) to get this column
    single_pid_TTP_results.columns = ['column', 'hours']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_TTP_results['sample_num'] = single_pid_TTP_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_TTP_results['column'] # original column name, don't need anymore
    single_pid_TTP_results = single_pid_TTP_results.sort_values('sample_num').reset_index(drop=True)
    
    ########################################## STEP 3: SMEAR GRADE ##########################################
            
    # get all sputum culture results for a single pid 
    single_pid_smear_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + list(df_trust_patients.columns[df_trust_patients.columns.str.contains('s_concafb_sputum_specimen')])].set_index('pid').loc[pid]).reset_index()
    
    single_pid_smear_results.columns = ['column', 'smear_grade']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_smear_results['sample_num'] = single_pid_smear_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_smear_results['column'] # original column name, don't need anymore
    single_pid_smear_results = single_pid_smear_results.sort_values('sample_num').reset_index(drop=True)

    # change to proper ordinal encoding
    single_pid_smear_results['smear_grade'] = single_pid_smear_results['smear_grade'].map(smear_encoding_dict)

    ########################################## STEP 4: COMBINE ALL SPUTUM RESULTS ##########################################
    
    # combine culture results (positive, negative, contaminated) with TTP results
    combined_sputum_results = single_pid_culture_results.merge(single_pid_TTP_results, on='sample_num', how='outer').merge(single_pid_smear_results, on='sample_num', how='outer')

    # for contaminated samples, the TTP is not valid, so replace with NaN. BMC group did this in their data cleaning as well
    combined_sputum_results.loc[combined_sputum_results['result'] != 'tb_positive', 'hours'] = np.nan

    return combined_sputum_results
    



def get_time_to_culture_conversion(single_sample_combined_sputum_results):
    '''
    This function computes the TCC for a single sample in the sputum results dataframe. A TTP is only valid if the culture result is tb_positive.
    '''

    # keep only samples up to 12 (in case the month 5 samples are still in the table)
    single_sample_combined_sputum_results = single_sample_combined_sputum_results.query("sample_num <= 12")

    # replacement that the BMC group did. This is only for the TCC calculation. Keep the original samples unchanged
    single_sample_combined_sputum_results['result'] = single_sample_combined_sputum_results['result'].replace('tb_positive_contaminated', 'tb_positive')
    
    start_positive = single_sample_combined_sputum_results.query("result=='tb_positive'").sample_num.min()

    if start_positive is None:
        raise ValueError(f"There is no start culture positivity time for {pid}")
        
    # the TCC will be the first of two negative result that are not followed by a positive result
    # don't consider positive smear because smear test can detect dead bacteria, which won't grow in the culture.
    end_positive = single_sample_combined_sputum_results.query("result=='tb_positive'").sample_num.max()

    # exclude the month 5 culture (sample_num = 20) from the TCC computation
    post_last_positive_results = single_sample_combined_sputum_results.query("sample_num > @end_positive").reset_index(drop=True)

    # initialize as None variable
    start_negative = None

    # check that there are at least 2 negative results, otherwise don't do the search below
    if len(post_last_positive_results.query("result=='tb_negative'")) >= 2:

        # check that they are consecutive results
        for i, row in post_last_positive_results.iterrows():

            # check that it's not the last culture, in which case there won't be a second negative afterwards
            if row['result'] == 'tb_negative' and i != len(post_last_positive_results) - 1:
                
                if post_last_positive_results.result.values[i+1] == 'tb_negative':
                    
                    # get the sample number of the first negative sample
                    start_negative = row['sample_num']

                    # can break because we already checked above that there are no positive cultures or smear grades afterwards
                    break

    # if no culture conversion (no event), then the patient did not culture convert, so take the maximum number of weeks
    if start_negative is None:
        # start_negative = 12 # the last culture sample in the treatment window. Don't consider 
        #start_negative = single_sample_combined_sputum_results.sample_num.max()

        # take the time of the last known positive culture. They will be censored at this time. If there are contaminated or single negative cultures after this time,
        # we can't interpret them because they are inconclusive. 
        # Exclude the values above 12 because those aren't weeks

        # if you take the last negative sample, sometimes you take a negative sample that occurs before a positive sample. The time of the last positive sample is probably the most informative time
        # take the last known time when the patient was smear positive or culture positive
        start_negative = end_positive
        
        culture_convert = 0
    else:
        culture_convert = 1

    # keep track of patients who culture converted
    # all patients in this study have TB (microbiologically confirmed), so take week 1 as the starting time
    return culture_convert, start_negative




def get_combined_culture_results(df_trust_patients):

    # combined_TTP_results = []
    df_TCC = pd.DataFrame(columns = ['pid', 'culture_convert', 'TCC'])
    df_TTP_smear = pd.DataFrame(columns = ['pid', 'culture_sample_num', 'TTP', 'smear_sample_num', 'smear_grade'])
    i = 0

    df_combined_culture = []
    
    for pid in df_trust_patients.pid.unique():

        # use the function above to get all smear and culture results for a single pid
        combined_sputum_results = get_combined_smear_and_culture_results_single_pid(df_trust_patients, pid).query("sample_num <= 13")

        # week 13 is actually month 5, so replace with 20
        combined_sputum_results.loc[combined_sputum_results['sample_num']==13, 'sample_num'] = 20
        
        combined_sputum_results['pid'] = pid
        df_combined_culture.append(combined_sputum_results)
        
        # get the first measured smear grade (so not NA). Smear test doesn't require culturing, so this is separate from the TTP calculation
        smear_grade_baseline = combined_sputum_results.dropna(subset='smear_grade')['smear_grade'].values[0]
        smear_grade_sample = combined_sputum_results.dropna(subset='smear_grade').sample_num.values[0]
        
        # get the first time to culture positivity (in hours) for a single sample in the sputum results dataframe
        baseline_positive_sample = combined_sputum_results.query("result=='tb_positive'").sample_num.min()

        # no positive culture sample for this pid. Probably only contaminated positive samples
        # similarly, get the smear grade at the first tb_positive culture
        if pd.isnull(baseline_positive_sample):
            TTP = np.nan
        else:
            TTP = combined_sputum_results.query("sample_num==@baseline_positive_sample")['hours'].values[0]
            
        # from BMC inclusion/exclusion criteria for TCC analysis:
        # 1. exclude patients with fewer than 3 culture samples because TCC analysis requires at least 1 positive and 2 negatives. Many of these withdrew. Some just have missing cultures
        # 2. exclude patients who don't have at least one negative culture because it's hard to reliably tell when they culture converted if you don't have that
        # 3. exclude patients who didn't have a positive culture in the first 5 weeks. This is because we assume that if they had a positive sample within the first 5 weeks, they were
        # positive at baseline, and we don't have to adjust the TCC timeline for them
        exclude_patient = False

        # first check that they had at least 3 total cultures. tb_positive_contaminated is okay as it is is replaced with tb_positive above. This is what they determined. 
        if len(combined_sputum_results.query("result in ['tb_negative', 'tb_positive']")) < 3:
            exclude_patient = True
            
        # check that they had a positive TB culture within the first 5 weeks
        elif 'tb_positive' not in combined_sputum_results.query("sample_num <= 5")['result'].values:
            exclude_patient = True

        if exclude_patient == True:
            # print(f"{pid} needs to be excluded!")
            culture_convert = np.nan
            TCC = np.nan
        else:
            culture_convert, TCC = get_time_to_culture_conversion(combined_sputum_results)
        
        df_TCC.loc[i, :] = [pid, culture_convert, TCC]
        df_TTP_smear.loc[i, :] = [pid, baseline_positive_sample, TTP, smear_grade_sample, smear_grade_baseline]
        
        i += 1

    return df_TTP_smear, df_TCC.dropna(subset='TCC').reset_index(drop=True), pd.concat(df_combined_culture).dropna(subset='result').reset_index(drop=True)



def get_reenrolled_patients(df):

    df_reenroll = df.loc[~pd.isnull(df['screen_prevpid'])].sort_values("pid").reset_index(drop=True)
    df_reenroll['screen_prevpid'] = df_reenroll['screen_prevpid'].str.replace(' and ', ',')
    
    df_reenroll = df_reenroll[['pid', 'screen_prevpid']]
    
    # for pids that have been enrolled in TRUST more than 2 times, take the latest pid because there will already be another line for the second pid mapping to the first
    # for example: T0210, T0254, and T0387 are all the same person. 
    # The line for T0387 maps to both T0210 and T0254, so split it so that there is (T0387, T0254) and (T0387, T0210)
    
    # so iterate through the multiple previous pids and append new lines to the dataframe
    for i, row in df_reenroll.iterrows():
        if ',' in row['screen_prevpid']:
            for prev_pid in row['screen_prevpid'].split(','):
                df_reenroll = pd.concat([df_reenroll, pd.DataFrame({'pid': [row['pid']], 'screen_prevpid': [prev_pid]}, index=[0])])
    
    df_reenroll = df_reenroll.query("~screen_prevpid.str.contains(',')").sort_values(['pid', 'screen_prevpid']).reset_index(drop=True)

    # print(f"{df_reenroll.pid.nunique()} pids have been previously enrolled")

    return df_reenroll


# this is all 452 patients, no exclusions based on the availability of WGS or validity for the TCC analysis
df_RedCap = pd.read_csv("./data/20240826_metadata_MIC_method_updates.csv")
print(f"{df_RedCap.pid.nunique()} patients with data in RedCap")


############### NOTE: I made some manual changes to the MICs above based on email communications with Brendon and Noorjahn. See exact replacements below ###############


# # from Brendon email September 5, 2024
# df_RedCap.loc[(pd.isnull(df_RedCap['s_inhmicmeth_sputum_specimen_1'])) & (~pd.isnull(df_RedCap['s_inhmic_sputum_specimen_1'])), 's_inhmicmeth_sputum_specimen_1'] = 3
# df_RedCap.loc[df_RedCap['pid']=='T0097', 's_inhmicmeth_sputum_specimen_2'] = 2
# df_RedCap.loc[df_RedCap['pid']=='T0063', 's_inhmicmeth_sputum_specimen_2'] = 3

# additional email from Brendon November 27, 2024:
# df_RedCap.loc[df_RedCap['pid']=='T0466', 's_embmicmeth_sputum_specimen_1'] = 2
# df_RedCap.loc[df_RedCap['pid']=='T0470', 's_rifmicmeth_sputum_specimen_1'] = 2

# # from Noorjahn's email on December 20, 2024:
# # INH MIC for sample S0108-01 was retested in duplicate in MGIT, giving an MIC of > 0.1, so update s_inhmic_sputum_specimen_1 to 5
# df_RedCap.loc[df_RedCap['pid']=='T0108', ['s_inhmicmeth_sputum_specimen_1', 's_inhmic_sputum_specimen_1']] = [2, 5]

# # INH MIC for sample S0108-08 was retested in triplicate in MGIT, giving MIC = 0.1-0.25, so update s_inhmic_sputum_specimen_1 to 4
# df_RedCap.loc[df_RedCap['pid']=='T0108', ['s_inhmicmeth_sputum_specimen_8', 's_inhmic_sputum_specimen_8']] = [2, 4]

# # INH MIC for sample S0030-01 was retested in MGIT, giving an MIC of 0.1. There was intermediate growth, so could be heteroresistance?
# # update s_inhmicmeth_sputum_specimen_1 to 2 (MGIT) and s_inhmic_sputum_specimen_1 to 3 (0.05-0.1)
# df_RedCap.loc[df_RedCap['pid']=='T0030', ['s_inhmicmeth_sputum_specimen_1', 's_inhmic_sputum_specimen_1']] = [2, 3]


############################## STEP 1: GET ALL SAMPLES WITH WGS AND MEASURED MICS ##############################


# combine the WGS data with the RedCap data. Also extract the measured MICs for the 4 drugs
df_trust_patients, TRUST_phenos = read_combine_WGS_MICs("./data/pids_WGS_mapping.csv", df_RedCap, baseline_only=True)


############################## STEP 2: GET SPUTUM CULTURE RESULTS AND COMPUTE TCC ##############################


df_TTP_smear, df_TCC, df_combined_culture_results = get_combined_culture_results(df_RedCap)
print(f"{df_TCC.pid.nunique()} pids with a valid time to culture conversion (TCC)")
print(f"{df_trust_patients.merge(df_TCC, on='pid').pid.nunique()}/{df_trust_patients.pid.nunique()} patients have baseline WGS and a valid TCC")

# merge with df_TCC
df_trust_patients = df_trust_patients.merge(df_TCC, on='pid')

print(f"{df_trust_patients.merge(TRUST_phenos, on='pid').pid.nunique()}/{df_trust_patients.pid.nunique()} patients have measured baseline MICs")


##################### STEP 3: REMOVE PIDS OF PATIENTS WHO PREVIOUSLY ENROLLED IN TRUST #####################


df_reenroll = get_reenrolled_patients(df_RedCap)

unique_cluster_dict = {}

cluster_num = 0

for i, row in df_reenroll.iterrows():
    
    if row['pid'] not in unique_cluster_dict.keys() and row['screen_prevpid'] not in unique_cluster_dict.keys():
        
        # add both pids (same person) to the dictionary
        unique_cluster_dict[row['pid']] = cluster_num
        unique_cluster_dict[row['screen_prevpid']] = cluster_num

        # then increment the cluster number
        cluster_num += 1

    else:
        if row['pid'] in unique_cluster_dict.keys():
            # get the existing number
            cluster_num_already_present = unique_cluster_dict[row['pid']]

            # add the new one with the same cluster number
            unique_cluster_dict[row['screen_prevpid']] = cluster_num_already_present
        else:
            cluster_num_already_present = unique_cluster_dict[row['screen_prevpid']]

            # same thing but using the cluster number determined from screen_prevpid
            unique_cluster_dict[row['pid']] = cluster_num_already_present

df_patient_clusters = pd.DataFrame(unique_cluster_dict, index=[0]).T.reset_index()
df_patient_clusters.columns = ['pid', 'cluster']
del df_patient_clusters

# add the unique patient identifiers (just integers)
df_trust_patients['unique_patient'] = df_trust_patients['pid'].map(unique_cluster_dict)

# increment the unique patient values for the rest of the pids, which have unique patients
start_cluster_num = np.max(list(unique_cluster_dict.values())) + 1

for i, row in df_trust_patients.iterrows():

    if pd.isnull(row['unique_patient']):
        if row['pid'] not in unique_cluster_dict.keys():
            df_trust_patients.loc[i, 'unique_patient'] = start_cluster_num
            unique_cluster_dict[row['pid']] = start_cluster_num
            start_cluster_num += 1
        else:
            df_trust_patients.loc[i, 'unique_patient'] = unique_cluster_dict[row['pid']]

# check that all pids have a value in unique_patient ID
assert sum(pd.isnull(df_trust_patients['unique_patient'])) == 0
print(f"{df_trust_patients['unique_patient'].nunique()} unique patients across {df_trust_patients['pid'].nunique()} pids")

df_trust_patients['unique_patient'] = df_trust_patients['unique_patient'].astype(int)

# keep the first instance of each re-enrolled patient
df_trust_patients = df_trust_patients.sort_values(['unique_patient', 'pid']).drop_duplicates(subset='unique_patient', keep='first')


############################## STEP 4: REMOVE PATIENTS WHO CHANGED TREATMENT REGIMEN ##############################


# T0122 was diagnosed with MDR. 
# T0137 died within 3 weeks of starting treatment, but their ESP reason says 'Changed to "liver-friendly regimen". No longer on first line drugs'
# T0137 was already removed by the TCC exclusion criterion because they died early and don't have enough cultures sampled.
exclude_pids_resistance = ['T0122', 'T0137']

# T0322 says they were started on RHZE + levofloxacin
# T0330 is fine, says they had a positive month 5 culture and were RIF- and INH-sensitive
# T0355 started MDR regimen upon incarceration
exclude_pids_diff_regimen = ['T0322', 'T0355']

df_trust_patients = df_trust_patients.query("pid not in @exclude_pids_resistance & pid not in @exclude_pids_diff_regimen")

print(f"{df_trust_patients.pid.nunique()} patients remaining for analysis")

# this includes 1 patient who does not have measured WGS for RIF, INH, and EMB. That patient will be dropped from models that use measured MICs as covariates
df_trust_patients.to_csv("./data/TRUST_data_for_analysis.csv", index=False)
import numpy as np
import pandas as pd
import glob, os
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from sklearn.preprocessing import StandardScaler
import lifelines, itertools
from sklearn.model_selection import KFold

drugs_lst = ['RIF', 'INH', 'EMB', 'PZA']

# ordinal encoding: bl_afbprog --> smear
smear_encoding_dict = {6: np.nan, # I think this was already done in their data cleaning
                       5: np.nan, # I think this was already done in their data cleaning
                       0: 0, # no AFB
                       4: 1, # scanty
                       1: 2, # +
                       2: 3, # ++
                       3: 4, # +++
                      }


drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETA",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMI",
                  "Pretomanid": "PTM",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LEV",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

abbr_drug_dict = {val: key for key, val in drug_abbr_dict.items()}

cc_df = pd.read_csv("/n/data1/hms/dbmi/farhat/rollingDB/metadata/MIC/critical_concentrations_WHO_catalog.csv")

# MIC encoding from the TRUST codebook
MIC_encoding_dicts = {'RIF': {1: '0,0.03', 2: '0.03,0.06', 3: '0.06,0.125', 4: '0.125,0.25', 5: '0.25,0.5', 6: '0.5,1', 7: '1,inf'},
                      'INH': {1: '0,0.025', 2: '0.025,0.05', 3: '0.05,0.1', 4: '0.1,0.2', 5: '0.2,inf'},
                      'EMB': {1: '0,0.6', 2: '0.6,1.25', 3: '1.25,2.5', 4: '2.5,5', 5: '5,inf'},
                      'PZA': {1: '0,25', 2: '25,50', 3: '50,75', 4: '75,100', 5: '100,inf'}
                     }




def add_household_numbers(df_full):

    assert len(df_full.loc[pd.isnull(df_full['bl_housecode'])].query("bl_household==1")) == 0
    assert len(df_full.loc[~pd.isnull(df_full['bl_housecode'])].query("bl_household==0")) == 0
    
    household_num_dict = {} #dict(zip(df_full.dropna(subset='bl_housecode')['pid'], df_full.dropna(subset='bl_housecode')['bl_housecode']))
    
    household_num_iter = 0
    
    for household in df_full.dropna(subset='bl_housecode')['bl_housecode'].unique():
    
        household_num_dict[household] = household_num_iter
        household_num_iter += 1
    
    no_share_household_num_dict = {}
    
    pids_no_shared_household_contacts = df_full.query("bl_household==0").pid.values
    # len(pids_no_shared_household_contacts)
    
    for pid in pids_no_shared_household_contacts:
        no_share_household_num_dict[pid] = household_num_iter
        household_num_iter += 1
    
    # fill in the household number for all patients
    df_full['household_num'] = df_full['bl_housecode'].map(household_num_dict).fillna(df_full['pid'].map(no_share_household_num_dict))
    
    # check that there are no NaNs
    assert len(df_full.loc[pd.isnull(df_full['household_num'])]) == 0
    
    print(f"{df_full.dropna(subset='bl_housecode').pid.nunique()} patients share a household with another patient, accounting for {df_full.dropna(subset='bl_housecode').bl_housecode.nunique()} households")
    print(f"{df_full.household_num.nunique()} unique households across {df_full.pid.nunique()} patients")

    df_full['household_num'] = df_full['household_num'].astype(int)
    
    return df_full
    
    
    
    
def compute_outlier_bounds(vals_array):
    '''
    Calibrate so that there are no outliers at the low end
    '''
    
    # lower bound should be 0 because the F2 score can't be negative
    lb = np.min(vals_array)
    
    # upper bound is the same distance away from the median as lb is from the median
    ub = np.median(vals_array) + (np.median(vals_array) - lb)

    return lb, ub
    
    
    
    
def get_percent_long_involvement_predictions():

    pli_predictions = pd.read_csv("trust_normal_imgs_ensemble_model_pli_predicted_labels.csv")
    pli_predictions['outlier'] = 0
    
    pli_outlier_predictions = pd.read_csv("trust_outlier_imgs_ensemble_model_pli_predicted_labels.csv")
    pli_outlier_predictions['outlier'] = 1
    
    pli_predictions['pid'] = pli_predictions['patient_id'].str.split('_').str[0]
    pli_outlier_predictions['pid'] = pli_outlier_predictions['patient_id'].str.split('_').str[0]
    
    print(f"{len(set(pli_outlier_predictions.pid).union(pli_predictions.pid))} total pids")

    # preferentially keep 1B over 2B and non-outliers over outliers, in that order
    pli_predictions_combined = pd.concat([pli_predictions, pli_outlier_predictions]).sort_values(['pid', 'patient_id', 'outlier'], ascending=True).drop_duplicates('pid', keep='first').reset_index(drop=True)
    
    pli_predictions_combined['sample'] = pli_predictions_combined['patient_id'].str.split('_').str[-1]
        
    return pli_predictions_combined[['pid', 'predicted_label', 'outlier']]




def get_timika_score_predictions():

    timika_score_predictions = pd.read_csv("/n/data1/hms/dbmi/farhat/rs527/trust_project/CXR/timika/regression/trust_normal_imgs_ensemble_model_timika_predicted_labels.csv")
    timika_score_predictions['outlier'] = 0
    
    timika_score_outlier_predictions = pd.read_csv("/n/data1/hms/dbmi/farhat/rs527/trust_project/CXR/timika/regression/trust_outlier_imgs_ensemble_model_timika_predicted_labels.csv")
    timika_score_outlier_predictions['outlier'] = 1
    
    timika_score_predictions['pid'] = timika_score_predictions['patient_id'].str.split('_').str[0]
    timika_score_outlier_predictions['pid'] = timika_score_outlier_predictions['patient_id'].str.split('_').str[0]
    
    print(f"{len(set(timika_score_outlier_predictions.pid).union(timika_score_predictions.pid))} total pids")

    # preferentially keep 1B over 2B and non-outliers over outliers, in that order
    timika_score_predictions_combined = pd.concat([timika_score_predictions, timika_score_outlier_predictions]).sort_values(['pid', 'patient_id', 'outlier'], ascending=True).drop_duplicates('pid', keep='first').reset_index(drop=True)
    
    timika_score_predictions_combined['sample'] = timika_score_predictions_combined['patient_id'].str.split('_').str[-1]
        
    return timika_score_predictions_combined[['pid', 'predicted_label', 'outlier']]




def process_patient_metadata_better_encodings(df, TRUST_phenos, df_TTP_smear=None, include_TTP=False):
    '''
    This function makes better encodings for some columns of interest. 
    '''

    # ordinal encoding: bl_afbprog --> smear
    smear_encoding_dict = {6: np.nan, 
                           5: np.nan, 
                           0: 0, 
                           4: 1,
                           1: 2,
                           2: 3,
                           3: 4, 
                          }

    # smear grade at baseline. Use the concafb column because it uses the concentrated Ziehl-Neelsen method
    # which concentrates the sample and improves sensitivity for low smear samples
    smear_grade_cols = df.columns[df.columns.str.contains('s_concafb_sputum_specimen')]

    for col in smear_grade_cols:
        # last suffix is the sample number. Keep only 1-12
        sample_num = int(col.split('_')[-1])

        if sample_num > 12:
            del df[col]
            #print(col)
        else:
            df[col] = df[col].map(smear_encoding_dict)
    
    # df['smear_grade_baseline'] = df['s_concafb_sputum_specimen_1'].map(smear_encoding_dict)
    if include_TTP:
        df = df.merge(df_TTP_smear.query("culture_sample_num <= 5")[['pid', 'TTP']], how='left').rename(columns={'TTP': 'TTP_baseline'})
        df['TTP_baseline'] = df['TTP_baseline'].astype(float)
    
    # make 1s and 0s. 1 = male, 2 = female, so in the new model, 0 = female
    df['screen_sex'] = df['screen_sex'].replace(2, 0)
    
    cxr_finding_dict = {1:0, 2:1, 3:2, 4:3, 5:np.nan}
    
    cxr_infiltrate_dict = {0:0, 1:1, 2:2, 3:0.5, 4:np.nan}
    
    cxr_miliary_cavity_lymph_effusion_granuloma_dict = {1: 1, 0: 0, 2: np.nan}
    
    df['cxr_finding_chest_radiograph_1'] = df['cxr_finding_chest_radiograph_1'].map(cxr_finding_dict)
    #df['cxr_infiltrate_chest_radiograph_1'] = df['cxr_infiltrate_chest_radiograph_1'].map(cxr_infiltrate_dict)

    # bilateral is encoded as 2, so convert to binary variable, where 0 = not bilateral and 1 = bilateral
    df['bilateral_infiltrates'] = (df['cxr_infiltrate_chest_radiograph_1'] == 2).astype(int)
    df.loc[pd.isnull(df['cxr_infiltrate_chest_radiograph_1']), 'bilateral_infiltrates'] = np.nan
    
    for col in ['cxr_miliary_chest_radiograph_1', 
                'cxr_cavity_chest_radiograph_1',
                'cxr_lymph_chest_radiograph_1',
                'cxr_effusion_chest_radiograph_1',
                'cxr_granuloma_chest_radiograph_1'
               ]:
    
        df[col] = df[col].map(cxr_miliary_cavity_lymph_effusion_granuloma_dict)


    # 1 for diabetes, 0 for no
    # df.rename(columns={'bl_medhx___5': 'diabetes'}, inplace=True)
    df['diabetes'] = df['bl_medhx___5']

    # all MICs have been transformed to MGIT
    INH_resistant_pids = TRUST_phenos.query("INH_lower_bound >= 0.1").pid.unique()

    df['inh_resistant'] = df['pid'].isin(INH_resistant_pids).astype(int)

    # make sure the patients without a baseline MIC are NA
    df.loc[~df['pid'].isin(TRUST_phenos.pid.values), 'inh_resistant'] = np.nan

    # df['bl_inh_monoresistant'] = pd.to_numeric(df['bl_inh_monoresistant'], errors='coerce')

    # add this for imputation. Can't pass in CD4 only because it was not measured for HIV- patients
    df.loc[(df['bl_hiv']==0), 'HIV_CD4'] = 0
    df.loc[(df['bl_hiv']==1) & (df['bl_cd4'] >= 200), 'HIV_CD4'] = 1
    df.loc[(df['bl_hiv']==1) & (df['bl_cd4'] < 200), 'HIV_CD4'] = 2

    # convert these columns to integers because R will not read True/False from Python
    df['smoked_substance_use'] = pd.to_numeric(df['smoked_substance_use'], errors='coerce').astype(int)

    return df




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




# def get_all_available_baseline_MICs_single_drug(df_trust_patients, drug):

#     # only interested in the baseline MIC measurement
#     df_MIC_single_drug = df_trust_patients[['Original_ID', 'pid', 'SampleID', f's_{drug.lower()}mic_sputum_specimen_1', f's_{drug.lower()}mic_sputum_specimen_2']].set_index(['Original_ID', 'pid', 'SampleID'])

#     # get the first column
#     df_MIC_single_drug[drug] = df_MIC_single_drug.iloc[:, 0].values

#     # iterate through the remaining and fill NaNs
#     for col in df_MIC_single_drug.columns[1:]:
#         df_MIC_single_drug[drug] = df_MIC_single_drug[drug].fillna(df_MIC_single_drug[col])

#     # keep only patient IDs with a measured MIC
#     return df_MIC_single_drug.dropna(subset=drug).reset_index()




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




def convert_categorical_to_actual_MICs(df_categorical, drug, MIC_encoding_dict):

    df = df_categorical.copy()

    # convert the MIC categorical variable to a lower bound and upper bound
    df[[f"{drug}_lower_bound", f"{drug}_upper_bound"]] = df[drug].map(MIC_encoding_dict).str.split(",", expand=True).astype(float)

    drug_full_name = abbr_drug_dict[drug]

    # CNN PZA MICs were in MGIT
    if drug != 'PZA':

        # these were the only two methods used (7H11 = agar proportion method)
        cc_7H11 = cc_df.query("Drug==@drug_full_name & Medium == '7H11'")['Value'].values[0]
        cc_MGIT = cc_df.query("Drug==@drug_full_name & Medium == 'MGIT'")['Value'].values[0]

        # normalize everything to MGIT because that's the majority scale
        df['MGIT_CC'] = cc_MGIT
        df['measured_CC'] = df[f"{drug}_method"].map({'7H11': cc_7H11, 'MGIT': cc_MGIT})
        
        # make sure all measured MICs are in MGIT, so the INH 7H11 ones need to be converted
        df[f"{drug}_lower_bound"] *= df['MGIT_CC'] / df['measured_CC']
        df[f"{drug}_upper_bound"] *= df['MGIT_CC'] / df['measured_CC']

    df[f'{drug}_midpoint'] = np.round(np.mean(df[[f'{drug}_lower_bound', f'{drug}_upper_bound']], axis=1), 6)

    # for measured MICs like >2 µg/mL, make the midpoint the lower bound, not the mean of the lower bound and infinity
    df.loc[df[f'{drug}_upper_bound']==np.inf, f'{drug}_midpoint'] = df[f'{drug}_lower_bound']

    del_cols = ['MGIT_CC', 'measured_CC', drug, f"{drug}_method"]#f"{drug}_lower_bound", f"{drug}_upper_bound", drug]

    for col in del_cols:
        if col in df.columns:
            del df[col]
    
    return df




def pool_imputation_results(df, num_samples, coef_col, se_col, alpha=0.05, invert_OR=False):
    '''
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2727536/table/T2/
    '''
    
    num_impute = df.imp_num.max()
    num_covars = df.query("covariate != 'intercept'").covariate.nunique()
    
    # print(f"Pooling parameters for {num_covars} covariates across {num_impute} imputations fit on {num_samples} samples")

    # for the coefficients, pool them by taking the simple mean across imputations
    df_pool = pd.DataFrame(df.groupby("covariate")[coef_col].mean()).reset_index().rename(columns={coef_col: 'coef_pooled'})

    # for the standard errors, first get the within-imputation variance by taking the mean across imputations of the SQUARED standard errors
    df['squared_se'] = df[se_col]**2
    df_pool = df_pool.merge(pd.DataFrame(df.groupby("covariate")['squared_se'].mean()).reset_index().rename(columns={'squared_se': 'V_w'}))
    
    # then get the between-imputation variance with the following formula: (\sum_i=1^N \theta_i - \bar{\theta})/(N-1)
    df = df.merge(df_pool, on='covariate') # merge so that you can access coef_pooled column
    df['squared_diff'] = (df[coef_col] - df['coef_pooled'])**2

    # then take the sum of the squares of the difference and divide by N - 1
    df_pool = df_pool.merge(pd.DataFrame(df.groupby("covariate")['squared_diff'].sum() / (num_impute - 1)).reset_index().rename(columns={'squared_diff': 'V_b'}))

    # combine them to get total variance. Then SE_pooled = sqrt(V_total)
    df_pool['V_t'] = df_pool['V_w'] + df_pool['V_b'] + df_pool['V_b'] / num_impute

    df_pool['se_pooled'] = np.sqrt(df_pool['V_t'])

    # RIV = relative increase in variance
    df_pool['riv'] = df_pool['V_b'] * (1 + 1 / num_impute) / df_pool['V_w']
    
    # the p-value is according to the Wald test. Wald statistic = (coef_pooled - coef_null)^2 / V_T, where coef_null is 1 or 0, depending on whether this is an odds ratio or not
    # this is for univariate association, testing the alternative hypothesis that each covariate's coefficient is not 0
    df_pool['wald_stat'] = (df_pool['coef_pooled'] - 0) / df_pool['se_pooled']

    # this then follows the t-distribution with degrees of freedom computed with a complicated formula. This is the old definition of dof
    df_pool['dof_old'] = (num_impute - 1) * (1 + 1 / df_pool['riv'])**2

    # it was later adjusted in 1999 using the formula: df_adj = (df_old * df_observed) / (df_old + df_observed)
    # df_observed = ((n - k) + 1) / ((n - k) + 3) * (n - k) * (1 - \lambda), where n = number of samples, k = number of covariates
    # \lambda = (V_b + (V_b / num_impute)) / V_t
    # \lambda 1 / (1 + 1/r)
    # the tutorial says that dof_old is larger than the dofs for each imputed dataset, which is inappropriate. So we expect df_adj to be smaller than dof_old (check below)
    # smaller dof gives narrower distribution around the mean, increasing the likelihood of extreme values. So maybe go with the larger dof? 
    df_pool['lambda'] = (df_pool['V_b'] + (df_pool['V_b'] / num_impute)) / df_pool['V_t']
    df_pool['dof_observed'] = (num_samples - num_covars + 1) / (num_samples - num_covars + 3) * (num_samples - num_covars) * (1 - df_pool['lambda'])
    df_pool['dof_adj'] = (df_pool['dof_old'] * df_pool['dof_observed']) / (df_pool['dof_old'] + df_pool['dof_observed'])

    # old formula for computing dof is very strict and makes the dof very large. Adjusted dof should always be smaller
    assert len(df_pool.query("dof_adj >= dof_old")) == 0

    # multiply by 2 for two-sided p-value. sf = survival function = 1 - CDF. Want the proportion of the curve that is greater (so 1-CDF) than the test statistic
    df_pool['pval'] = 2 * st.t.sf(abs(df_pool['wald_stat']), df_pool['dof_old'])

    # compute confidence intervals. CI = coef_pooled ± t_stat * se_pooled
    # df_pool['t_critical'] = np.abs(st.t.ppf(1 - alpha / 2, df_pool['dof_adj']))
    df_pool['t_critical'] = np.abs(st.t.ppf(1 - alpha / 2, df_pool['dof_old']))
    df_pool['coef_lower'] = df_pool['coef_pooled'] - df_pool['t_critical'] * df_pool['se_pooled']
    df_pool['coef_upper'] = df_pool['coef_pooled'] + df_pool['t_critical'] * df_pool['se_pooled']

    # finally for interpretation, invert the hazard ratio (which is the exponentiated coefficient) so that >1 means associated with longer TCC (more hazardous, longer time to cure)
    if invert_OR:
        df_pool['HR_TCC_assoc'] = 1 / np.exp(df_pool['coef_pooled'])
        df_pool['HR_TCC_assoc_lower'] = 1 / np.exp(df_pool['coef_upper'])
        df_pool['HR_TCC_assoc_upper'] = 1 / np.exp(df_pool['coef_lower'])
    else:
        # or simply exponentiate the coefficient to get OR
        df_pool['OR'] = np.exp(df_pool['coef_pooled'])
        df_pool['OR_lower'] = np.exp(df_pool['coef_lower'])
        df_pool['OR_upper'] = np.exp(df_pool['coef_upper'])
    
    return df_pool




def forest_plot(df, val_col='OR', alpha=0.05, saveName=None):
    
    # Filter out the intercept and add a "significant" column if it doesn't exist
    df = df.query("covariate != 'intercept'")

    df.loc[df['pval'] <= alpha, 'significant'] = 1
    df['significant'] = df['significant'].fillna(0).astype(int)

    # improve the names for tick labels
    df['plot_column'] = df['covariate'].map({'INH_monoresistant': 'Isoniazid Resistant',
                                           'L1_L3': 'Lineages 1 or 3',
                                           'bilateral_infiltrates': 'Bilateral Disease',
                                           'bl_hiv': 'HIV',
                                           'bl_prevtb': 'Previous TB Disease',
                                           'bl_inh_monoresistant': 'INH Monoresistant',
                                           'cxr_cavity_chest_radiograph_1': 'Cavitation',
                                           'diabetes': 'Diabetes',
                                           'fstrom1_baseline': 'Smoking',
                                           'mixed_infect': 'Mixed Infection',
                                           'screen_sex': 'Male Sex',
                                           'screen_years': 'Age',
                                           'smear_pos_no_contam_sputum_specimen_1': 'Smear Positivity',
                                             'smear_positivity_baseline': 'Smear Positivity',
                                           'total_adherence': 'Adherence Rate',
                                           'underweight': 'Underweight',
                                           'EMB_pred_MIC': 'Predicted EMB MIC',
                                             'INH_pred_MIC': 'Predicted INH MIC',
                                             'PZA_pred_MIC': 'Predicted PZA MIC',
                                             'RIF_pred_MIC': 'Predicted RIF MIC',
                                             'EMB_midpoint': 'Measured EMB MIC',
                                             'INH_midpoint': 'Measured INH MIC',
                                             'RIF_midpoint': 'Measured RIF MIC',
                                             'EMB_pred_MIC_resistant': 'Resistant Predicted EMB MIC',
                                             'INH_pred_MIC_resistant': 'Resistant Predicted INH MIC',
                                             'PZA_pred_MIC_resistant': 'Resistant Predicted PZA MIC',
                                             'RIF_pred_MIC_resistant': 'Resistant Predicted RIF MIC',
                                             'EMB_midpoint_resistant': 'Resistant Measured EMB MIC',
                                             'INH_midpoint_resistant': 'Resistant Measured INH MIC',
                                             'PZA_midpoint_resistant': 'Resistant Measured PZA MIC',
                                             'RIF_midpoint_resistant': 'Resistant Measured RIF MIC',
                                             'RIF_AUC': 'Rifampicin PK AUC',
                                             'smear_grade_baseline': 'Smear Grade',
                                             'inh_resistant': 'INH Resistant',
                                             'cxr_miliary_chest_radiograph_1': 'Miliary TB',
                                             'cxr_finding_chest_radiograph_1': 'Abnormal Chest X-Ray Findings',
                                             'espR_indel': 'espR Indel',
                                             'adherence_12week': '12-Week Adherence Rate',
                                             'peth_value_baseline': 'PEth Value',
                                             'HIV_High_CD4': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$',
                                             'HIV_Low_CD4': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$',
                                             'bl_bmi': 'BMI',
                                             'TTP_baseline': 'TTP (Hours)',
                                             'smoked_substance_use': 'Smoked Substance Use',
                                             'predicted_label': '% Lung Involvement',
                                             'high_bacterial_burden': 'High Bacterial Burden',
                                             'smear_grade_1': 'Smear Grade',
                                             'bl_hiv_RIF_midpoint': 'HIV x Measured RIF MIC',
                                             'bl_hiv_INH_midpoint': 'HIV x Measured INH MIC',
                                             'bl_hiv_RIF_midpoint_resistant': 'HIV x Resistant Measured RIF MIC',
                                             'bl_prevtb_RIF_midpoint': 'Previous TB Disease x Measured RIF MIC',
                                             'bl_prevtb_INH_midpoint': 'Previous TB Disease x Measured INH MIC',
                                             'diabetes_RIF_midpoint': 'Diabetes x Measured RIF MIC',
                                             'diabetes_INH_midpoint': 'Diabetes x Measured INH MIC',
                                             'diabetes_RIF_midpoint_resistant': 'Diabetes x Resistant Measured RIF MIC',
                                             'mixed_infect_RIF_midpoint': 'Mixed Infection x Measured RIF MIC',
                                             'mixed_infect_INH_midpoint': 'Mixed Infection x Measured INH MIC',
                                             'bl_hiv_RIF_pred_MIC': 'HIV x Predicted RIF MIC',
                                             'bl_hiv_RIF_pred_MIC_resistant': 'HIV x Resistant RIF MIC',
                                             'bl_hiv_INH_pred_MIC': 'HIV x Predicted INH MIC',
                                             'bl_prevtb_RIF_pred_MIC': 'Previous TB Disease x Predicted RIF MIC',
                                             'bl_prevtb_RIF_pred_MIC_resistant': 'Previous TB Disease x Resistant Predicted RIF MIC',
                                             'bl_prevtb_INH_pred_MIC': 'Previous TB Disease x Predicted INH MIC',
                                             'diabetes_RIF_pred_MIC': 'Diabetes x Predicted RIF MIC',
                                             'diabetes_RIF_pred_MIC_resistant': 'Diabetes x Resistant Predicted RIF MIC',
                                             'diabetes_INH_pred_MIC': 'Diabetes x Predicted INH MIC',
                                             'mixed_infect_RIF_pred_MIC': 'Mixed Infection x Predicted RIF MIC',
                                             'mixed_infect_RIF_pred_MIC_resistant': 'Mixed Infection x Predicted RIF MIC',
                                             'mixed_infect_INH_pred_MIC': 'Mixed Infection x Predicted INH MIC',
                                             'F2': 'F2 Lineage Mixing Metric',
                                             'diabetes_bl_prevtb': 'Diabetes x Previous TB Disease',
                                             'HIV_High_CD4_INH_pred_MIC': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Predicted INH MIC',
                                             'HIV_High_CD4_RIF_pred_MIC_high': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Binarized Predicted RIF MIC',
                                             'HIV_High_CD4_bl_prevtb': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Previous TB Disease',
                                             'HIV_High_CD4_diabetes': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Diabetes',
                                             'HIV_Low_CD4_INH_pred_MIC': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Predicted INH MIC',
                                             'HIV_Low_CD4_RIF_pred_MIC_high': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Binarized Predicted RIF MIC',
                                             'HIV_Low_CD4_bl_prevtb': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Previous TB Disease',
                                             'HIV_Low_CD4_diabetes': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Diabetes',
                                             'HIV_High_CD4_INH_midpoint': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Measured INH MIC',
                                             'HIV_High_CD4_RIF_midpoint': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Measured RIF MIC',
                                             'HIV_Low_CD4_INH_midpoint': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Measured INH MIC',
                                             'HIV_Low_CD4_RIF_midpoint': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Measured RIF MIC',
                                             'HIV_High_CD4_RIF_pred_MIC': 'HIV, CD4 ≥ 200 cells/$\mathregular{mm^3}$ x Predicted RIF MIC',
                                             'HIV_Low_CD4_RIF_pred_MIC': 'HIV, CD4 < 200 cells/$\mathregular{mm^3}$ x Predicted RIF MIC',
                                             'Lineage_2': 'Lineage 2',
                                             'Lineage_3': 'Lineage 3',
                                             'Lineage_4': 'Lineage 4',
                                             'RIF_pred_MIC_ordinal': 'Predicted Rifampicin MIC, Ordinal',
                                             'high_lung_involvement': '>20% Lung Affected',
                                             'INH_lower_bound_resistant': 'INH Binary Resistance',
                                          })

    if sum(pd.isnull(df['plot_column'])) > 0:
        raise ValueError(f"{df.loc[pd.isnull(df['plot_column'])].covariate.values} don't have label mappings")
    
    # Sort by significance and value
    df = df.sort_values(["significant", val_col], ascending=[False, False]).reset_index(drop=True)
    
    # Separate significant and non-significant predictors
    significant_df = df.query("significant==1")
    non_significant_df = df.query("significant==0")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, len(df) * 0.6))

    conf_lower_col = f"{val_col}_lower"
    conf_upper_col = f"{val_col}_upper"
    
    # Plot significant predictors in blue
    ax.errorbar(
        significant_df[val_col], range(len(significant_df)),
        xerr=[significant_df[val_col] - significant_df[conf_lower_col], significant_df[conf_upper_col] - significant_df[val_col]],
        fmt='o', color='blue', ecolor='lightblue', capsize=3, label='Significant'
    )

    # Plot non-significant predictors in gray
    ax.errorbar(
        non_significant_df[val_col], range(len(significant_df), len(significant_df) + len(non_significant_df)),
        xerr=[non_significant_df[val_col] - non_significant_df[conf_lower_col], non_significant_df[conf_upper_col] - non_significant_df[val_col]],
        fmt='o', color='black', ecolor='gray', capsize=3, label='Non-Significant'
    )

    for i, row in df.iterrows():
        ax.text(df[conf_upper_col].max() * 1.025, i+0.1, f"p = {np.round(row['pval'], 2)}")

    # Customize plot appearance
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["plot_column"], fontsize=11)
    
    if 'HR' in val_col:
        ax.set_xlabel("Hazard Ratio for Longer TCC")
    else:
        ax.set_xlabel("Hazard Ratio")
    
    # Add a vertical reference line at OR = 1
    ax.axvline(x=1.0, color='red', linestyle='--', lw=0.75, zorder=0)
    
    sns.despine()
    plt.gca().invert_yaxis()
    
    # Show legend
    # ax.legend()

    # Show or save the plot
    if saveName is None:
        plt.show()
    else:
        plt.savefig(saveName, bbox_inches='tight')
        plt.close()



def read_combine_all_TRUST_data(patient_WGS_data_fName, drug_lineage_inclusion_dict, CNN_results_dir="/n/data1/hms/dbmi/farhat/Sanjana/CNN_results", F2_thresh=0.03, baseline_only=True):
    '''
    This function keeps only measured MICs and WGS samples taken in the first two weeks of treatment because we are interested in associating baseline characteristics with outcome.
    '''

    ############################################# STEP 1: READ IN THE COMBINED PATIENT-WGS DATAFRAME #############################################

    
    df_trust_patients = pd.read_csv(patient_WGS_data_fName)
    
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

    
    ###################################################### STEP 3: ADD LINEAGE ANNOTATIONS ######################################################

    
    # add additional lineage information on mixed infections and primary lineage group
    df_trust_patients['mixed_infect'] = (df_trust_patients['F2'] > F2_thresh).astype(int)

    df_trust_patients.loc[pd.isnull(df_trust_patients['F2']), 'mixed_infect'] = np.nan
    
    df_trust_patients.loc[df_trust_patients['Lineage'].isin(['1', '3', '1,3']), 'L1_L3'] = 1
    df_trust_patients.loc[df_trust_patients['Lineage'].isin(['2', '4', '2,4']), 'L1_L3'] = 0
    df_trust_patients.loc[df_trust_patients['Lineage'].isin(['1,2', '2,3', '3,4']), 'L1_L3'] = 0.5
    
    df_trust_patients.loc[pd.isnull(df_trust_patients['Lineage']), 'L1_L3'] = np.nan

    
    ###################################################### STEP 4: REMOVE PATIENTS WITH SEQUENCING AT THE SAME TIMEPOINT WITH DIFFERENT LINEAGES ###########################
    

    pids_multiple_sequences_same_timepoint = df_trust_patients.iloc[df_trust_patients.index.values[df_trust_patients.duplicated('Original_ID', keep=False)]].pid.unique()

    pids_multiple_sequences_same_timepoint_discordant_lineages = pd.DataFrame(df_trust_patients.groupby(['pid', 'Original_ID'])['Coll2014'].nunique()).query("Coll2014 > 1").reset_index().pid.values
    
    print(f"{len(pids_multiple_sequences_same_timepoint)} patients: {pids_multiple_sequences_same_timepoint} have multiple sequences at the same timepoint")
    print(f"Removing {len(pids_multiple_sequences_same_timepoint_discordant_lineages)} patients: {pids_multiple_sequences_same_timepoint_discordant_lineages} because there are multiple WGS samples at the same timepoint with different lineages")

    df_trust_patients = df_trust_patients.query("pid not in @pids_multiple_sequences_same_timepoint_discordant_lineages")
    
    
    ###################################################### STEP 5: READ IN ALL AVAILABLE MEASURED MICS ######################################################

    
    TRUST_phenos = []
    
    for drug in drugs_lst:
        
        # this contains all WGS runs, so keep only the unique pids for counting/plotting purposes
        # all MICs here were converted to MGIT. The only drug for which that makes any difference is INH, whose MGIT (0.1) and 7H10 (0.2) critical concentrations are different.
        df_single_drug = get_all_available_MICs_single_drug(df_trust_patients, drug, baseline_only=baseline_only)
        df_single_drug = df_single_drug[['pid', drug, f"{drug}_method"]]
        
        print(f"{len(df_single_drug)} patients have MICs for {drug}")
        TRUST_phenos.append(df_single_drug)
    
    TRUST_phenos = reduce(lambda left, right: pd.merge(left, right, on='pid', how='outer'), TRUST_phenos).drop_duplicates()
    
    for drug in drugs_lst:
        if drug in TRUST_phenos.columns:
            TRUST_phenos = convert_categorical_to_actual_MICs(TRUST_phenos, drug, MIC_encoding_dicts[drug])

    if baseline_only:
        print(f"{TRUST_phenos.pid.nunique()} patients have measured MICs in the first 2 weeks\n")
    else:
        print(f"{TRUST_phenos.pid.nunique()} patients have measured MICs\n")

    
    #################################################### STEP 6: READ IN ALL AVAILABLE PREDICTED MICS #########################################################

    
    df_pred_combined = []
    
    for drug in drugs_lst:

        # which models to use for each drug
        assert drug_lineage_inclusion_dict[drug] in ['lineage_amino_acid', 'amino_acid']
        df_pred = pd.read_csv(os.path.join(CNN_results_dir, f"{drug}_{drug_lineage_inclusion_dict[drug]}", "TRUST", "test_predictions.csv")).rename(columns={'ROLLINGDB_ID': 'SampleID', 'pred_MIC': f'{drug}_pred_MIC'})

        # drop patient duplicates (because multiple WGS samples per pid)
        df_pred = df_pred.merge(df_trust_patients[['SampleID', 'Original_ID', 'pid']]).sort_values(['pid', 'Original_ID']).drop_duplicates('pid', keep='first').reset_index(drop=True)
    
        print(f"Found predicted {drug} MICs for {len(df_pred)} pids")
        
        df_pred_combined.append(df_pred[['pid', f'{drug}_pred_MIC']])
    
    df_pred_combined = reduce(lambda left, right: pd.merge(left, right, on='pid', how='outer'), df_pred_combined)
    print(f"{df_pred_combined.pid.nunique()} patients have predicted MICs")

    # these should have all the same patients because the predicted MICs come from the high-quality WGS samples 
    assert len(set(df_trust_patients.pid).symmetric_difference(df_pred_combined.pid)) == 0

    # categorical variable for this, interesting to look at by plotting
    df_trust_patients.loc[(df_trust_patients['bl_hiv']==0), 'HIV_CD4'] = 0
    df_trust_patients.loc[(df_trust_patients['bl_hiv']==1) & (df_trust_patients['bl_cd4'] >= 200), 'HIV_CD4'] = 1
    df_trust_patients.loc[(df_trust_patients['bl_hiv']==1) & (df_trust_patients['bl_cd4'] < 200), 'HIV_CD4'] = 2

    return df_trust_patients, TRUST_phenos, df_pred_combined



def get_combined_patient_WGS_data_only(patient_WGS_data_fName):
    '''
    Keep all patients with high quality WGS, regardless of when it was measured. This is to see if the re-enrollments have different or same lineages as before
    '''
    df_trust_patients = pd.read_csv(patient_WGS_data_fName)
    
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

    return df_trust_patients



def determine_MIC_binarization_threshold(df, col, include_median=False, verbose=False):
    '''
    Use this function to determine a threshold to binarize MICs -- predicted or measured -- into two equal sized groups. 
    '''

    df_copy = df.copy()

    # just return the median??? which is the middle value
    bisection_value = np.median(df_copy[col].dropna())

    # use > because if you exclude the median, then because many isolates have the value, the high class will have fewer
    # better to have the high class (which is value = 1) to have fewer isolates that the baseline class
    # for the predicted MICs, seems better to include the median, but for measured MICs, exclude it
    if include_median:
        df_copy[f"{col}_high"] = (df_copy[col] >= bisection_value).astype(int)
    else:
        df_copy[f"{col}_high"] = (df_copy[col] > bisection_value).astype(int)
        
    df_copy.loc[pd.isnull(df_copy[col]), f"{col}_high"] = np.nan

    if verbose:
        print(f"Binarized {col} at {bisection_value} µg/mL, {np.round(df_copy[f'{col}_high'].dropna().mean()*100, 1)}% are high")

    # remove the original column from the dataframe
    del df_copy[col]
    
    return df_copy





def dummy_encode_lineages(df, lineage_col):
    '''
    This will include mixed infections if listed in the Coll2014 column
    '''
    
    df = df.reset_index(drop=True)
    split_lineages_dict = {}
    
    unique_lineages = []
    
    for i, row in df.iterrows():
        
        lineage = row[lineage_col]
        lineages_lst = []
    
        # this will separate the lineages in the mixed infections and also make a list if it is not mixed. Keep as strings (not ints) in case of non-integer lineage names
        for split_lineage in lineage.split(','):
        
            # weird case that they get converted to strings of floats sometimes, like '1.0'
            try:
                split_lineage = str(int(float(split_lineage)))
            except:
                # leave it as is
                split_lineage = split_lineage
                
            lineages_lst.append(split_lineage)
            split_lineages_dict[row['SampleID']] = lineages_lst
        
        unique_lineages += lineages_lst
        
    # these will be the columns to dummy encode
    unique_lineages = np.sort(np.unique(unique_lineages))
    unique_lineages = [f'Lineage_{num}' for num in unique_lineages]
    
    df_lineage = pd.DataFrame(columns = ['pid', 'SampleID'] + list(unique_lineages))
    df_lineage[['pid', 'SampleID']] = df[['pid', 'SampleID']]
    df_lineage = df_lineage.set_index('SampleID')
    
    for SampleID, split_lineages in split_lineages_dict.items():
        
        for lineage in split_lineages:
            df_lineage.loc[SampleID, f'Lineage_{lineage}'] = 1
    
    df_lineage = df_lineage.reset_index().set_index(['pid', 'SampleID']).fillna(0).astype(int)
    
    # add column denoting the number of lineages per pid (relevant for mixed infections)
    df_lineage['num_lineages'] = df_lineage.sum(axis=1)
    
    # remove extra columns
    df_lineage = df_lineage.reset_index()[['pid'] + list(unique_lineages)]
    
    # remove the column with the most 0s to consider that the baseline
    baseline_lineage = df_lineage[unique_lineages].sum(axis=0).sort_values().index.values[0]
    
    del df_lineage[baseline_lineage]
    
    return df_lineage




def process_input_features_for_model(df, model_cols, MIC_type='none', binarize_MICs=False, include_drugs=[], include_interactions=False, RIF_MIC_ordinal_groups=[]):

    df_model = df.copy()
    cols_lst = model_cols.copy()

    df_model['high_lung_involvement'] = (df_model['predicted_label'] > 20).astype(int)

    # this is the imputed smear grade sample 1. Mapping using smear_encoding_dict has already been done
    if 'smear_grade_1' in df_model.columns:
        df_model['smear_grade_baseline'] = df_model['smear_grade_1'].copy()
    
    elif 'smear_positivity_1' in df_model.columns:
        df_model['smear_positivity_baseline'] = df_model['smear_positivity_1'].copy()
        
    # dummy encode lineages, accounting for mixed infections
    if 'Lineage' in model_cols:
        df_lineage = dummy_encode_lineages(df_model, 'Lineage')
        df_model = df_model.merge(df_lineage)
        
        # remove this from the list of predictors and add in the dummmy-encoded lineage names
        cols_lst.remove('Lineage')
        cols_lst += list(set(df_lineage.columns) - set(['pid']))

    if len(RIF_MIC_ordinal_groups) > 0 and MIC_type == 'predicted':

        for idx, num in enumerate(RIF_MIC_ordinal_groups):
            if idx == 0:
                df_model.loc[df_model['RIF_pred_MIC'] <= num, 'RIF_pred_MIC_ordinal'] = idx
            else:
                df_model.loc[(df_model['RIF_pred_MIC'] > RIF_MIC_ordinal_groups[idx-1]) & (df_model['RIF_pred_MIC'] <= num), 'RIF_pred_MIC_ordinal'] = idx

        # top category
        df_model.loc[df_model['RIF_pred_MIC'] > RIF_MIC_ordinal_groups[-1], 'RIF_pred_MIC_ordinal'] = len(RIF_MIC_ordinal_groups)

        MIC_cols = ['RIF_pred_MIC_ordinal']
        
        binarize_cols = []        
    
    # binarize MICs at the critical concentration
    elif binarize_MICs and MIC_type != 'none':
        
        if MIC_type == 'predicted':
            binarize_cols = [f"{drug}_pred_MIC" for drug in include_drugs]
            # include_median = False
        
        elif MIC_type == 'measured':
            binarize_cols = [f"{drug}_lower_bound" for drug in include_drugs]
            # include_median = False

        for col in binarize_cols:

            # get the critical concentration to binarize each MIC column at
            drug = col.split('_')[0]
            full_drug_name = abbr_drug_dict[drug]

            if drug == 'PZA':
                cc = cc_df.query("Drug==@full_drug_name & Medium=='MGIT'").Value.values[0]
            else:
                cc = cc_df.query("Drug==@full_drug_name & Medium=='7H10'").Value.values[0]
            # df_model = determine_MIC_binarization_threshold(df_model, col, include_median=include_median)

            if '_lower_bound' in col:
                df_model[f"{col}_resistant"] = (df_model[col] >= cc).astype(int)
            elif '_midpoint' in col:
                df_model[f"{col}_resistant"] = (df_model[col] > cc).astype(int)
            elif 'pred_MIC' in col:
                df_model[f"{col}_resistant"] = (df_model[col] > cc).astype(int)
            
            df_model.loc[pd.isnull(df_model[col]), f"{col}_resistant"] = np.nan
        
            # if verbose:
            #     print(f"Binarized {col} at {cc} µg/mL, {np.round(df_model[f'{col}_resistant'].dropna().mean()*100, 1)}% are resistant")

        MIC_cols = [f"{col}_resistant" for col in binarize_cols]
        
    else:
        if MIC_type == 'predicted':
            MIC_cols = [f"{drug}_pred_MIC" for drug in include_drugs]

        elif MIC_type == 'measured':
            MIC_cols = [f"{drug}_midpoint" for drug in include_drugs]
        else:
            MIC_cols = []
        
    if include_interactions:
        interaction_tuples = [('diabetes', 'bl_prevtb')]
        # interaction_tuples = []fdsa
        interact_group1 = ['diabetes', 'bl_prevtb']
        
        interaction_tuples += list(itertools.product(interact_group1, MIC_cols))

        # add interaction terms
        for col1, col2 in interaction_tuples:
            
            # if col1 in df_model.columns and col2 in df_model.columns:
            if col1 in model_cols and col2 in model_cols:
                df_model[f"{col1}_{col2}"] = df_model[col1] * df_model[col2]

                # add the interactions between MICs and patient covariates to cols_lst 
                cols_lst += [f"{col1}_{col2}"]

    model_features_lst = np.unique(cols_lst + MIC_cols) 
    # print(f"All columns: {model_features_lst}")

    # HIV_CD4 is a categorical variable, so need to dummy encode
    if 'HIV_CD4' in model_features_lst:
        
        # print(df_model['HIV_CD4'].value_counts(dropna=False))

        # this removes the first column (which is No HIV), so no HIV is the baseline
        df_model = pd.get_dummies(df_model, columns=['HIV_CD4'], drop_first=True)
        
        # rename to easier names
        df_model = df_model.rename(columns={'HIV_CD4_1.0': 'HIV_High_CD4', 'HIV_CD4_2.0': 'HIV_Low_CD4'})
        
        # convert from bool to int
        df_model[['HIV_High_CD4', 'HIV_Low_CD4']] = df_model[['HIV_High_CD4', 'HIV_Low_CD4']].astype(int)
    
        # remove HIV_CD4 from model_features_lst and add in the dummy variables
        model_features_lst = list(set(model_features_lst) - set(['HIV_CD4']))
        model_features_lst += ['HIV_High_CD4', 'HIV_Low_CD4']

        # separately interact MICs with HIV_CD4 variable after they have been dummy encoded
        if include_interactions:

            hiv_interaction_tuples = [('HIV_High_CD4', 'bl_prevtb'), ('HIV_Low_CD4', 'bl_prevtb'), ('HIV_High_CD4', 'bl_prevtb'), ('HIV_Low_CD4', 'bl_prevtb')]

            if MIC_type == 'predicted':
    
                interact_group1 = ['HIV_High_CD4', 'HIV_Low_CD4']
                hiv_interaction_tuples += list(itertools.product(interact_group1, MIC_cols))
    
            elif MIC_type == 'measured':
            
                interact_group1 = ['HIV_High_CD4', 'HIV_Low_CD4']
                hiv_interaction_tuples += list(itertools.product(interact_group1, MIC_cols))

            for (col1, col2) in hiv_interaction_tuples:
                df_model[f"{col1}_{col2}"] = df_model[col1] * df_model[col2]
                model_features_lst += [f"{col1}_{col2}"]

    if 'high_bacterial_burden' in model_features_lst:
        df_model.loc[(df_model['TTP_baseline'] <= 200), 'high_bacterial_burden'] = 1
        df_model.loc[(pd.isnull(df_model['TTP_baseline'])) & (df_model['smear_grade_baseline'] >= 3), 'high_bacterial_burden'] = 1
        df_model['high_bacterial_burden'] = df_model['high_bacterial_burden'].fillna(0).astype(int)
        # print(df_model['high_bacterial_burden'].value_counts())

    df_model = df_model.set_index('pid')[model_features_lst].reset_index() 

#     # there shouldn't be any NaNs because we imputed the predictors
#     for col in df.columns:
#         if sum(pd.isnull(df[col])) > 0:
#             print(f"Column {col} has NAs")
            
    # print(df_model.pid.nunique())
    # assert len(df) == len(df.dropna())
    df_model = df_model.dropna()
    # print(df_model.pid.nunique())
    
    # log transform age, BMI, TTP, PETH, and MICs. Also get all columns from model_features_lst (MICs and interactions of them) with the proper suffixes. The binarized variables will end with "_high"
    # log_transform_cols = np.unique(['screen_years', 'bl_bmi', 'peth_value_baseline', 'F2', 'RIF_AUC', 'TTP_baseline', 'predicted_label'] + [col for col in model_features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])

    # variables with a skew coef wth an absolute values >= 2
    log_transform_cols = np.unique(['F2', 'total_adherence', 'adherence_12week', 'RIF_AUC'] + [col for col in model_features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
    
    for col in log_transform_cols:

        if col in df_model.columns:
            
            df_model[col] = np.log2(df_model[col].astype(float))

            # smallest value is 0, so replace with 0 after you do the log-transform. This makes the distribution more continuous than i.e. replacing 0 with a very small value
            # because the log of that very small value will be very negative instead of close to 0. It will be computed as -inf
            df_model[col] = df_model[col].replace(-np.inf, 0)
            
    # drop duplicate columns if there are any
    df_model = df_model.loc[:, ~df_model.columns.duplicated(keep='first')]

    return df_model.drop_duplicates(), model_features_lst
    
    
    
def fit_cox_hazard_ratio_model(df, df_outcome, cols_lst, event_col, time_col, MIC_type='none', binarize_MICs=False, include_drugs=drugs_lst, penalize_model=False, include_interactions=False, RIF_MIC_ordinal_groups=[], stratify_variables=None, non_linear_term_variables=[], cluster_col=None):

    # Process input features
    df_model_processed, features_lst = process_input_features_for_model(df, cols_lst, MIC_type=MIC_type, binarize_MICs=binarize_MICs, include_drugs=include_drugs, include_interactions=include_interactions, RIF_MIC_ordinal_groups=RIF_MIC_ordinal_groups)
    
    # Add outcome data
    df_model_processed = df_model_processed.merge(df_outcome, on='pid')
    df_model_processed_save = df_model_processed.copy()

    # remove any columns that are the same everywhere to reduce model fitting time
    remove_cols = df_model_processed.columns[df_model_processed.nunique() == 1]
    # print(f"    Removing features {remove_cols} because they are the same everywhere")
    features_lst = list(set(features_lst) - set(remove_cols))

    # keep track of these for un-normalizing the final odds ratios
    means_dict = dict(df_model_processed[features_lst].mean(axis=0))
    std_dict = dict(df_model_processed[features_lst].std(axis=0))
    
    # Normalize features
    scaler = StandardScaler()
    df_model_processed[features_lst] = scaler.fit_transform(df_model_processed[features_lst])

    cph = lifelines.CoxPHFitter()
    
    for col in features_lst + [time_col, event_col, 'unique_patient']:
        num_na = sum(pd.isnull(df_model_processed[col]))

        if num_na > 0:
            print(col, num_na, df_model_processed.loc[pd.isnull(df_model_processed[col])])

    # need to define the cubic basis splines with a formula call to fit(). Use the training dataframe to get the bounds because it will have had log-transforms and standard scaling done to it
    non_linear_term_vars_min = [int(np.floor(df_model_processed[variable].min())) for variable in non_linear_term_variables]
    non_linear_term_vars_max = [int(np.ceil(df_model_processed[variable].max())) for variable in non_linear_term_variables]

    # remove them from the features list
    linear_term_variables = list(set(features_lst) - set(non_linear_term_variables))

    # if you're stratifying by categorical variables, have to remove them from the formula
    if stratify_variables is not None:
        stratify_variables = [col for col in stratify_variables if col in features_lst]
        linear_term_variables = list(set(linear_term_variables) - set(stratify_variables))

    # combine the features without non-linear terms into a string
    model_formula = " + ".join(linear_term_variables).strip(' ')

    # add the variables with potential non-linear effects to the formula
    # lifelines uses cubic splines, which is the default because you get smoothness but also not too many parameters
    # degrees of freedom = k + d, where d is the polynomial degree (in this case, 3) and k is the number of values in the interval being tested
    for i, variable in enumerate(non_linear_term_variables):
        if variable in features_lst:
            lb = non_linear_term_vars_min[i]
            ub = non_linear_term_vars_max[i]
            k = len(np.arange(lb, ub+1))
            model_formula += f" + bs({variable}, df={k}, lower_bound={lb}, upper_bound={ub}, degree=3)"
        
    cph.fit(df_model_processed,
            duration_col=time_col, 
            event_col=event_col, 
            cluster_col=cluster_col,
            fit_options={'step_size': 0.1},
            strata=stratify_variables,
            formula=model_formula
           )
    
    # Get results
    df_model_results = cph.summary

    # Last step: undo the variable transformations. First, we log2-transformed, then standard-scaled. So have to indo in the reverse order
    # 1) Undo the standard-scaling
    df_model_results['original_mean'] = df_model_results.index.map(means_dict)
    df_model_results['original_std'] = df_model_results.index.map(std_dict)

    # have to add in the mean and std (from the original variables) to each spline of the non linear variables
    for variable in non_linear_term_variables:
        df_model_results.loc[df_model_results.index.str.contains(variable), 'original_mean'] = means_dict[variable]
        df_model_results.loc[df_model_results.index.str.contains(variable), 'original_std'] = std_dict[variable]
    
    df_model_results['coef_transformed'] = df_model_results['coef'] / df_model_results['original_std']
    df_model_results['se_transformed'] = df_model_results['se(coef)'] / df_model_results['original_std']

    # 2) Undo the log2-transform for the variables that were log2-transformed
    # To do this, exponentiate the coefficients, so 2**coef. SE is approximately ln(2) * 2**coef * SE(coef)
    # log_transform_cols = np.unique(['screen_years', 'bl_bmi', 'peth_value_baseline', 'F2', 'RIF_AUC', 'TTP_baseline', 'predicted_label'] + [col for col in features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
    log_transform_cols = np.unique(['F2', 'total_adherence', 'adherence_12week', 'RIF_AUC'] + [col for col in features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
    
    # the current coefficient is the factor increase if the value is multiplied by the base. i.e. if log2-transformed with beta = 2, then a doubling of x leads to a 2 * 2 = 4 multiplier on the log HR
    # so to scale it to the original scale, you would multiply by the base of the logarithm you took
    df_model_results.loc[df_model_results.index.isin(log_transform_cols), 'coef_transformed'] = 2 * df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['coef_transformed'] #np.exp(df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['coef_transformed'])

    # is it the same transformation?
    df_model_results.loc[df_model_results.index.isin(log_transform_cols), 'se_transformed'] = 2 * df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['se_transformed'] 

    return df_model_results, df_model_processed, cph
    


def fit_cox_models_all_imputations(df_trust_patients, df_pred_combined, TRUST_phenos, df_imputed, df_outcome, cols_lst, event_col, time_col, alpha=0.05, invert_OR=True, exclude_resistance=False, MIC_type=None, binarize_MICs=False, include_drugs=[], penalize_model=False, tb_deaths_only=False, include_interactions=False, original_dataset_only=False, stratify_variables=None, non_linear_term_variables=[], cluster_col=None):

    coef_col = 'coef_transformed'
    se_col = 'se_transformed'

    df_estimates = []

    # keep track of log-likelihoods across imputations because need to match them to get the difference between nested models
    log_like_imputations = []
    cph_models_imputations = []
    df_model_processed_imputations = [] # need to keep track of the individual training datasets as well for testing the proportional hazards assumptions
    
    # restrict to samples that were collected in the first 2 weeks
    df_WGS_info = df_trust_patients[['pid', 'Original_ID', 'SampleID', 'Lineage', 'F2']]

    # preferentialy keep the first WGS sample for each one. Sort by Lineage with NAs last to preferentially keep the first uncontaminated sample. Contaminated samples are NA in the Lineage column
    df_WGS_info = df_WGS_info.sort_values(['pid', 'Lineage', 'Original_ID'], na_position='last').drop_duplicates(subset=['pid'], keep='first')

    # only use the original dataset
    if original_dataset_only:
        imp_lst = [0]
    else:
        imp_lst = np.sort(df_imputed['.imp'].unique())[1:]
    
    # skip the first one because it's the initializing one. There will be N + 1 unique values in the .imp column
    for imp_num in imp_lst:

        # combine with lineage and F2 information and remove highly contaminated WGS (which will be NaN in those columns)
        df_single_imputation = df_imputed.loc[df_imputed['.imp']==imp_num].merge(df_WGS_info[['pid', 'Lineage', 'SampleID', 'F2']])

        if 'imp_num' in df_outcome.columns:
            df_outcome_single_imputation = df_outcome.query("imp_num==@imp_num")
        else:
            df_outcome_single_imputation = df_outcome.copy()
        
        # add measured and predicted MICs, then the process_input_features_for_model function will keep only those that should be here, determined by the MIC_type argument
        if MIC_type == 'predicted':
            df_single_imputation = df_single_imputation.merge(df_pred_combined, on='pid')
        elif MIC_type == 'measured':
            df_single_imputation = df_single_imputation.merge(TRUST_phenos, on='pid')

        if exclude_resistance:
            
            # remove all pids with measured resistant MICs
            for col in df_single_imputation.columns[df_single_imputation.columns.str.contains('_upper_bound')]:
                
                drug = col.split("_")[0]

                drug_full_name = abbr_drug_dict[drug]
                
                # all measured MICs have been normalized to MGIT, which is the majority medium. So get the critical concentration for MGIT
                mgit_cc = cc_df.query("Medium=='MGIT' & Drug==@drug_full_name").Value.values[0]
    
                # keep only pids where the upper bound is less than or equal to the critical concentration.
                # Do the inverse (exclude pids where the upper bound is greater than the CC because of MICs that are NA
                # print(drug, mgit_cc, len(df_single_imputation.query(f"~({drug}_upper_bound > @mgit_cc)")))
                num_measured_resistant = df_single_imputation.query(f"{drug}_upper_bound > @mgit_cc").pid.nunique()
                df_single_imputation = df_single_imputation.query(f"~({drug}_upper_bound > @mgit_cc)")

                isolates_with_genotypic_resistance = df_WHO_variants.query("drug==@drug_full_name").SampleID.values
                num_genotypic_resistant = df_single_imputation.query("SampleID in @isolates_with_genotypic_resistance").pid.nunique()
                df_single_imputation = df_single_imputation.query("SampleID not in @isolates_with_genotypic_resistance")

                if imp_num in [0, 1]:
                    print(f"    Dropped {num_measured_resistant} patients with measured and {num_genotypic_resistant} patients with genotypic {drug} resistance")

        if tb_deaths_only:

            non_tb_death_pids = df_outcome_single_imputation.query("event_type == 'death' & TB_death == 0").pid.unique()
            
            if imp_num in [0, 1]:
                print(f"    Censored {len(non_tb_death_pids)} patients with non-TB deaths")
    
            # censor the patients who died of non-TB causes
            df_outcome_single_imputation.loc[df_outcome_single_imputation['pid'].isin(non_tb_death_pids), 'event'] = 0
            
        df_model_results, df_model_processed, cph = fit_cox_hazard_ratio_model(df_single_imputation, 
                                                                              df_outcome_single_imputation,
                                                                             cols_lst,
                                                                             event_col,
                                                                             time_col,
                                                                             MIC_type=MIC_type,
                                                                             penalize_model=penalize_model,
                                                                             binarize_MICs=binarize_MICs,
                                                                             include_drugs=include_drugs,
                                                                             include_interactions=include_interactions,
                                                                             stratify_variables=stratify_variables,
                                                                               non_linear_term_variables=non_linear_term_variables,
                                                                               cluster_col=cluster_col
                                                                         )
        
        # save all the models for checking the proportional hazards assumptions
        cph_models_imputations.append(cph)

        df_model_processed['imp_num'] = imp_num
        df_model_processed_imputations.append(df_model_processed)

        assert len(df_model_processed) == df_model_processed.pid.nunique()
        
        if imp_num in [0, 1]:
            print(f"    {df_model_processed.pid.nunique()} total pids")

            if len(non_linear_term_variables) > 0:
                print(f"    Allowing non-linear associations for {non_linear_term_variables}")
            # print(f"    Fitting Cox proportional hazards model with L2 penalty of {best_penalizer}")

        # need to keep the original standard deviation (keep mean for comparison sake) to transform later. But need to transform after 
        df_save = df_model_results[['coef', 'exp(coef)', 'se(coef)', 'coef_transformed', 'se_transformed']]
        df_save['imp_num'] = imp_num
        df_estimates.append(df_save.reset_index())

    df_estimates = pd.concat(df_estimates)

    # finally, pool results across the models. Doesn't matter which dataframe you take the length of, they all have the same pids
    if not original_dataset_only:
        return pool_imputation_results(df_estimates, df_imputed.pid.nunique(), coef_col, se_col, alpha=alpha, invert_OR=invert_OR), df_model_processed_imputations, cph_models_imputations
        #return df_estimates, df_model_processed_imputations, cph_models_imputations
    else:
        # add p-values. The Wald stat follows the normal distribution
        # survival function = 1 - CDF, which is the probability of being greater than the critical value. Take the absolute value so that you get the positive z-score
        df_estimates['pval'] = 2 * st.norm.sf(np.abs(df_estimates[coef_col] / df_estimates[se_col]))

        # add confidence intervals
        df_estimates['OR'] = np.exp(df_estimates[coef_col])
        df_estimates['OR_lower'] = np.exp(df_estimates[coef_col] + st.norm.ppf(alpha / 2) * df_estimates[se_col])
        df_estimates['OR_upper'] = np.exp(df_estimates[coef_col] + st.norm.ppf(1 - alpha / 2) * df_estimates[se_col])
        
        return df_estimates, df_model_processed, cph






def LRT_multiple_imputations(ll_large_model, ll_small_model, num_param_large, num_param_small):
    '''
    Arguments: 1) array of log-likelihoods of the large model, and 2) array of log-likelihoods of the small model
    '''

    try:
        assert num_param_large > num_param_small
    except:
        raise ValueError(f"Numbers of parameters are inconsistent: Large = {num_param_large}, small = {num_param_small}")
        
    assert len(ll_large_model) == len(ll_small_model)

    num_impute = len(ll_large_model)
    log_like_ratio = 2 * (np.array(ll_large_model) - np.array(ll_small_model))

    assert np.min(log_like_ratio) >= 0

    log_like_ratio_mean = np.mean(log_like_ratio)

    # between-imputation variance
    V_b = 1 / (num_impute - 1) * np.mean((log_like_ratio - log_like_ratio_mean)**2)

    # total variance
    V_t = log_like_ratio_mean + (1 + 1 / num_impute) * V_b

    dof = (num_impute - 1) * log_like_ratio_mean ** 2 / V_b**2

    # F distribution statistic
    F_stat = V_t / (num_param_large - num_param_small)

    return st.f.sf(F_stat, num_param_large - num_param_small, dof)




def run_LRT_single_predictor(test_covariate, df_trust_patients, df_pred_combined, TRUST_phenos, df_imputed_outcomes, df_final, cols_lst, event_col, time_col, tb_deaths_only=False, MIC_type='none', binarize_MICs=True, include_drugs=[], stratify_variables=None, non_linear_term_variables=[], cluster_col=None):

    if test_covariate in cols_lst:
        updated_cols_lst = cols_lst.copy()
        updated_cols_lst.remove(test_covariate)
        assert len(updated_cols_lst) == len(cols_lst) - 1

        # same empty drugs list for both models because we're testing a patient predictor, not an MIC
        include_drugs = []

    # then test_covariate must be a drug MIC
    else:
        if MIC_type == 'none':
            raise ValueError(f"{test_covariate} is not in cols_lst and is not an MIC")

        # unchanged because we're testing the effect of removing a drug MIC predictor from the full model
        updated_cols_lst = cols_lst.copy()
    
        # here we're testing the effect of adding a drug variable to the model. So the first model (large model) will have the drug variable, and the second model will not
        assert len(include_drugs) > 0

    # larger model
    df_results_1, _, cph_imputations_1 = fit_cox_models_all_imputations(
                                                                        df_trust_patients, 
                                                                        df_pred_combined,
                                                                     TRUST_phenos,
                                                                     df_imputed_outcomes,
                                                                      df_final,
                                                                      cols_lst,
                                                                      event_col = event_col,
                                                                      time_col = time_col,
                                                                      penalize_model=False,
                                                                      alpha = 0.05,
                                                                      exclude_resistance=False,
                                                                      tb_deaths_only=tb_deaths_only,
                                                                      MIC_type=MIC_type,
                                                                      binarize_MICs=binarize_MICs,
                                                                      include_drugs=include_drugs,
                                                                      invert_OR=False,
                                                                        stratify_variables=stratify_variables,
                                                                        non_linear_term_variables=non_linear_term_variables,
                                                                        cluster_col=cluster_col
                                                                     )

    log_like_1 = [cph.log_likelihood_ for cph in cph_imputations_1]

    # smaller model
    df_results_2, _, cph_imputations_2 = fit_cox_models_all_imputations(df_trust_patients, 
                                                                        df_pred_combined,
                                                                         TRUST_phenos,
                                                                         df_imputed_outcomes,
                                                                          df_final,
                                                                          updated_cols_lst,
                                                                          event_col = event_col,
                                                                          time_col = time_col,
                                                                          penalize_model=False,
                                                                          alpha = 0.05,
                                                                          exclude_resistance=False,
                                                                          tb_deaths_only=tb_deaths_only,
                                                                          MIC_type=MIC_type,
                                                                          binarize_MICs=binarize_MICs,
                                                                          include_drugs=[],
                                                                          invert_OR=False,
                                                                        stratify_variables=stratify_variables,
                                                                        non_linear_term_variables=non_linear_term_variables,
                                                                        cluster_col=cluster_col
                                                                         )

    log_like_2 = [cph.log_likelihood_ for cph in cph_imputations_2]

    # LRT_multiple_imputations requires the larger model to be passed in first
    pval = LRT_multiple_imputations(log_like_1, log_like_2, len(df_results_1), len(df_results_2))
    
    return pval



def fit_cox_hazard_ratio_model_with_penalty(df, df_outcome, cols_lst, event_col, time_col, MIC_type='none', binarize_MICs=False, include_drugs=drugs_lst, penalize_model=False, include_interactions=False, RIF_MIC_ordinal_groups=[], stratify_variables=None):

    # Process input features
    df_model_processed, features_lst = process_input_features_for_model(df, cols_lst, MIC_type=MIC_type, binarize_MICs=binarize_MICs, include_drugs=include_drugs, include_interactions=include_interactions, RIF_MIC_ordinal_groups=RIF_MIC_ordinal_groups)
    
    # Add outcome data
    df_model_processed = df_model_processed.merge(df_outcome, on='pid')
    df_model_processed_save = df_model_processed.copy()

    # remove any columns that are the same everywhere to reduce model fitting time
    remove_cols = df_model_processed.columns[df_model_processed.nunique() == 1]
    # print(f"    Removing features {remove_cols} because they are the same everywhere")
    features_lst = list(set(features_lst) - set(remove_cols))

    # keep track of these for un-normalizing the final odds ratios
    means_dict = dict(df_model_processed[features_lst].mean(axis=0))
    std_dict = dict(df_model_processed[features_lst].std(axis=0))
    
    # Normalize features
    scaler = StandardScaler()
    df_model_processed[features_lst] = scaler.fit_transform(df_model_processed[features_lst])

    # if penalize_model:
        
    #     # Cross-validation setup
    #     # kf = KFold(n_splits=5, shuffle=True)
    #     skf = KFold(n_splits=5, shuffle=True)
        
    #     penalizer_results = []
    #     best_penalizer = None
    #     best_score = -np.inf
    
    #     # Iterate over penalizer values
    #     for penalizer in np.logspace(-5, 5, 11):
            
    #         fold_scores = []
            
    #         for train_idx, val_idx in skf.split(df_model_processed, df_model_processed[event_col]):
                
    #         # for train_idx, val_idx in kf.split(df_model_processed):
    #             train_data = df_model_processed.iloc[train_idx]
    #             val_data = df_model_processed.iloc[val_idx]
    
    #             # Fit the model with penalization
    #             cph = lifelines.CoxPHFitter(penalizer=penalizer, l1_ratio=0)
                
    #             cph.fit(train_data[features_lst + [time_col, event_col, 'unique_patient']],
    #                     duration_col=time_col, 
    #                     event_col=event_col, 
    #                     cluster_col='unique_patient'
    #                    )

    #             # log-likelihood on the validation data, whch we want to MAXIMIZE
    #             fold_scores.append(cph.score(val_data))
    
    #         # Compute average score for current penalizer
    #         mean_score = np.mean(fold_scores)
    #         penalizer_results.append((penalizer, mean_score))
            
    #         if mean_score > best_score:
    #             best_score = mean_score
    #             best_penalizer = penalizer

    # else:
    #     # if not using a penalty, set it to 0
    #     best_penalizer = 0

    # Fit the final model on the entire dataset with the best penalizer. Default behavior is l1_ratio = 0, so L2 penalty
    # print(f"Fitting Cox proportional hazards model with L2 penalty of {best_penalizer}")
    # cph = lifelines.CoxPHFitter(penalizer=best_penalizer, l1_ratio=0)
    cph = lifelines.CoxPHFitter()
    
    for col in features_lst + [time_col, event_col, 'unique_patient']:
        num_na = sum(pd.isnull(df_model_processed[col]))

        if num_na > 0:
            print(col, num_na, df_model_processed.loc[pd.isnull(df_model_processed[col])])
    
    cph.fit(df_model_processed[features_lst + [time_col, event_col, 'unique_patient']],
            duration_col=time_col, 
            event_col=event_col, 
            cluster_col='unique_patient',
            # show_progress=True,
            fit_options={'step_size': 0.1},
            strata=stratify_variables
           )
    
    # Get results
    df_model_results = cph.summary

    # have to un-log-transform the variables that were log2-transformed
    log_transform_cols = np.unique(['bl_bmi', 'peth_value_baseline', 'F2', 'RIF_AUC', 'TTP_baseline', 'predicted_label'] + [col for col in model_features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
    
    # Transform coefficients to original scale
    df_model_results['original_mean'] = df_model_results.index.map(means_dict)
    df_model_results['original_std'] = df_model_results.index.map(std_dict)
    df_model_results['coef_transformed'] = df_model_results['coef'] / df_model_results['original_std']
    df_model_results['se_transformed'] = df_model_results['se(coef)'] / df_model_results['original_std']
    
    return df_model_results, df_model_processed, best_penalizer, cph




def combine_survival_probabilities_across_imputations(df_samples_imputations, cph_models_imputations, stratify_covariates):

    df_survival_estimate_imputations = []
    
    for model_idx in range(len(cph_models_imputations)):
    
        df_survival_estimate = cph_models_imputations[model_idx].baseline_survival_.copy()
    
        # col_lst is a tuple of the values for each of the N covariate. len(col_lst) = N
        for col_tup in df_survival_estimate.columns:
    
            stratify_covariates_min_vals = [np.min(df_samples_imputations[model_idx][col]) for col in stratify_covariates]
            stratify_covariates_max_vals = [np.max(df_samples_imputations[model_idx][col]) for col in stratify_covariates]
    
            new_col = ''
            
            for i, val in enumerate(list(col_tup)):
        
                if val == stratify_covariates_min_vals[i]:
                    new_col += f"{stratify_covariates[i]}=0,"
                
                elif val == stratify_covariates_max_vals[i]:
                    new_col += f"{stratify_covariates[i]}=1,"
    
            new_col = new_col.rstrip(',')
        
            df_survival_estimate.rename(columns={col_tup: new_col}, inplace=True)
        
        df_survival_estimate = df_survival_estimate.reset_index().rename(columns={'index': 'Week'})#.melt(id_vars=['Week'])
    
        # Add a row of 1s using loc for week = 1 and survival probability = 1
        df_survival_estimate.loc[len(df_survival_estimate)] = [1] * len(df_survival_estimate.columns)
                
        df_survival_estimate_imputations.append(df_survival_estimate.sort_values("Week").reset_index(drop=True))

    # combine the estimated probabilities for all imputations into a single dataframe
    df_survival_estimate_imputations = pd.concat(df_survival_estimate_imputations)

    # compute the average survival probability and standard deviation across imputations for each week
    df_mean_survival = df_survival_estimate_imputations.groupby(['Week']).mean()
    df_std_survival = df_survival_estimate_imputations.groupby(['Week']).std()

    # melt for plotting
    df_plot_mean = df_mean_survival.reset_index().melt(id_vars='Week').rename(columns={'value': 'Mean'}).sort_values("variable")
    df_plot_std = df_std_survival.reset_index().melt(id_vars='Week').rename(columns={'value': 'SD'}).sort_values("variable")

    assert df_plot_mean.variable.nunique() == np.exp2(len(stratify_covariates))
    
    print(f"{df_plot_mean.variable.nunique()} unique condition combinations")

    df_plot_mean['variable'] = df_plot_mean['variable'].str.replace('cxr_cavity_chest_radiograph_1=0', 'No Cavitation').str.replace('cxr_cavity_chest_radiograph_1=1', 'Cavitation').str.replace('high_lung_involvement=0', '≤ 20% Lung Affected').str.replace('high_lung_involvement=1', '> 20% Lung Affected').str.replace('smear_pos_no_contam_sputum_specimen_1=0', 'Smear Negative').str.replace('smear_pos_no_contam_sputum_specimen_1=1', 'Smear Positive').str.replace(',', '\n')

    df_plot_std['variable'] = df_plot_std['variable'].str.replace('cxr_cavity_chest_radiograph_1=0', 'No Cavitation').str.replace('cxr_cavity_chest_radiograph_1=1', 'Cavitation').str.replace('high_lung_involvement=0', '≤ 20% Lung Affected').str.replace('high_lung_involvement=1', '> 20% Lung Affected').str.replace('smear_pos_no_contam_sputum_specimen_1=0', 'Smear Negative').str.replace('smear_pos_no_contam_sputum_specimen_1=1', 'Smear Positive').str.replace(',', '\n')

    # combine into a single dataframe
    return df_plot_mean.merge(df_plot_std)
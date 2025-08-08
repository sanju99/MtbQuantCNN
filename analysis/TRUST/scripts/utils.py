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




def get_PLI_timika_score_predictions():
    
    PLI_combined = []

    for fName in glob.glob("/n/data1/hms/dbmi/farhat/rs527/trust_project/CXR/pli/regression/*.csv"):

        df = pd.read_csv(fName)
        if 'outlier' in fName:
            df['Outlier'] = 1
        else:
            df['Outlier'] = 0

        PLI_combined.append(df)

    PLI_combined = pd.concat(PLI_combined)
    PLI_combined[['pid', 'View']] = PLI_combined['patient_id'].str.split('_', expand=True)
    del PLI_combined['patient_id']

    PLI_combined = PLI_combined.sort_values(['pid', 'Outlier', 'View'], ascending=[True, True, True]).drop_duplicates('pid', keep='first')
    print(PLI_combined.Outlier.value_counts())

    Timika_combined = []

    for fName in glob.glob("/n/data1/hms/dbmi/farhat/rs527/trust_project/CXR/timika/regression/*.csv"):

        df = pd.read_csv(fName)
        if 'outlier' in fName:
            df['Outlier'] = 1
        else:
            df['Outlier'] = 0

        Timika_combined.append(df)

    Timika_combined = pd.concat(Timika_combined)
    Timika_combined[['pid', 'View']] = Timika_combined['patient_id'].str.split('_', expand=True)
    del Timika_combined['patient_id']

    Timika_combined = Timika_combined.sort_values(['pid', 'Outlier', 'View'], ascending=[True, True, True]).drop_duplicates('pid', keep='first')
    print(Timika_combined.Outlier.value_counts())

    return PLI_combined.rename(columns={'predicted_label': 'predicted_PLI'}).merge(Timika_combined.rename(columns={'predicted_label': 'predicted_Timika_Score'}), on=['pid', 'View', 'Outlier']).set_index(['pid', 'View', 'Outlier'])
          
          
          


def process_patient_metadata_better_encodings(df, df_TTP_smear=None, include_TTP=False):
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
    
    df['smear_grade_baseline'] = df['s_concafb_sputum_specimen_1'].map(smear_encoding_dict)
    
    if include_TTP:
        df = df.merge(df_TTP_smear.query("culture_sample_num <= 2")[['pid', 'TTP']], how='left').rename(columns={'TTP': 'TTP_baseline'})
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
    # INH_resistant_pids = TRUST_phenos.query("INH_lower_bound >= 0.1").pid.unique()
    # df['inh_resistant'] = df['pid'].isin(INH_resistant_pids).astype(int)

    # make sure the patients without a baseline MIC are NA
    # df.loc[~df['pid'].isin(TRUST_phenos.pid.values), 'inh_resistant'] = np.nan

    # df['bl_inh_monoresistant'] = pd.to_numeric(df['bl_inh_monoresistant'], errors='coerce')

    # add this for imputation. Can't pass in CD4 only because it was not measured for HIV- patients
    df.loc[(df['bl_hiv']==0), 'HIV_CD4'] = 0
    df.loc[(df['bl_hiv']==1) & (df['bl_cd4'] >= 200), 'HIV_CD4'] = 1
    df.loc[(df['bl_hiv']==1) & (df['bl_cd4'] < 200), 'HIV_CD4'] = 2

    # convert these columns to integers because R will not read True/False from Python
    # df['smoked_substance_use'] = pd.to_numeric(df['smoked_substance_use'], errors='coerce').astype(int)

    return df




def get_reenrolled_patients(df):
    
    df_reenroll = df.loc[~pd.isnull(df['screen_prevpid'])].sort_values("pid").reset_index(drop=True)
    df_reenroll['screen_prevpid'] = df_reenroll['screen_prevpid'].str.replace(' and ', ',')
    
    df_reenroll = df_reenroll[['pid', 'screen_prevpid']].drop_duplicates()
    
    # for pids that have been enrolled in TRUST more than 2 times, take the latest pid because there will already be another line for the second pid mapping to the first
    # for example: T0210, T0254, and T0387 are all the same person. 
    # The line for T0387 maps to both T0210 and T0254, so split it so that there is (T0387, T0254) and (T0387, T0210)
    
    # so iterate through the multiple previous pids and append new lines to the dataframe
    for i, row in df_reenroll.iterrows():
        if ',' in row['screen_prevpid']:
            for prev_pid in row['screen_prevpid'].split(','):
                df_reenroll = pd.concat([df_reenroll, pd.DataFrame({'pid': [row['pid']], 'screen_prevpid': [prev_pid]}, index=[0])])
    
    df_reenroll = df_reenroll.query("~screen_prevpid.str.contains(',')").sort_values(['pid', 'screen_prevpid']).reset_index(drop=True)

    return df_reenroll



def get_reenrollments_same_lineage(patient_WGS_data_fName, lineage_col='Coll2014'):

    df_trust_patients = pd.read_csv(patient_WGS_data_fName)
    
    df_trust_patients['screen_date'] = pd.to_datetime(df_trust_patients['screen_date'])
    
    # get all pairs of pids that are re-enrollments. This includes the patient who enrolled 3 times split across 3 lines
    df_reenroll = get_reenrolled_patients(df_trust_patients)
    print(f"{df_reenroll.pid.nunique()} pids have been previously enrolled")
    
    df_reenroll_with_dates = df_reenroll.merge(df_trust_patients[['pid', 'screen_date']].rename(columns={'screen_date': 'pid_screen_date'}), how='left').merge(df_trust_patients[['pid', 'screen_date']].rename(columns={'pid': 'screen_prevpid', 'screen_date': 'screen_prevpid_screen_date'}), how='left')
    
    # check that the new screening date is always later than the previous one
    assert len(df_reenroll_with_dates.query("pid_screen_date < screen_prevpid_screen_date")) == 0
    
    # compute the difference in days between the date each patient was screened and rescreened
    df_reenroll_with_dates['screening_diff'] = df_reenroll_with_dates['pid_screen_date'] - df_reenroll_with_dates['screen_prevpid_screen_date']
    
    # convert to float days. The difference above is in default seconds
    df_reenroll_with_dates['screening_diff'] = df_reenroll_with_dates['screening_diff'].dt.total_seconds() / 60 / 60 / 24

    # preferentially keep....okay not sure because of the T0267/T0066 difference. Both times, the person went from having 2.2.1.1./4.3.2.1 mixture at the baseline WGS to 2.2.1.1 only
    df_trust_patients = df_trust_patients.sort_values("SampleID").drop_duplicates('pid', keep='first')

    df_reenroll_with_dates_lineages = df_reenroll_with_dates.merge(df_trust_patients[['pid', 'SampleID', lineage_col, 'F2']], how='left').merge(df_trust_patients[['pid', 'SampleID', lineage_col, 'F2']].rename(columns={'pid': 'screen_prevpid', 'SampleID': 'prevpid_SampleID', lineage_col: f'prevpid_{lineage_col}', 'F2': 'prevpid_F2'}), how='left')

    # nothing should have been dropped here
    # assert len(df_reenroll_with_dates_lineages) == len(df_reenroll)

    # separate patients we can confirm had the same lineage from others. This will necessarily drop patients who had sequencing of 1 or neither sample
    df_reenroll_same_lineage = df_reenroll_with_dates_lineages.dropna().query(f"{lineage_col} == prevpid_{lineage_col}")

    df_reenroll_unknown_or_diff_lineages = df_reenroll_with_dates_lineages.query("pid not in @df_reenroll_same_lineage.pid")

    return df_reenroll_same_lineage.drop_duplicates(), df_reenroll_unknown_or_diff_lineages.drop_duplicates()



def add_unique_patient_ID_to_dataframe(df):

    df_reenroll = get_reenrolled_patients(df)
    
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
    
    print(f"{df_patient_clusters['cluster'].nunique()} patients are duplicated across {df_patient_clusters['pid'].nunique()} pids")
    del df_patient_clusters
    
    # add the unique patient identifiers (just integers)
    df['unique_patient'] = df['pid'].map(unique_cluster_dict)
    
    # increment the unique patient values for the rest of the pids, which have unique patients
    start_cluster_num = np.max(list(unique_cluster_dict.values())) + 1
    
    for i, row in df.iterrows():
    
        if pd.isnull(row['unique_patient']):
            if row['pid'] not in unique_cluster_dict.keys():
                df.loc[i, 'unique_patient'] = start_cluster_num
                unique_cluster_dict[row['pid']] = start_cluster_num
                start_cluster_num += 1
            else:
                df.loc[i, 'unique_patient'] = unique_cluster_dict[row['pid']]
    
    assert sum(pd.isnull(df['unique_patient'])) == 0
    
    return df




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
    culture_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('culture_conversion')) & (~df_trust_patients.columns.str.contains('additional'))])
    single_pid_culture_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + culture_cols].set_index('pid').loc[pid]).reset_index()
    
    single_pid_culture_results.columns = ['column', 'result']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_culture_results['sample_num'] = single_pid_culture_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_culture_results['column'] # original column name, don't need anymore
    single_pid_culture_results = single_pid_culture_results.sort_values('sample_num').reset_index(drop=True)

    ########################################## STEP 2: TIME TO CULTURE POSITIVITY ########################################## 
    
    # get all TTP culture results for a single pid
    TTP_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('ttp')) & (~df_trust_patients.columns.str.contains('|'.join(['analysis', 'hour', 'day', 'additional'])))])
    single_pid_TTP_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + TTP_cols].set_index('pid').loc[pid]).reset_index()
    
    # BMC Group combined TTP in days with TTP hours (so days + 24 * hours) to get this column
    single_pid_TTP_results.columns = ['column', 'hours']
    
    # get the sample week and sort by that. Can't sort by the raw column name itself because _2 will be considered greater than _10. So need to convert them to integers
    single_pid_TTP_results['sample_num'] = single_pid_TTP_results['column'].str.split('_').str[-1].astype(int)
    del single_pid_TTP_results['column'] # original column name, don't need anymore
    single_pid_TTP_results = single_pid_TTP_results.sort_values('sample_num').reset_index(drop=True)
    
    ########################################## STEP 3: SMEAR GRADE ##########################################
            
    # get all sputum culture results for a single pid 
    smear_grade_cols = list(df_trust_patients.columns[(df_trust_patients.columns.str.contains('s_concafb_sputum_specimen')) & (~df_trust_patients.columns.str.contains('additional'))])
    single_pid_smear_results = pd.DataFrame(df_trust_patients.drop_duplicates(subset='pid')[['pid'] + smear_grade_cols].set_index('pid').loc[pid]).reset_index()
    
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

    # check that there are no duplicates. This would occur if there are other smear grade / TTP / culture columns like for additional visits. These should be excluded
    assert len(combined_sputum_results) == combined_sputum_results.sample_num.nunique()
    
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




def compute_difference_between_dates(df, col_1, col_2, new_col_name):

    df[col_1] = pd.to_datetime(df[col_1])
    df[col_2] = pd.to_datetime(df[col_2])
    
    df[new_col_name] = df[col_1] - df[col_2]

    # convert to float days
    df[new_col_name] = df[new_col_name].dt.total_seconds() / 60 / 60 / 24

    return df
    
        
def process_dates_table(df_full, dates_fName):

    df_dates = pd.read_csv(dates_fName)
    
    # skip first column, which is pid, and convert the rest to datetime format
    for col in df_dates.columns[1:]:
        df_dates[col] = pd.to_datetime(df_dates[col], format='%m/%d/%Y')
    
        if " " in col:
            df_dates.rename(columns={col: col.replace(' ', '_')}, inplace=True)

    # combine with the treatment outcomes for easy look up
    df_dates = df_dates.merge(df_full[['pid', 'to_studyto_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_reasonoth_end_of_study_parti', 'esp_cod_end_of_study_parti']], on='pid')
        
    # treatment duration
    df_dates = compute_difference_between_dates(df_dates, 'Treatment_Completion_Date', 'Date_of_Treatment_Initiation', 'tx_duration')
    
    # time between screening date and tx completion date. It's usually the same as tx_duration but sometimes screening date and tx initiation date are not the same
    df_dates = compute_difference_between_dates(df_dates, 'Treatment_Completion_Date', 'Screen_Date', 'tx_completion_time')
    
    # to_comments_treatment_outcome = 'Participant died on 25/09/2021, last clinic appointment 16/04/2021'
    df_dates.loc[df_dates['pid']=='T0252', 'Date_of_Death'] = datetime(2021, 9, 25)
    
    # to_comments_treatment_outcome = 'Cause of death unknown. Participant died 12/10/2021'
    df_dates.loc[df_dates['pid']=='T0274', 'Date_of_Death'] = datetime(2021, 10, 12)

    # pulled from RedCap notes by Sue and the field team
    df_dates.loc[df_dates['pid']=='T0248', 'Date_of_Death'] = datetime(2021, 5, 22)
    df_dates.loc[df_dates['pid']=='T0240', 'Date_of_Death'] = datetime(2022, 8, 28)
    df_dates.loc[df_dates['pid']=='T0137', 'Date_of_Death'] = datetime(2019, 2, 14)
    df_dates.loc[df_dates['pid']=='T0318', 'Date_of_Death'] = datetime(2022, 9, 11)

    # this one is discrepant. Currently they said to use 17-06-2022, but something is still off because why would pid T0274 have a 6-month outcome of death in that case
    df_dates.loc[df_dates['pid']=='T0274', 'Date_of_Death'] = datetime(2022, 6, 17)
    
    # date of death relative to screening date
    df_dates = compute_difference_between_dates(df_dates, 'Date_of_Death', 'Screen_Date', 'dod')

    # because some dates were entered manually, check the accuracy. Can't be before the initial screening date
    assert len(df_dates.query("dod < 0")) == 0

    return df_dates





def get_dictionary_mapping_months_to_cols(cols_lst):

    cols_dict = {}
    
    for name in cols_lst:
        month = int(name.split('_')[-1].replace('mo', ''))
        cols_dict[month] = name
    
    # sort by month, descending. Need to iterate through them descending to get the latest completed follow-up
    cols_dict = {key: cols_dict[key] for key in sorted(cols_dict.keys())[::-1]}

    return cols_dict




def get_last_date_of_fu(df, pid, cols_dict):

    found_last_follow_up = False

    # check the post-tx appointments first, then the during-tx appointments. 
    # There are some cases where the patients have a 6-month treatment outcome, but they haven't come for post-tx follow-up appointments yet
    for col_name in cols_dict.values():
            
        if not found_last_follow_up:
            
            latest_date = df.query("pid==@pid")[col_name].values[0]
            
            if not pd.isnull(latest_date):
                found_last_follow_up = True    

    if found_last_follow_up:
        return pd.to_datetime(latest_date)
    else:
        return None
    
    
    
def get_dates_of_deaths(df):
    '''
    All dates are relative to the screening date in units of days.
    '''
    
    during_tx_fu_cols = df.columns[(df.columns.str.startswith('fu_date_treatment')) & (df.columns.str.endswith('mo'))]
    post_tx_fu_cols = df.columns[(df.columns.str.startswith('tlfb_date')) & (df.columns.str.contains('posttreatment'))]

    during_tx_fu_cols_dict = get_dictionary_mapping_months_to_cols(during_tx_fu_cols)
    post_tx_fu_cols_dict = get_dictionary_mapping_months_to_cols(post_tx_fu_cols)

    # known_dod_dict = dict(zip(df[['pid', 'esp_dod_end_of_study_parti']].dropna()['pid'], df[['pid', 'esp_dod_end_of_study_parti']].dropna()['esp_dod_end_of_study_parti']))

    # df_deaths from Sue has 3 more patients than df_full. Also checked that all the death dates match between the two dataframes for the other 11
    known_dod_dict = dict(zip(df.dropna(subset='Date_of_Death')['pid'], df.dropna(subset='Date_of_Death')['Date_of_Death']))  

    # if the waiver specifies death, then they must also have a death date in the table. Otherwise, it's likely that their death occurred after leaving the trial
    died_pids = df.query("to_studyto_treatment_outcome==6 | esp_reason_end_of_study_parti == 8 | (waiver_reason_end_of_study_parti == 4 & pid in @known_dod_dict.keys())").pid.unique()

    # add a column annotating TB-attributable deaths
    tb_death_pids = []
    non_tb_death_pids = []

    # check these descriptor columns for TB-related strings
    for col in ['esp_cod_end_of_study_parti', 'waiver_deathcause_end_of_study_parti']:
        
        tb_death_pids += list(df.dropna(subset=col).query(f"{col}.str.contains('|'.join(['pulmonary', 'ashtma', 'asthma', 'tuberc', 'embolism']), case=False)").pid.values)

        # manually add from additional data that Sue sent on December 3, 2024. T0318 died of MDR, so check that they aren't in the study
        tb_death_pids += ['T0240', 'T0318']

        non_tb_death_pids += list(df.dropna(subset=col).query(f"{col}.str.contains('|'.join(['cancer', 'heart', 'attack', 'accident', 'covid']), case=False)").pid.values)        
    
    df_death_events = pd.DataFrame()
    
    # the two columns esp_dod_end_of_study_parti and waiver_deathdate_end_of_study_parti are the same for all patients
    # so arbitrarily use esp_dod_end_of_study_parti as the date
    # there are 18/32 patients without a date of death currently. The only patients with a date of death are those with esp_reason_end_of_study_parti = 8
    # found 2 in the to_comments column
    
    df_death_events['pid'] = died_pids
    df_death_events['date'] = df_death_events['pid'].map(known_dod_dict)

    # add outcomes to use them for checking
    df_death_events = df_death_events.merge(df[['pid', 'to_studyto_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_date_end_of_study_parti', 'waiver_reason_end_of_study_parti', 'esp_cod_end_of_study_parti', 'waiver_deathcause_end_of_study_parti']], on='pid', how='left')

    # for the ones without dates according to the dataframe above, take the last appointment date
    # every patient with ESP = death has a value in df_deaths from Sue, which matches esp_dod_end_of_study_parti BUT NOT esp_date_end_of_study_parti
    
    # for pid in df_death_events.loc[pd.isnull(df_death_events['date'])].index.unique():
    for i, row in df_death_events.iterrows():

        if pd.isnull(row['date']):
            
            pid = row['pid']
            
            # the missing ones must have to_studyto_treatment_outcome = 6 because we already required a date to be present if waiver_reason_end_of_study_parti = 4
            # and already checked that all patients with esp_reason_end_of_study_parti = 8 have a death date
            assert row['to_studyto_treatment_outcome'] == 6
    
            # so get the last appointment during the 6 months of treatment
            latest_date = get_last_date_of_fu(df, pid, during_tx_fu_cols_dict)
    
            if latest_date is not None:
                df_death_events.loc[i, 'date'] = latest_date

            # if no follow-up appointment date found, then take the ESP date
            # little bit weird not to censor these patients, but they are known to have a 6 month outcome of death
            else:
                latest_date = pd.to_datetime(row['esp_date_end_of_study_parti'])

                if pd.isnull(latest_date):
                    raise ValueError(f"No last date of contact or follow-up found for {pid}")
                else:
                    df_death_events.loc[i, 'date'] = latest_date

    df_death_events['event_type'] = 'death'

    # df_death_events.loc[df_death_events['pid'].isin(tb_death_pids), 'TB_death'] = 1
    # df_death_events['TB_death'] = df_death_events['TB_death'].fillna(0).astype(int)
    
    df_death_events.loc[df_death_events['pid'].isin(non_tb_death_pids), 'TB_death'] = 0
    df_death_events['TB_death'] = df_death_events['TB_death'].fillna(1).astype(int)
    
    return df_death_events.set_index('pid')





def get_dates_of_relapses(df, df_reenroll_same_lineage):
    '''
    Relapse is not a 6 month outcome because it was assessed later.

    For patients whose ESP is relapse, 

    For patients who don't have ESP = relapse, take the date when they were screened for re-enrollment as the relapse date.
    '''
    
    # relapse is only in the esp_reason_end_of_study_parti column. HOWEVER, pids in screen_prevpid also relapsed because they reenrolled.
    # also note that there is a patient (T0328) with the word relapse in waiver_deathcause_end_of_study_parti, but they are not listed as a relapse nor were they reenrolled
    # HOWEVER, they are listed as death in the waiver column, and the description is this. Regardless, they have a death date, so use that, and it's TB-attributable
    relapse_pids = np.unique(np.concatenate([df.query("esp_reason_end_of_study_parti == 5").pid.values,
                                             df_reenroll_same_lineage.screen_prevpid.values
                                            ]))

    df_relapse_events = pd.DataFrame(columns = ['pid'])

    screening_diff_dict = dict(zip(df_reenroll_same_lineage['screen_prevpid'], df_reenroll_same_lineage['pid_screen_date']))

    df_relapse_events['pid'] = relapse_pids

    # some people who re-enrolled also have esp_reason_end_of_study_parti = 5, so the following code will replace their dates
    df_relapse_events['date'] = df_relapse_events['pid'].map(screening_diff_dict)
    
    # all of the people who relapsed completed treatment or were cured in the 6 months (to_studyto_treatment_outcome in [1, 2]), so no need to worry about them i.e. failing tx first
    # so for the remaining patients who were not re-enrolled, take the relapse date to be the last post-tx follow-up appointment they came to 
    df_relapse_events = df_relapse_events.set_index('pid')

    # only get the date of most recent follow-up for patients whose ESP is 5. If they don't have ESP = 5, then take the date of re-enrollment (done above already)
    for pid in df.query("esp_reason_end_of_study_parti == 5").pid.values:
    
        # a single patient -- T0395 -- came for an additional visit. So take that date for the ones that relapse
        addl_visit_date = df.query("pid==@pid")['screen_date_add_additional_visit'].values[0]

        # because they have relapse as their ESP, first check the esp_date_end_of_study_parti column
        if not pd.isnull(addl_visit_date):
            df_relapse_events.loc[pid, 'date'] = addl_visit_date
        else:
            latest_date = pd.to_datetime(df.query("pid==@pid")['esp_date_end_of_study_parti'].values[0])

            if not pd.isnull(latest_date):
                df_relapse_events.loc[pid, 'date'] = latest_date
            # if not found, then take the most recent follow-up date
            # check during the post treatment follow-up stage only because we know they completed treatment
            else:
                latest_date = get_last_date_of_fu(df, pid, post_tx_fu_cols_dict)

                if latest_date is not None:
                    df_relapse_events.loc[pid, 'date'] = latest_date
                else:
                    raise ValueError(f"No follow-up date found for {pid}")

    df_relapse_events['event_type'] = 'relapse'
    df_relapse_events = df_relapse_events.reset_index().merge(df[['pid', 'to_studyto_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_date_end_of_study_parti', 'waiver_reason_end_of_study_parti']], how='left')

    return df_relapse_events.set_index('pid')





def get_treatment_failure_dates(df):
    '''
    This includes patients who had treatment failure, extension, and positive cultures at 5 months
    '''
    
    during_tx_fu_cols = df.columns[(df.columns.str.startswith('fu_date_treatment')) & (df.columns.str.endswith('mo'))]
    post_tx_fu_cols = df.columns[(df.columns.str.startswith('tlfb_date')) & (df.columns.str.contains('posttreatment'))]

    during_tx_fu_cols_dict = get_dictionary_mapping_months_to_cols(during_tx_fu_cols)
    post_tx_fu_cols_dict = get_dictionary_mapping_months_to_cols(post_tx_fu_cols)

    # treatment failure or extension at 6 months. Their primary endpoint is the failure, so take the number of days between screen date and tx completion
    df_failure_events = df.query("to_studyto_treatment_outcome in [4, 5]")[['pid']].set_index('pid')

    # take the last completed appointment during the 6-month treatment window. Most of these didn't come to follow-up appointments
    for pid in df_failure_events.index.values:
        
        # check the during-tx appointments only because these patients didn't complete or had to extend treatment, so restrict to the 6 months
        latest_date = get_last_date_of_fu(df, pid, during_tx_fu_cols_dict)
        
        if latest_date is not None:
            df_failure_events.loc[pid, 'date'] = latest_date
        else:
            raise ValueError(f"No follow-up date found for {pid}")

    # people who had positive month 5 cultures should all meet the criteria above, but check just to make sure
    month5_positive_pids = df.query("culture_conversion_sputum_specimen_13=='tb_positive'").pid.unique()

    if len(set(month5_positive_pids) - set(df_failure_events.index.values)):
        raise ValueError(f"There are patients with positive month 5 cultures who aren't listed as tx failures or extensions")

    df_failure_events['event_type'] = 'tx_failure'

    # waiver_reason_end_of_study_parti
    df_failure_events = df_failure_events.reset_index().merge(df[['pid', 'to_studyto_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_reasonoth_end_of_study_parti', 'esp_date_end_of_study_parti']], how='left')

    return df_failure_events.set_index('pid')




def get_completion_dates(df, df_events):
    '''
    This function covers patients who were actually cured / completed treatment and patients who were lost to follow-up, withdrew, or moved. 

    Patients who completed the study (ESP = 1) should have the ESP date as their censored date.

    Patients who were lost to follow-up, withdrew, or moved should take the last appointment date as the censored date.
    '''
    
    during_tx_fu_cols = df.columns[(df.columns.str.startswith('fu_date_treatment')) & (df.columns.str.endswith('mo'))]
    post_tx_fu_cols = df.columns[(df.columns.str.startswith('tlfb_date')) & (df.columns.str.contains('posttreatment'))]

    during_tx_fu_cols_dict = get_dictionary_mapping_months_to_cols(during_tx_fu_cols)
    post_tx_fu_cols_dict = get_dictionary_mapping_months_to_cols(post_tx_fu_cols)
    
    # include patients who haven't completed treatment yet (to_studyto_treatment_outcome = NA). So just exclude patients who had a death, relapse, or treatment failure
    df_completed = df.query("pid not in @df_events.pid")

    # for these patients, the last date that we know that they are still TB-free is esp_date_end_of_study_parti
    df_completed = df_completed[['pid', 'to_studyto_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_date_end_of_study_parti']].rename(columns={'esp_date_end_of_study_parti': 'date'})
    
    # fill in dates for patients who haven't completed follow-up or treatment
    for i, row in df_completed.iterrows():
    
        pid = row['pid']

        # don't use the ESP date for the moved (6), lost to fu (7), or other (9) cases because often the ESP date is very late if the patient was lost to follow-up
        # they weren't contacted up to the ESP date, that's just the date that was recorded as the last date for them. More accurate to take an appointment
        if pd.isnull(row['date']) or row['esp_reason_end_of_study_parti'] in [6, 7, 9]:

            if pd.isnull(row['date']):
                # if there is no date, double check that there is no ESP event            
                assert pd.isnull(row['esp_reason_end_of_study_parti'])
            
            # check the post-tx appointments first, then the during-tx appointments. 
            # There are some cases where the patients have a 6-month treatment outcome, but they haven't come for post-tx follow-up appointments yet
            latest_date = get_last_date_of_fu(df, pid, post_tx_fu_cols_dict)
    
            # treatment completed, but follow-up not
            if latest_date is not None:
                df_completed.loc[i, 'date'] = latest_date
            else:
                latest_date = get_last_date_of_fu(df, pid, during_tx_fu_cols_dict)

                # if still nothing, then only take the ESP date
                if latest_date is not None:
                    df_completed.loc[i, 'date'] = latest_date
                else:
                    latest_date = row['date']

                    if latest_date is not None:
                        df_completed.loc[i, 'date'] = latest_date
                    else:
                        raise ValueError(f"Neither treatment date nor follow-up date found for {pid}")  
    
    assert len(df_completed.loc[pd.isnull(df_completed['date'])]) == 0
    df_completed['event'] = 0
    
    return df_completed[['pid', 'date', 'event']].merge(df[['pid', 'to_studyto_treatment_outcome', 'esp_reason_end_of_study_parti', 'esp_date_end_of_study_parti']], how='left')
import numpy as np
import pandas as pd
import glob, os, argparse, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from utils import *


############################## THIS SCRIPT PREPARES THE TRUST DATA FOR ANALYSES ON TREATMENT OUTCOMES AND TCC ##############################
############################## IT KEEPS ONLY PIDS WITH VALID TCC AND WGS DATA ##############################


parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("-i", dest='patient_data_fName', type=str, required=True, help='Full path to a filename for the RedCap data and WGS data already merged. Created by combined_patient_WGS_data.py')

cmd_line_args = parser.parse_args()

patient_data_fName = cmd_line_args.patient_data_fName

# save new files here
out_dir = os.path.dirname(patient_data_fName)


############################## STEP 0. COMBINE PATIENT AND WGS DATA INTO A SINGLE DATAFRAME ##############################


df_trust_patients = pd.read_csv(patient_data_fName, low_memory=False)
print(f"{df_trust_patients.pid.nunique()} pids have high quality WGS data\n")

# the lineage variable always gets weird because it's strings, but some are NA
for i, row in df_trust_patients.iterrows():
    if not pd.isnull(row['Lineage']):
        if type(row['Lineage']) == str:
            if ',' not in row['Lineage']:
                df_trust_patients.loc[i, 'Lineage'] = str(int(float(row['Lineage'])))
        else:
            df_trust_patients.loc[i, 'Lineage'] = str(int(row['Lineage']))
                
# useful for grouping later
df_trust_patients.loc[df_trust_patients['bl_hiv']==0, 'HIV_CD4'] = 0
df_trust_patients.loc[(df_trust_patients['bl_hiv']==1) & (df_trust_patients['bl_cd4'] >= 200), 'HIV_CD4'] = 1
df_trust_patients.loc[(df_trust_patients['bl_hiv']==1) & (df_trust_patients['bl_cd4'] < 200), 'HIV_CD4'] = 2



############################## STEP 1. MAKE SOME REPLACEMENTS IN THE MIC DATA REGARDING TESTING METHODS. THESE WERE SENT IN EMAILS FROM BRENDON. MOST WERE INCORPORATED INTO REDCAP, BUT KEEP THEM HERE TO CHECK ##############################


# from Brendon email September 5, 2024
df_trust_patients.loc[(pd.isnull(df_trust_patients['s_inhmicmeth_sputum_specimen_1'])) & (~pd.isnull(df_trust_patients['s_inhmic_sputum_specimen_1'])), 's_inhmicmeth_sputum_specimen_1'] = 3
df_trust_patients.loc[df_trust_patients['pid']=='T0097', 's_inhmicmeth_sputum_specimen_2'] = 2
df_trust_patients.loc[df_trust_patients['pid']=='T0063', 's_inhmicmeth_sputum_specimen_2'] = 3

# additional email from Brendon November 27, 2024:
df_trust_patients.loc[df_trust_patients['pid']=='T0466', 's_embmicmeth_sputum_specimen_1'] = 2
df_trust_patients.loc[df_trust_patients['pid']=='T0470', 's_rifmicmeth_sputum_specimen_1'] = 2

# from Noorjahn's email on December 20, 2024:
# INH MIC for sample S0108-01 was retested in duplicate in MGIT, giving an MIC of > 0.1, so update s_inhmic_sputum_specimen_1 to 5
df_trust_patients.loc[df_trust_patients['pid']=='T0108', ['s_inhmicmeth_sputum_specimen_1', 's_inhmic_sputum_specimen_1']] = [2, 5]

# INH MIC for sample S0108-08 was retested in triplicate in MGIT, giving MIC = 0.1-0.25, so update s_inhmic_sputum_specimen_1 to 4
df_trust_patients.loc[df_trust_patients['pid']=='T0108', ['s_inhmicmeth_sputum_specimen_8', 's_inhmic_sputum_specimen_8']] = [2, 4]

# INH MIC for sample S0030-01 was retested in MGIT, giving an MIC of 0.1. There was intermediate growth, so could be heteroresistance?
# update s_inhmicmeth_sputum_specimen_1 to 2 (MGIT) and s_inhmic_sputum_specimen_1 to 3 (0.05-0.1)
df_trust_patients.loc[df_trust_patients['pid']=='T0030', ['s_inhmicmeth_sputum_specimen_1', 's_inhmic_sputum_specimen_1']] = [2, 3]


############################## STEP 2. GET SPUTUM AND CULTURE RESULTS. IT'S USEFUL TO HAVE THEM SAVED BECAUSE THE DATA IN THE ORIGINAL PATIENT DATAFRAME IS NOT EASILY INTERPRETABLE ##############################


df_TTP_smear, df_TCC, df_combined_culture_results = get_combined_culture_results(df_trust_patients)
print(f"{df_TCC.pid.nunique()} have a valid time to culture conversion\n")
print(f"{df_TTP_smear.dropna(subset='culture_sample_num').query('culture_sample_num <= 2').pid.nunique()} pids with baseline TTPs within the first 2 weeks\n")

# save all 3 dataframes
df_TTP_smear.to_csv(os.path.join(out_dir, "TTP_smear_results.csv"), index=False)
df_TCC.to_csv(os.path.join(out_dir, "TCC.csv"), index=False)
df_combined_culture_results.to_csv(os.path.join(out_dir, "cultures.csv"), index=False)

# add TTP at baseline and change some of the other encodings to be more intuitive
df_trust_patients = process_patient_metadata_better_encodings(df_trust_patients, df_TTP_smear=df_TTP_smear, include_TTP=True)


############################## STEP 3. MERGE IN PK DRUG AUC AND CHEST X-RAY PREDICTIONS ##############################
    
    
df_drug_AUC = pd.read_csv(f"{out_dir}/PK_AUC_predictions.csv")

df_trust_patients = df_trust_patients.merge(df_drug_AUC, on='pid', how='left')

# combine with smear grade at baseline and PLI predictions
PLI_fName = f"{out_dir}/PLI_Timika_predictions.csv"

if not os.path.isfile(PLI_fName):
    pli_predictions = get_PLI_timika_score_predictions()
    pli_predictions.to_csv(PLI_fName)
    
pli_predictions = pd.read_csv(PLI_fName)

df_trust_patients = df_trust_patients.merge(pli_predictions, on='pid', how='left')


##################### STEP 4: REMOVE PIDS OF PATIENTS WHO PREVIOUSLY ENROLLED IN TRUST #####################


# this function adds the unique_patient column to the argument dataframe
df_trust_patients = add_unique_patient_ID_to_dataframe(df_trust_patients)

print(f"{df_trust_patients['unique_patient'].nunique()} unique patients across {df_trust_patients['pid'].nunique()} pids\n")

df_trust_patients['unique_patient'] = df_trust_patients['unique_patient'].astype(int)

# keep the first instance of each re-enrolled patient
df_trust_patients = df_trust_patients.sort_values(['unique_patient', 'pid']).drop_duplicates(subset='unique_patient', keep='first')


############################## STEP 5: REMOVE PATIENTS WHO CHANGED TREATMENT REGIMEN ##############################


# T0122 was diagnosed with MDR. 
# T0137: ESP reason says 'Changed to "liver-friendly regimen". No longer on first line drugs'
exclude_pids_resistance = ['T0122', 'T0137']

# T0322 says they were started on RHZE + levofloxacin
# T0330 is fine, says they had a positive month 5 culture and were RIF- and INH-sensitive
# T0355 started MDR regimen upon incarceration
exclude_pids_diff_regimen = ['T0322', 'T0355']

df_trust_patients = df_trust_patients.query("pid not in @exclude_pids_resistance & pid not in @exclude_pids_diff_regimen")


############################## STEP 6. GET COMPOSITE PATIENT OUTCOMES. THIS TAKES A WHILE TO COMPUTE, SO SAVE FOR LATER USE ##############################


# to_studyto_treatment_outcome column. 
# Other descriptions in to_studyto_other_treatment_outcome
# additional comments in to_comments_treatment_outcome
treatment_outcome_dict = {1: 'cure',
                          2: 'complete',
                          3: 'defaulted', # typically they stopped taking treatment entirely, not just interruption. May have been doctor-recommended, like due to liver injury or something
                          4: 'failure',
                          5: 'extension',
                          6: 'death',
                          7: 'transferred',
                          8: 'lost_fu',
                          9: 'other'
                        }

# esp_reason_end_of_study_parti column
# Other descriptions in esp_reasonoth_end_of_study_parti
esp_dict = {1: 'complete', # completed all study requirements
            2: 'inappropriate', # should not have been included in the study
            3: 'withdrew',
            4: 'failure', # includes both groups 4 and 5 in treatment_outcome_dict, meaning the patient was culture positive at month 5 or tx was extended
            5: 'relapse',
            6: 'moved', # moved/transferred out,
            7: 'lost_fu',
            8: 'death',
            9: 'other'
            }



# fix death dates before computing death dates

# convert to datetime
df_trust_patients['screen_date'] = pd.to_datetime(df_trust_patients['screen_date'])
df_trust_patients['Date_of_Death'] = pd.to_datetime(df_trust_patients['waiver_deathdate_end_of_study_parti'])

# pulled from RedCap notes by Sue and the field team
df_trust_patients.loc[df_trust_patients['pid']=='T0137', 'Date_of_Death'] = datetime(2019, 2, 14)
df_trust_patients.loc[df_trust_patients['pid']=='T0318', 'Date_of_Death'] = datetime(2022, 9, 11)

# save df_trust_patients
print(f"Keeping {df_trust_patients.pid.nunique()} pids with {df_trust_patients.SampleID.nunique()} WGS samples for downstream analyses\n")
df_trust_patients.to_csv(os.path.join(out_dir, "cleaned_patient_data_for_analysis.csv"), index=False)

# compute event times
df_death_events = get_dates_of_deaths(df_trust_patients)
df_failure_events = get_treatment_failure_dates(df_trust_patients)

# for patients who re-enrolled, consider them as relapses if they had infections of the same lineage both times
df_reenroll_same_lineage, _ = get_reenrollments_same_lineage(patient_data_fName, lineage_col='Coll2014')
df_relapse_events = get_dates_of_relapses(df_trust_patients, df_reenroll_same_lineage)

# combine the types of events
df_combined_events = pd.concat([df_failure_events, df_relapse_events, df_death_events]).reset_index()
df_combined_events['event'] = 1

# add screen dates and make sure they are all datetime objects
df_combined_events = df_combined_events.merge(df_trust_patients[['pid', 'screen_date']])
df_combined_events['event_date'] = pd.to_datetime(df_combined_events['date'])
df_combined_events['screen_date'] = pd.to_datetime(df_combined_events['screen_date'])

# create "date" column, which is the number of days to the event
df_combined_events = compute_difference_between_dates(df_combined_events, 'event_date', 'screen_date', 'date')

# deduplicate events, keeping the earliest instance
df_combined_events_dedup = df_combined_events.sort_values(['pid', 'date'], na_position='last').drop_duplicates(subset='pid', keep='first')
print(f"{df_combined_events_dedup.pid.nunique()} pids had an event")
del df_combined_events

df_completed = get_completion_dates(df_trust_patients, df_combined_events_dedup)
df_completed['event_type'] = 'cure'

# add screen dates and make sure they are all datetime objects
df_completed = df_completed.merge(df_trust_patients[['pid', 'screen_date']])
df_completed['censor_date'] = pd.to_datetime(df_completed['date'])

# create "date" column, which is the number of days to the event
df_completed = compute_difference_between_dates(df_completed, 'censor_date', 'screen_date', 'date')

print(f"{df_completed.pid.nunique()} pids did not have an event")

# check no overlap between the groups
assert len(set(df_combined_events_dedup.pid).intersection(df_completed.pid)) == 0

df_final = pd.concat([df_combined_events_dedup, df_completed])

# do a double check that anyone who has death in either of these columns has an event?
assert len(df_final.query("(to_studyto_treatment_outcome==6 | esp_reason_end_of_study_parti == 8) & event != 1")) == 0

# check that no negative dates
assert len(df_final.query("date < 0")) == 0

# don't consider events after the ESP date because there are some relapses with very long intervals
# can only consider what happened during the trial. If that's too short, then the trial follow-up period should have been longer
# censor these patients at their ESP date
df_final = compute_difference_between_dates(df_final, 'esp_date_end_of_study_parti', 'screen_date', 'time_in_trial')
df_final.loc[(df_final['event_date'] > df_final['esp_date_end_of_study_parti']) & (df_final['event']==1), 'event'] = 0
df_final.loc[(df_final['event_date'] > df_final['esp_date_end_of_study_parti']), 'date'] = df_final.loc[(df_final['event_date'] > df_final['esp_date_end_of_study_parti'])]['time_in_trial']

# # keep events of patients if the event occurred within 2 years of the screening date (2 years = 365 * 2 = 730 days).
# censor_day = 730
# df_final.loc[(df_final['event'] == 1) & (df_final['event_type'] != 'cure') & (df_final['date'] > censor_day), 'event'] = 0
# df_final.loc[(df_final['event_type'] != 'cure') & (df_final['date'] > censor_day), 'date'] = df_final.loc[(df_final['event_type'] != 'cure') & (df_final['date'] > censor_day)]['time_in_trial']

df_final['weeks'] = df_final['date'] / 7
df_final['months'] = df_final['date'] / 30

df_final.drop_duplicates().to_csv(os.path.join(out_dir, "tx_outcomes.csv"), index=False)
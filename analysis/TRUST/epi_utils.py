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

log_metadata_cols = ['F2', 'TTP_baseline', 'peth_value_baseline']

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

cc_df = pd.read_csv("./data_processing/data_utils/drug_CC.csv")

# MIC encoding from the TRUST codebook
MIC_encoding_dicts = {'RIF': {1: '0,0.03', 2: '0.03,0.06', 3: '0.06,0.125', 4: '0.125,0.25', 5: '0.25,0.5', 6: '0.5,1', 7: '1,inf'},
                      'INH': {1: '0,0.025', 2: '0.025,0.05', 3: '0.05,0.1', 4: '0.1,0.2', 5: '0.2,inf'},
                      'EMB': {1: '0,0.6', 2: '0.6,1.25', 3: '1.25,2.5', 4: '2.5,5', 5: '5,inf'},
                      'PZA': {1: '0,25', 2: '25,50', 3: '50,75', 4: '75,100', 5: '100,inf'}
                     }




def boundedLoss_predict(pred_df, y_pred_col="y_pred", lower_bounds_col="lower", upper_bounds_col="upper"):
    '''
    y_true and y_pred are log-MICs. lower_bounds and upper_bounds are exponentiated. 
    
    This function returns bounded MAE, MSE, and the proportion of points measured within 1 MIC doubling (1 log2 unit)
    ''' 
    
    del_cols = [f"{y_pred_col}_exp", "within_doubling", "within_1bin", "compute_error", f"{lower_bounds_col}_rounded", f"{upper_bounds_col}_rounded"]

    for col in del_cols:
        if col in pred_df.columns:
            del pred_df[col]

    # first add essential agreement (proportion within 1 doubling dilution)
    # not always helpful because some "doubling" dilutions are not exact, i.e. 0.3, 0.6, 0.125, 0.5. But the number is here if needed
    pred_df[f"{y_pred_col}_exp"] = np.round(np.exp2(pred_df[y_pred_col]).astype(float), 2)
    
    pred_df.loc[(pred_df[lower_bounds_col] / 2 <= pred_df[f"{y_pred_col}_exp"]) & 
                (pred_df[upper_bounds_col] * 2 >= pred_df[f"{y_pred_col}_exp"])
                , "within_doubling"] = 1

    pred_df.loc[(pred_df[f"{y_pred_col}_exp"] == 0.06) & (pred_df[lower_bounds_col] == 0.12), "within_doubling"] = 1
    pred_df["within_doubling"] = pred_df["within_doubling"].fillna(0).astype(int)
        
    # make copies to avoid changing the original dataframe
    lower_bounds = np.copy(pred_df[lower_bounds_col].values) #pred_df[lower_bounds_col].values / 2
    upper_bounds = np.copy(pred_df[upper_bounds_col].values) #pred_df[upper_bounds_col].values * 2
    
    lower_bounds[lower_bounds==0] += 1e-10
    lower_bounds = np.log2(lower_bounds)
    upper_bounds = np.log2(upper_bounds)

    # use less than or equal to because the true MIC is in the range (lower, upper], so it is not equal to lower.
    pred_df["compute_error"] = ((pred_df[y_pred_col].values <= lower_bounds) | (pred_df[y_pred_col].values > upper_bounds)).astype(int)

    # compute the error relative to the bounds, NOT RELATIVE TO THE MIDPOINT (y_test) of each isolate
    # np.clip returns one of the values from lower_bounds or upper_bounds, whichever is closest to the prediction, if the value is outside the bounds
    # if the test values are within the bounds, the values themselves are returned
    bound_to_compute_error = np.clip(pred_df[y_pred_col].values, lower_bounds, upper_bounds)
    mae = np.mean((np.abs(bound_to_compute_error - pred_df[y_pred_col])))
    mse = np.mean((np.square(bound_to_compute_error - pred_df[y_pred_col])))

    return mae, mse, pred_df["within_doubling"].mean()



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
    df_pool['pval'] = 2 * st.t.sf(abs(df_pool['wald_stat']), df_pool['dof_adj'])

    # compute confidence intervals. CI = coef_pooled ± t_stat * se_pooled
    df_pool['t_critical'] = np.abs(st.t.ppf(1 - alpha / 2, df_pool['dof_adj']))
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




def forest_plot(df, covariates_order, labels_dict, val_col='OR', alpha=0.05, log=False, pval_offset=1.05, x_lim=None, df_stratify_variables_results=None, saveName=None):
    
    # add the relative risks dataframe to plot those as well
    if df_stratify_variables_results is not None:
        df = pd.concat([df, df_stratify_variables_results])
    
    # Filter out the intercept and add a "significant" column if it doesn't exist
    df = df.query("covariate != 'intercept'")

    df.loc[df['pval'] <= alpha, 'significant'] = 1
    df['significant'] = df['significant'].fillna(0).astype(int)

    # plot asterisks instead of p-values
    df.loc[df['pval'] <= 1e-4, 'asterisk'] = '****'
    df.loc[df['pval'] <= 1e-3, 'asterisk'] = '***'
    df.loc[df['pval'] <= 1e-2, 'asterisk'] = '**'
    df.loc[df['pval'] <= 5e-2, 'asterisk'] = '*'
    df.loc[df['pval'] > 5e-2, 'asterisk'] = ''

    # improve the names for tick labels
    df['plot_column'] = df['covariate'].map(labels_dict)

    if sum(pd.isnull(df['plot_column'])) > 0:
        raise ValueError(f"{df.loc[pd.isnull(df['plot_column'])].covariate.values} don't have label mappings")
    
    # Sort by significance and value
    # df = df.sort_values(["significant", val_col], ascending=[False, False]).reset_index(drop=True)
    
    # check that everything in the dataframe is in the order list
    # assert len(set(df['covariate']) - set(covariates_order)) == 0
    
    covariates_order = [covar for covar in covariates_order if covar in df['covariate'].values]
    
    # sort by provided order of variables
    df = df.set_index('covariate').loc[covariates_order].reset_index()
    
    df.loc[df['significant']==1, 'point_color'] = 'darkorange'
    df.loc[df['significant']==1, 'err_color'] = 'darkorange'
    df.loc[df['significant']==0, 'point_color'] = 'black'
    df.loc[df['significant']==0, 'err_color'] = 'gray'

    # Plotting
    fig, ax = plt.subplots(figsize=(6, len(df) * 0.6))

    conf_lower_col = f"{val_col}_lower"
    conf_upper_col = f"{val_col}_upper"
    
    def format_sigfig_fixed(x, sig=2):
        if x == 0:
            return 0.00
        elif x < 0.01:
            # return these in scientific notation because they are so small
            return "{:.1e}".format(x)
        elif x < 0.1:
            return np.round(x, 2)
        else:
            # Determine how many decimal places to show
            digits = sig - int(np.floor(np.log10(abs(x)))) - 1
            if digits < 0:
                # Round to nearest 10s, 100s, etc.
                rounded = round(x, digits)
                return f"{rounded:.0f}"
            else:
                return f"{x:.{digits}f}"
    
        # digits = sig - int(np.floor(np.log10(abs(x)))) - 1
        # return f"{round(x, digits):f}".rstrip('0').rstrip('.')  # strip trailing 0s and .
        
    for i, row in df.iterrows():
    
        # Plot significant predictors in orange
        ax.errorbar(
            row[val_col], 
            i,
            xerr = [[row[val_col] - row[conf_lower_col]],
                    [row[conf_upper_col] - row[val_col]]],
            fmt='o', color=row['point_color'], ecolor=row['err_color'], markeredgewidth=0.7, markeredgecolor='black', capsize=3
        )

        # ax.text(df[conf_upper_col].max() * pval_offset, i + 0.1, f"p = {format_sigfig_fixed(row['pval'])}")
        ax.text(df[conf_upper_col].max() * pval_offset, i + 0.1, row['asterisk'])

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
    
    if log:
        plt.xscale('log', base=10)
        
        # change the tick labels from 10^0 to 1
        ax.set_xticks(ax.get_xticks(), [format_sigfig_fixed(val) for val in ax.get_xticks()])
        
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])

    # Show or save the plot
    if saveName is None:
        plt.show()
    else:
        plt.savefig(saveName, bbox_inches='tight')
        plt.close()
        
        
        
        
        
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





def read_combine_all_TRUST_data(patient_WGS_data_fName, drug_lineage_inclusion_dict, CNN_results_dir="/n/data1/hms/dbmi/farhat/Sanjana/CNN_results", F2_thresh=0.03, baseline_only=True):
    '''
    This function keeps only measured MICs and WGS samples taken in the first two weeks of treatment because we are interested in associating baseline characteristics with outcome.
    '''

    ############################################# STEP 1: READ IN THE COMBINED PATIENT-WGS DATAFRAME #############################################


    # exclude sample MFS-742 (pid T0114). There are two samples for this pid at week 4. One matches the lineage of the week 1 sample, and the other (MFS-742) does not
    df_trust_patients = pd.read_csv(patient_WGS_data_fName).query("SampleID != 'MFS-742'")
    
    # remove reenrollments. They were already removed from the outcomes table, but do it here to get accurate counts of included/excluded participants
    df_trust_patients = df_trust_patients.loc[pd.isnull(df_trust_patients['screen_prevpid'])]
    
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
    df_trust_patients['Sampling_Week'] = df_trust_patients['Original_ID'].str.split('-').str[1]
    
    # replace month 5 with 20 for weeks
    df_trust_patients['Sampling_Week'] = df_trust_patients['Sampling_Week'].replace('01A', '01').replace('m5', '20')
    
    df_trust_patients['Sampling_Week'] = df_trust_patients['Sampling_Week'].astype(int)
    
    # keep only WGS samples collected in the first 2 weeks
    if baseline_only:
        df_trust_patients = df_trust_patients.query("Sampling_Week <= 2").reset_index(drop=True)
    
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




def determine_MIC_binarization_threshold(df, col, include_median=False, verbose=False):
    '''
    Use this function to determine a threshold to binarize MICs -- predicted or measured -- into two equal sized groups. 
    '''

    df_copy = df.copy()

    # just return the median??? which is the middle value
    bisection_value = np.median(df_copy[col].dropna())

    # exclusive by default because if you exclude the median, then because many isolates have the value, the high class will have fewer
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






def dummy_encode_lineages(df, lineage_col, binary=False):
    '''
    This will include mixed infections if listed in the Coll2014 column
    
    If binary, then make a single lineage variable that is 1 if the lineage is NOT the most common lineage and 0 if the sample is the most common lineage
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
    
    # consider the baseline to be the majority lineage (most 1s). So when you sort by the sum, take the last one
    baseline_lineage = df_lineage[unique_lineages].sum(axis=0).sort_values().index.values[-1]
    
    # create a single column that is 1 if the baseline_lineage column is 0. Else 0
    if binary:
        # df_lineage[f'Not_{baseline_lineage}'] = 1 - df_lineage[baseline_lineage]
        
        # later functions take all the columns in df_lineage that are not pid, so make sure that only relevant lineage columns are returned
        # return df_lineage[['pid', f'Not_{baseline_lineage}']]
        return df_lineage[['pid', baseline_lineage]]
    else:
        del df_lineage[baseline_lineage]
        return df_lineage





def process_input_features_for_model(df, model_cols, stratify_variables=[], MIC_type='none', include_drugs=[], binarize_drugs=[], log_transform_drugs=[], interact_MIC_lineage=False, interact_indel_change_lineage=False, binary_lineage=False):

    df_model = df.copy()
    cols_lst = model_cols.copy()
    
    # remove the stratify covariates from cols_lst. This is mainly for the HIV_CD4 variable, which gets changed due to having more than 2 levels
    cols_lst = list(set(cols_lst) - set(stratify_variables))

    df_model['high_lung_involvement'] = (df_model['predicted_PLI'] > 20).astype(int)

    # this is the imputed smear grade sample 1. Mapping using smear_encoding_dict has already been done
    if 'smear_grade_1' in df_model.columns:
        df_model['smear_grade_baseline'] = df_model['smear_grade_1'].copy()
    
    elif 'smear_positivity_1' in df_model.columns:
        df_model['smear_positivity_baseline'] = df_model['smear_positivity_1'].copy()

    if 'underweight' in model_cols:
        df_model['underweight'] = (df_model['bl_bmi'] < 18).astype(int)

    if len(binarize_drugs) > 0 and MIC_type != 'none':
                
        if MIC_type == 'predicted':
            cols_to_binarize = [f"{drug}_pred_MIC" for drug in binarize_drugs]
            non_binarized_cols = [f"{drug}_pred_MIC" for drug in include_drugs if drug not in binarize_drugs]
            # include_median = False
        
        elif MIC_type == 'measured':
            cols_to_binarize = [f"{drug}_midpoint" for drug in binarize_drugs]
            non_binarized_cols = [f"{drug}_midpoint" for drug in include_drugs if drug not in binarize_drugs]
            # include_median = False

        binarized_cols = []
        
        for col in cols_to_binarize:
            df_model = determine_MIC_binarization_threshold(df_model, col, include_median=False, verbose=True)
            binarized_cols.append(f"{col}_high")
        
        MIC_cols = binarized_cols + non_binarized_cols
    
    else:
        if MIC_type == 'predicted':
            MIC_cols = [f"{drug}_pred_MIC" for drug in include_drugs]

        elif MIC_type == 'measured':
            MIC_cols = [f"{drug}_midpoint" for drug in include_drugs]
        else:
            MIC_cols = []
                        
    # dummy encode lineages, accounting for mixed infections
    if 'Lineage' in model_cols and 'Lineage' not in stratify_variables:
        df_lineage = dummy_encode_lineages(df_model, 'Lineage', binary=binary_lineage)
        df_model = df_model.merge(df_lineage)
        
        # remove this from the list of predictors and add in the dummmy-encoded lineage names
        cols_lst.remove('Lineage')
        lineage_cols = list(set(df_lineage.columns) - set(['pid']))
        cols_lst += lineage_cols
        
        # interact the lineage with the indel change variable because the change could be different based on the genetic background of the strain
        if interact_indel_change_lineage and 'Indel_Change' in cols_lst:
            for col in lineage_cols:
                df_model[f"Indel_Change_x_{col}"] = df_model['Indel_Change'] * df_model[col]
                cols_lst.append(f"Indel_Change_x_{col}") 
                
        if interact_MIC_lineage and len(MIC_cols) > 0:
            for lineage_col in lineage_cols:
                for MIC_col in MIC_cols:
                    df_model[f"{lineage_col}_{MIC_col}"] = df_model[lineage_col] * df_model[MIC_col]
                    cols_lst.append(f"{lineage_col}_{MIC_col}") 
    
    features_lst = np.unique(list(cols_lst) + list(MIC_cols)) 

    # HIV_CD4 is a categorical variable with more than 2 levels, so need to dummy encode
    if 'HIV_CD4' in features_lst and 'HIV_CD4' not in stratify_variables:
            
        # this removes the first column (which is No HIV), so no HIV is the baseline
        df_model = pd.get_dummies(df_model, columns=['HIV_CD4'], drop_first=True)
        
        # rename to easier names
        df_model = df_model.rename(columns={'HIV_CD4_1.0': 'HIV_High_CD4', 'HIV_CD4_2.0': 'HIV_Low_CD4'})
        
        # convert from bool to int
        df_model[['HIV_High_CD4', 'HIV_Low_CD4']] = df_model[['HIV_High_CD4', 'HIV_Low_CD4']].astype(int)
    
        # remove HIV_CD4 from features_lst and add in the dummy variables
        features_lst = list(set(features_lst) - set(['HIV_CD4']))
        features_lst += ['HIV_High_CD4', 'HIV_Low_CD4']
        
    if 'high_bacterial_burden' in features_lst and 'high_bacterial_burden' not in stratify_variables:
        df_model.loc[(df_model['TTP_baseline'] <= 200), 'high_bacterial_burden'] = 1
        df_model.loc[(pd.isnull(df_model['TTP_baseline'])) & (df_model['smear_grade_baseline'] >= 3), 'high_bacterial_burden'] = 1
        df_model['high_bacterial_burden'] = df_model['high_bacterial_burden'].fillna(0).astype(int)
        # print(df_model['high_bacterial_burden'].value_counts())

    # make sure to keep the stratify_variables here as well
    df_model = df_model.set_index('pid')[list(features_lst) + list(stratify_variables)].reset_index() 
    
    # print(df_model.pid.nunique())
    # print(df_model.columns)

#     for col in df_model.columns:
#         print(col, sum(pd.isnull(df_model[col])))
            
    df_model = df_model.dropna()

    # variables with high degree of skew or needed to satisfy the proportional hazards assumption
    # log_transform_cols = np.unique(log_metadata_cols + [col for col in features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
    if MIC_type == 'predicted':
        MIC_log_transform_cols = [f"{drug}_pred_MIC" for drug in list(set(include_drugs).intersection(log_transform_drugs))]
    elif MIC_type == 'measured':
         MIC_log_transform_cols = [f"{drug}_midpoint" for drug in list(set(include_drugs).intersection(log_transform_drugs))]
    else:
         MIC_log_transform_cols = []
            
    print(MIC_log_transform_cols)
    log_transform_cols = np.unique(log_metadata_cols + MIC_log_transform_cols)

    for col in log_transform_cols:

        if col in df_model.columns:
            
            df_model[col] = np.log2(df_model[col].astype(float))

            # smallest value is 0, so replace with 0 after you do the log-transform. This makes the distribution more continuous than i.e. replacing 0 with a very small value
            # because the log of that very small value will be very negative instead of close to 0. It will be computed as -inf
            df_model[col] = df_model[col].replace(-np.inf, 0)
            
    # drop duplicate columns if there are any
    df_model = df_model.loc[:, ~df_model.columns.duplicated(keep='first')]

    return df_model.drop_duplicates(), features_lst




def fit_cox_hazard_ratio_model(df, df_outcome, cols_lst, event_col, time_col, MIC_type='none', include_drugs=drugs_lst, binarize_drugs=[], log_transform_drugs=[], stratify_variables=[], non_linear_term_variables=[], interact_MIC_lineage=False, interact_indel_change_lineage=False, cluster_col=None, penalize_features=[], binary_lineage=False):
    
    # the penalize_features argument is not used, but I wanted to keep the arguments the same for both the fit_cox_hazard_ratio_model and fit_cox_hazard_ratio_model_with_L2_penalty functions
    
    # Process input features
    df_model_processed, features_lst = process_input_features_for_model(df, 
                                                                        cols_lst, 
                                                                        stratify_variables=stratify_variables, 
                                                                        MIC_type=MIC_type, 
                                                                        include_drugs=include_drugs, 
                                                                        binarize_drugs=binarize_drugs, 
                                                                        log_transform_drugs=log_transform_drugs,
                                                                        interact_MIC_lineage=interact_MIC_lineage,
                                                                        interact_indel_change_lineage=interact_indel_change_lineage,
                                                                        binary_lineage=binary_lineage
                                                                       )
    
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
    
#     for col in features_lst + [time_col, event_col, 'unique_patient']:
#         num_na = sum(pd.isnull(df_model_processed[col]))

#         if num_na > 0:
#             print(col, num_na, df_model_processed.loc[pd.isnull(df_model_processed[col])])

    # need to define the cubic basis splines with a formula call to fit(). Use the training dataframe to get the bounds because it will have had log-transforms and standard scaling done to it
    non_linear_term_vars_min = [int(np.floor(df_model_processed[variable].min())) for variable in non_linear_term_variables]
    non_linear_term_vars_max = [int(np.ceil(df_model_processed[variable].max())) for variable in non_linear_term_variables]

    # remove them from the features list
    linear_term_variables = list(set(features_lst) - set(non_linear_term_variables))

    # if you're stratifying by categorical variables, have to remove them from the formula
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
            
    # if there are no variables to stratify by, the argument must be None, per the Cox model function
    if len(stratify_variables) == 0:
        stratify_variables = None
        
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

    # Last step: undo the variable transformations. First, we log2-transformed, then standard-scaled. So have to undo in the reverse order
    # 1) Undo the standard-scaling
    df_model_results['original_mean'] = df_model_results.index.map(means_dict)
    df_model_results['original_std'] = df_model_results.index.map(std_dict)

    # have to add in the mean and std (from the original variables) to each spline of the non linear variables
    for variable in non_linear_term_variables:
        df_model_results.loc[df_model_results.index.str.contains(variable), 'original_mean'] = means_dict[variable]
        df_model_results.loc[df_model_results.index.str.contains(variable), 'original_std'] = std_dict[variable]
    
    df_model_results['coef_transformed'] = df_model_results['coef'] / df_model_results['original_std']
    df_model_results['se_transformed'] = df_model_results['se(coef)'] / df_model_results['original_std']

#     # 2) Undo the log2-transform for the variables that were log2-transformed
#     # To do this, exponentiate the coefficients, so 2**coef. SE is approximately ln(2) * 2**coef * SE(coef)
#     # log_transform_cols = np.unique(log_metadata_cols + [col for col in features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
#     if MIC_type == 'predicted':
#         MIC_log_transform_cols = [f"{drug}_pred_MIC" for drug in list(set(include_drugs).intersection(log_transform_drugs))]
#     elif MIC_type == 'measured':
#          MIC_log_transform_cols = [f"{drug}_midpoint" for drug in list(set(include_drugs).intersection(log_transform_drugs))]
#     else:
#          MIC_log_transform_cols = []
            
#     log_transform_cols = np.unique(log_metadata_cols + MIC_log_transform_cols)
    
#     # the current coefficient is the factor increase if the value is multiplied by the base. i.e. if log2-transformed with beta = 2, then a doubling of x leads to a 2 * 2 = 4 multiplier on the log HR
#     # so to scale it to the original scale, you would multiply by the base of the logarithm you took
#     df_model_results.loc[df_model_results.index.isin(log_transform_cols), 'coef_transformed'] = 2 * df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['coef_transformed'] #np.exp(df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['coef_transformed'])

#     # is it the same transformation?
#     df_model_results.loc[df_model_results.index.isin(log_transform_cols), 'se_transformed'] = 2 * df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['se_transformed'] 

    return df_model_results, df_model_processed, cph





def fit_cox_hazard_ratio_model_with_L2_penalty(df, df_outcome, cols_lst, event_col, time_col, MIC_type='none', include_drugs=drugs_lst, binarize_drugs=[], log_transform_drugs=[], stratify_variables=[], non_linear_term_variables=[], interact_MIC_lineage=False, interact_indel_change_lineage=False, cluster_col=None, penalize_features=[], binary_lineage=False):
    
    # Process input features
    df_model_processed, features_lst = process_input_features_for_model(df, 
                                                                        cols_lst, 
                                                                        stratify_variables=stratify_variables, 
                                                                        MIC_type=MIC_type, 
                                                                        include_drugs=include_drugs, 
                                                                        binarize_drugs=binarize_drugs, 
                                                                        log_transform_drugs=log_transform_drugs,
                                                                        interact_MIC_lineage=interact_MIC_lineage,
                                                                        interact_indel_change_lineage=interact_indel_change_lineage,
                                                                        binary_lineage=binary_lineage
                                                                       )
        
    # Add outcome data. Reset index so that indices are in the appropriate order for straitifed k-fold CV
    df_model_processed = df_model_processed.merge(df_outcome, on='pid').reset_index(drop=True)
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
        
    # need to define the cubic basis splines with a formula call to fit(). Use the training dataframe to get the bounds because it will have had log-transforms and standard scaling done to it
    non_linear_term_vars_min = [int(np.floor(df_model_processed[variable].min())) for variable in non_linear_term_variables]
    non_linear_term_vars_max = [int(np.ceil(df_model_processed[variable].max())) for variable in non_linear_term_variables]

    # remove them from the features list
    linear_term_variables = list(set(features_lst) - set(non_linear_term_variables))

    # if you're stratifying by categorical variables, have to remove them from the formula
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
            
    # if there are no variables to stratify by, the argument must be None, per the Cox model function
    if len(stratify_variables) == 0:
        stratify_variables = None
                
    # perform cross-validation to select the strength of the L2 penalty
    alphas_lst = np.logspace(-5, 5, 11)
    
    param_to_maximize = 'log_likelihood'
    
    cv_results = pd.DataFrame(columns = ['alpha', 'CV', param_to_maximize])
    k = 0

    for _, alpha in enumerate(alphas_lst):
                
        # only penalize the columns that are passed in. First check that all the argument columns are in features_lst
        if len(set(penalize_features) - set(features_lst)) > 0:
            raise ValueError(f"{set(penalize_features) - set(features_lst)} features are in penalize_features but not features_lst")
            
        # next, get the indices of the columns in penalize_features in order to generate an array of penalty terms
        penalize_features_indices = [features_lst.index(col) for col in penalize_features]
        
        # initialize with all penalty = 0
        penalty_array = np.zeros(len(features_lst))
        
        # then change each index that should be penalized to alpha
        penalty_array[penalize_features_indices] = alpha
        
        # perform 5-fold cross-validation. Manually using sklearn instead of using lifelines.utils.k_fold_cross_validation because you can't stratify
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        # stratify by event
        for i, (train_index, test_index) in enumerate(skf.split(df_model_processed, df_model_processed[event_col])):
            
            # initialize model
            cph = lifelines.CoxPHFitter(penalizer=penalty_array, l1_ratio=0)
        
            cph.fit(df_model_processed.iloc[train_index],
                    time_col,
                    event_col=event_col,
                    cluster_col=cluster_col,
                    fit_options={'step_size': 0.1},
                    strata=stratify_variables,
                    formula=model_formula
                   )
            
            # cph.log_likelihood_ returns the LL on the training data. cph.score(df) returns the LL on a new dataset (pass in the test data)
            cv_results.loc[k, :] = [alpha, i+1, cph.score(df_model_processed.iloc[test_index])]
            k += 1
        
#         # Run 5-fold CV with the custom fit function
#         cv_scores = lifelines.utils.k_fold_cross_validation(cph, 
#                                                             df_model_processed, 
#                                                             time_col, 
#                                                             event_col=event_col, 
#                                                             k=5, 
#                                                             scoring_method="log_likelihood", 
#                                                             fitter_kwargs={
#                                                                          'cluster_col': cluster_col, 
#                                                                          'fit_options':{'step_size': 0.1},
#                                                                          'strata': stratify_variables,
#                                                                          'formula': model_formula
#                                                                         })
            
#         cv_results.append(pd.DataFrame({'alpha': alpha, 'CV': np.arange(5)+1, param_to_maximize: cv_scores}))
        
#     cv_results = pd.concat(cv_results)
#     cv_results.columns = ['alpha', 'CV', 'log_likelihood']
            
#     # Higher log-likelihood is a better model is better
    best_penalty = cv_results.groupby('alpha')[param_to_maximize].mean().sort_values().index.values[-1]
    print(f"Fitting model with L2 penalty of {best_penalty} with {param_to_maximize} = {np.max(cv_results.groupby('alpha')[param_to_maximize].mean())}")
    
    # need this again for the final model
    penalize_features_indices = [features_lst.index(col) for col in penalize_features]

    # initialize with all penalty = 0
    penalty_array = np.zeros(len(features_lst))

    # then change each index that should be penalized to the best penalty
    penalty_array[penalize_features_indices] = best_penalty
    
    print(dict(zip(features_lst, penalty_array)))
              
    # fit the new model
    cph = lifelines.CoxPHFitter(penalizer=penalty_array, l1_ratio=0)

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

    # Last step: undo the variable transformations. First, we log2-transformed, then standard-scaled. So have to undo in the reverse order
    # 1) Undo the standard-scaling
    df_model_results['original_mean'] = df_model_results.index.map(means_dict)
    df_model_results['original_std'] = df_model_results.index.map(std_dict)

    # have to add in the mean and std (from the original variables) to each spline of the non linear variables
    for variable in non_linear_term_variables:
        df_model_results.loc[df_model_results.index.str.contains(variable), 'original_mean'] = means_dict[variable]
        df_model_results.loc[df_model_results.index.str.contains(variable), 'original_std'] = std_dict[variable]
    
    df_model_results['coef_transformed'] = df_model_results['coef'] / df_model_results['original_std']
    df_model_results['se_transformed'] = df_model_results['se(coef)'] / df_model_results['original_std']

#     # 2) Undo the log2-transform for the variables that were log2-transformed
#     # To do this, exponentiate the coefficients, so 2**coef. SE is approximately ln(2) * 2**coef * SE(coef)
#     # log_transform_cols = np.unique(log_metadata_cols + [col for col in features_lst if col.endswith('_pred_MIC') or col.endswith('_midpoint')])
#     if MIC_type == 'predicted':
#         MIC_log_transform_cols = [f"{drug}_pred_MIC" for drug in list(set(include_drugs).intersection(log_transform_drugs))]
#     elif MIC_type == 'measured':
#          MIC_log_transform_cols = [f"{drug}_midpoint" for drug in list(set(include_drugs).intersection(log_transform_drugs))]
#     else:
#          MIC_log_transform_cols = []
            
#     log_transform_cols = np.unique(log_metadata_cols + MIC_log_transform_cols)    
#     # the current coefficient is the factor increase if the value is multiplied by the base. i.e. if log2-transformed with beta = 2, then a doubling of x leads to a 2 * 2 = 4 multiplier on the log HR
#     # so to scale it to the original scale, you would multiply by the base of the logarithm you took
#     df_model_results.loc[df_model_results.index.isin(log_transform_cols), 'coef_transformed'] = 2 * df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['coef_transformed'] 

#     # same for the standard error
#     df_model_results.loc[df_model_results.index.isin(log_transform_cols), 'se_transformed'] = 2 * df_model_results.loc[df_model_results.index.isin(log_transform_cols)]['se_transformed'] 

    return df_model_results, df_model_processed, cph




def fit_cox_models_all_imputations(df_trust_patients, df_pred_combined, TRUST_phenos, df_outcome, cols_lst, event_col, time_col, alpha=0.05, invert_OR=True, exclude_resistance=False, MIC_type=None, include_drugs=[], binarize_drugs=[], log_transform_drugs=[], tb_deaths_only=False, stratify_variables=[], non_linear_term_variables=[], interact_MIC_lineage=False, interact_indel_change_lineage=False, cluster_col=None, L2_penalty=False, penalize_features=[], binary_lineage=False):

    coef_col = 'coef_transformed'
    se_col = 'se_transformed'

    df_estimates = []

    # keep track of log-likelihoods across imputations because need to match them to get the difference between nested models
    log_like_imputations = []
    cph_models_imputations = []
    df_model_processed_imputations = [] # need to keep track of the individual training datasets as well for testing the proportional hazards assumptions
    
    df_model = df_trust_patients.copy()
    
    if 'imp_num' not in df_outcome.columns:
        df_outcome['imp_num'] = 1
    
    imp_lst = np.arange(df_outcome['imp_num'].max()) + 1

    # add measured and predicted MICs, then the process_input_features_for_model function will keep only those that should be here, determined by the MIC_type argument
    if MIC_type == 'predicted':
        df_model = df_model.merge(df_pred_combined, on='pid')

    elif MIC_type == 'measured':
        df_model = df_model.merge(TRUST_phenos, on='pid')
    
    # skip the first one because it's the unimputed one one. There will be N + 1 unique values in the .imp column
    for imp_num in imp_lst:

        if 'imp_num' in df_outcome.columns:
            df_outcome_single_imputation = df_outcome.query("imp_num==@imp_num")
        else:
            df_outcome_single_imputation = df_outcome.copy()
            
        if 'imp_num' in df_model.columns:
            df_single_model = df_model.query("imp_num==@imp_num")
        else:
            df_single_model = df_model.copy()

        if exclude_resistance:
            
            # remove all pids with measured resistant MICs
            for col in df_single_model.columns[df_single_model.columns.str.contains('_upper_bound')]:
                
                drug = col.split("_")[0]

                drug_full_name = abbr_drug_dict[drug]
                
                # all measured MICs have been normalized to MGIT, which is the majority medium. So get the critical concentration for MGIT
                mgit_cc = cc_df.query("Medium=='MGIT' & Drug==@drug_full_name").Value.values[0]
    
                # keep only pids where the upper bound is less than or equal to the critical concentration.
                # Do the inverse (exclude pids where the upper bound is greater than the CC because of MICs that are NA
                # print(drug, mgit_cc, len(df_single_model.query(f"~({drug}_upper_bound > @mgit_cc)")))
                num_measured_resistant = df_single_model.query(f"{drug}_upper_bound > @mgit_cc").pid.nunique()
                df_single_model = df_single_model.query(f"~({drug}_upper_bound > @mgit_cc)")

                isolates_with_genotypic_resistance = df_WHO_variants.query("drug==@drug_full_name").SampleID.values
                num_genotypic_resistant = df_single_model.query("SampleID in @isolates_with_genotypic_resistance").pid.nunique()
                df_single_model = df_single_model.query("SampleID not in @isolates_with_genotypic_resistance")

                if imp_num in [0, 1]:
                    print(f"    Dropped {num_measured_resistant} patients with measured and {num_genotypic_resistant} patients with genotypic {drug} resistance")

        if tb_deaths_only:

            non_tb_death_pids = df_outcome_single_imputation.query("event_type == 'death' & TB_death == 0").pid.unique()
            
            if imp_num in [0, 1]:
                print(f"    Censored {len(non_tb_death_pids)} patients with non-TB deaths")
    
            # censor the patients who died of non-TB causes
            df_outcome_single_imputation.loc[df_outcome_single_imputation['pid'].isin(non_tb_death_pids), 'event'] = 0
            
        if L2_penalty:
            model_func = fit_cox_hazard_ratio_model_with_L2_penalty
        else:
            model_func = fit_cox_hazard_ratio_model
            
        df_model_results, df_model_processed, cph = model_func(df_single_model, 
                                                               df_outcome_single_imputation,
                                                               cols_lst,
                                                               event_col,
                                                               time_col,
                                                               MIC_type=MIC_type,
                                                               include_drugs=include_drugs,
                                                               binarize_drugs=binarize_drugs,
                                                               log_transform_drugs=log_transform_drugs,
                                                               stratify_variables=stratify_variables,
                                                               non_linear_term_variables=non_linear_term_variables,
                                                               interact_MIC_lineage=interact_MIC_lineage, 
                                                               interact_indel_change_lineage=interact_indel_change_lineage,
                                                               cluster_col=cluster_col,
                                                               penalize_features=penalize_features,
                                                               binary_lineage=binary_lineage
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

        # need to keep the original standard deviation (keep mean for comparison sake) to transform later. But need to transform after 
        df_save = df_model_results[['coef', 'exp(coef)', 'se(coef)', 'coef_transformed', 'se_transformed']]
        df_save['imp_num'] = imp_num
        df_estimates.append(df_save.reset_index())

    df_estimates = pd.concat(df_estimates)

    # finally, pool results across the models. Doesn't matter which dataframe you take the length of, they all have the same pids
    if len(imp_lst) > 1:
        return pool_imputation_results(df_estimates, df_trust_patients.pid.nunique(), coef_col, se_col, alpha=alpha, invert_OR=invert_OR), df_model_processed_imputations, cph_models_imputations
    else:
        # add p-values. The Wald stat follows the normal distribution
        # survival function = 1 - CDF, which is the probability of being greater than the critical value. Take the absolute value so that you get the positive z-score
        df_estimates['pval'] = 2 * st.norm.sf(np.abs(df_estimates[coef_col] / df_estimates[se_col]))

        # add confidence intervals
        df_estimates['OR'] = np.exp(df_estimates[coef_col])
        df_estimates['OR_lower'] = np.exp(df_estimates[coef_col] + st.norm.ppf(alpha / 2) * df_estimates[se_col])
        df_estimates['OR_upper'] = np.exp(df_estimates[coef_col] + st.norm.ppf(1 - alpha / 2) * df_estimates[se_col])
        
        # return the last two inside lists for consistency with the case when there are imputations
        return df_estimates, [df_model_processed], [cph]
    
    
    


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




def run_LRT_single_predictor(test_covariate, df_trust_patients, df_pred_combined, TRUST_phenos, df_imputed_outcomes, df_final, cols_lst, event_col, time_col, tb_deaths_only=False, MIC_type='none', include_drugs=[], stratify_variables=None, non_linear_term_variables=[], cluster_col=None):

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
                                                                      alpha = 0.05,
                                                                      exclude_resistance=False,
                                                                      tb_deaths_only=tb_deaths_only,
                                                                      MIC_type=MIC_type,
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
                                                                          alpha = 0.05,
                                                                          exclude_resistance=False,
                                                                          tb_deaths_only=tb_deaths_only,
                                                                          MIC_type=MIC_type,
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




def combine_survival_probabilities_across_imputations(df_samples_imputations, cph_models_imputations, stratify_variables):

    df_survival_estimate_imputations = []
    
    for model_idx in range(len(cph_models_imputations)):
    
        df_survival_estimate = cph_models_imputations[model_idx].baseline_survival_.copy()
    
        # col_lst is a tuple of the values for each of the N covariate. len(col_lst) = N
        for col_tup in df_survival_estimate.columns:
    
            stratify_variables_min_vals = [np.min(df_samples_imputations[model_idx][col]) for col in stratify_variables]
            stratify_variables_max_vals = [np.max(df_samples_imputations[model_idx][col]) for col in stratify_variables]
    
            new_col = ''
            
            for i, val in enumerate(list(col_tup)):
        
                if val == stratify_variables_min_vals[i]:
                    new_col += f"{stratify_variables[i]}=0,"
                
                elif val == stratify_variables_max_vals[i]:
                    new_col += f"{stratify_variables[i]}=1,"
    
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

    assert df_plot_mean.variable.nunique() == np.exp2(len(stratify_variables))
    
    print(f"{df_plot_mean.variable.nunique()} unique condition combinations")

    df_plot_mean['variable'] = df_plot_mean['variable'].str.replace('cxr_cavity_chest_radiograph_1=0', 'No Cavitation').str.replace('cxr_cavity_chest_radiograph_1=1', 'Cavitation').str.replace('high_lung_involvement=0', '≤ 20% Lung Affected').str.replace('high_lung_involvement=1', '> 20% Lung Affected').str.replace('smear_pos_no_contam_sputum_specimen_1=0', 'Smear Negative').str.replace('smear_pos_no_contam_sputum_specimen_1=1', 'Smear Positive').str.replace(',', '\n')

    df_plot_std['variable'] = df_plot_std['variable'].str.replace('cxr_cavity_chest_radiograph_1=0', 'No Cavitation').str.replace('cxr_cavity_chest_radiograph_1=1', 'Cavitation').str.replace('high_lung_involvement=0', '≤ 20% Lung Affected').str.replace('high_lung_involvement=1', '> 20% Lung Affected').str.replace('smear_pos_no_contam_sputum_specimen_1=0', 'Smear Negative').str.replace('smear_pos_no_contam_sputum_specimen_1=1', 'Smear Positive').str.replace(',', '\n')

    # combine into a single dataframe
    return df_plot_mean.merge(df_plot_std)



def compute_relative_risk_single_imputation(df, variable):
    
    # compute relative risk. for HIV, do it each HIV group vs. the no HIV group
    # for HIV, only do it between low CD4 group and the other 2 groups
    if variable == 'HIV_Low_CD4':
        var_positive_no_conversion = len(df.query(f"{variable}==2 & culture_convert==0"))
        var_positive_conversion = len(df.query(f"{variable}==2 & culture_convert==1"))
        var_negative_no_conversion = len(df.query(f"{variable}==0 & culture_convert==0"))
        var_negative_conversion = len(df.query(f"{variable}==0 & culture_convert==1"))   
    else:
        var_positive_no_conversion = len(df.query(f"{variable}==1 & culture_convert==0"))
        var_positive_conversion = len(df.query(f"{variable}==1 & culture_convert==1"))
        var_negative_no_conversion = len(df.query(f"{variable}==0 & culture_convert==0"))
        var_negative_conversion = len(df.query(f"{variable}==0 & culture_convert==1"))

    # doesn't hold for the HIV variable
    # assert var_positive_no_conversion + var_positive_conversion + var_negative_no_conversion + var_negative_conversion == len(df)
    
    RR = (var_positive_no_conversion / (var_positive_no_conversion + var_negative_no_conversion)) / (var_negative_no_conversion / (var_negative_no_conversion + var_negative_conversion))
    
    # standard error of RR is expressed as stderr of log(RR)
    log_RR_stderr = np.sqrt(1 / var_positive_no_conversion + 1 / var_positive_conversion + 1 / var_negative_no_conversion + 1 / var_negative_conversion)
    
    # then return the log of relative risk
    return np.log(RR), log_RR_stderr



def compute_relative_risk_with_confidence_interval_on_imputations(df_trust_patients, df_TCC_imputed_all, variable):
    
    # include all imputations
    df_for_RR_calculation = df_trust_patients.merge(df_TCC_imputed_all.query("imp_num > 0"))

    df_for_RR_calculation['high_lung_involvement'] = (df_for_RR_calculation['predicted_PLI'] > 20)
    df_for_RR_calculation = df_for_RR_calculation.dropna(subset=cols_lst)
    df_for_RR_calculation['high_lung_involvement'] = df_for_RR_calculation['high_lung_involvement'].astype(int)

    # compute the relative risk for the variable on each imputation
    log_RRs_single_variable = []
    stderrs_log_RR_single_variable = []
    
    for num in df_for_RR_calculation.imp_num.unique():
        log_RR, log_RR_stderr = compute_relative_risk_single_imputation(df_for_RR_calculation.query("imp_num==@num"), variable)
        log_RRs_single_variable.append(log_RR)
        stderrs_log_RR_single_variable.append(log_RR_stderr)
        
    assert len(log_RRs_single_variable) == 30
    assert len(stderrs_log_RR_single_variable) == 30
        
    return log_RRs_single_variable, stderrs_log_RR_single_variable



def rubins_rules_relative_risk(log_estimates, stderr_log_estimates, alpha=0.05):
    
    # number of imputations
    m = len(log_estimates)
    
    # mean of the estimates. This is the log of the relative risk
    theta_bar = np.mean(log_estimates)
    
    # mean of the standard errors. This is also the within-imputation variance
    U_bar = np.mean(stderr_log_estimates)
    
    # between-imputation variance
    B = np.var(log_estimates, ddof=1)
    
    # total variance
    T = U_bar + (1 + 1/m) * B
    
    # degrees of freedom
    dof = (m - 1) * (1 + U_bar / ((1 + 1/m) * B))**2

    # confidence interval = mean +/- t* * np.sqrt(total_variance). Use the student's t distribution
    t_critical = np.abs(st.t.ppf(1 - alpha / 2, dof))
    
    # get bounds, and exponentiate because we took the log earlier
    lower_bound = np.exp(theta_bar - t_critical * np.sqrt(T))
    upper_bound = np.exp(theta_bar + t_critical * np.sqrt(T))

    # p-value for theta_bar not being 0 (meaning the relative risk is not 1)
    t_stat = (theta_bar - 0) / np.sqrt(T)
    pval = 2 * st.t.sf(abs(t_stat), dof)
    
    return lower_bound, upper_bound, np.exp(theta_bar), pval





def fit_poisson_regression_single_imputation(df, df_outcome, cols_lst, outcome_variable, binary_lineage=True):
    
    # Process input features
    df_model_processed, features_lst = process_input_features_for_model(df, 
                                                                        cols_lst, 
                                                                        stratify_variables=[], # don't stratify because the function will process the variables for model fitting
                                                                        MIC_type='none', # fit without MICs for this
                                                                        binary_lineage=binary_lineage
                                                                       )
    
    # Add outcome data
    df_model_processed = df_model_processed.merge(df_outcome, on='pid')

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

    formula = f"{outcome_variable} ~ " + " + ".join(features_lst)
    
    # Fit Poisson regression with log link
    poisson_model = smf.glm(
        formula=formula,
        data=df_model_processed,
        family=sm.families.Poisson()
    ).fit(cov_type='HC0')  # HC0 = robust standard errors

    rr_df = pd.DataFrame({
        'coef': poisson_model.params, # return the log(RR) because that's required for Rubin's rules. It will be exponentiated at the end
        'se(coef)': poisson_model.bse, # return standard errors for Rubin's rules,
    })
    
    # Last step: undo the variable transformations. First, we log2-transformed, then standard-scaled. So have to undo in the reverse order
    # 1) Undo the standard-scaling
    rr_df['original_mean'] = rr_df.index.map(means_dict)
    rr_df['original_std'] = rr_df.index.map(std_dict)

    rr_df['coef_transformed'] = rr_df['coef'] / rr_df['original_std']
    rr_df['se_transformed'] = rr_df['se(coef)'] / rr_df['original_std']
    
    # also return num_samples and num_covars for use in Rubin's rules
    return rr_df, df_model_processed.pid.nunique(), len(features_lst)




def compute_relative_risks_rubins_rules(df, df_outcome, cols_lst, outcome_variable, alpha=0.05):
    
    imputation_results = []

    # number of imputations
    num_impute = df_outcome.imp_num.max()

    # exclude imp_num == 0, which is the unimputed dataset
    for num in np.arange(1, num_impute + 1):

        rr_df, num_samples, num_covars = fit_poisson_regression_single_imputation(df, 
                                                                                 df_outcome.query("imp_num==@num"), 
                                                                                 cols_lst, 
                                                                                 outcome_variable, 
                                                                                 binary_lineage=True
                                                                                 )

        rr_df['imp_num'] = num

        imputation_results.append(rr_df)
        
    imputation_results = pd.concat(imputation_results).reset_index().rename(columns={'index': 'covariate'})

    coef_col = 'coef'
    se_col = 'se(coef)'
    
    # pooled log-RR estimate
    df_pool = pd.DataFrame(imputation_results.groupby("covariate")[coef_col].mean()).reset_index()

    # Within-imputation variance (Ū)
    imputation_results['squared_se'] = imputation_results[se_col]**2
    pooled_variances = pd.DataFrame(imputation_results.groupby("covariate")['squared_se'].mean()).rename(columns={'squared_se': 'V_w'})

    # Between-imputation variance (B)
    df_B = pd.DataFrame(imputation_results.groupby("covariate")[coef_col].var(ddof=1)).reset_index().rename(columns={coef_col: 'V_b'})

    # merge variance dataframes
    df_pool = df_pool.merge(pooled_variances, on='covariate').merge(df_B, on='covariate').rename(columns={coef_col: 'coef_pooled'})

    # Total variance (T)
    df_pool['V_t'] = df_pool['V_w'] + df_pool['V_b'] + df_pool['V_b'] / num_impute

    # Standard errors
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
    df_pool['pval'] = 2 * st.t.sf(abs(df_pool['wald_stat']), df_pool['dof_adj'])

    # compute confidence intervals. CI = coef_pooled ± t_stat * se_pooled
    df_pool['t_critical'] = np.abs(st.t.ppf(1 - alpha / 2, df_pool['dof_adj']))
    df_pool['coef_lower'] = df_pool['coef_pooled'] - df_pool['t_critical'] * df_pool['se_pooled']
    df_pool['coef_upper'] = df_pool['coef_pooled'] + df_pool['t_critical'] * df_pool['se_pooled']

    # get bounds, and exponentiate because we took the log earlier
    df_pool['HR_TCC_assoc'] = np.exp(df_pool['coef_pooled'])
    df_pool['HR_TCC_assoc_lower'] = np.exp(df_pool['coef_lower'])
    df_pool['HR_TCC_assoc_upper'] = np.exp(df_pool['coef_upper'])
    
    return df_pool
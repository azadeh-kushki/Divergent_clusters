import pandas as pd
import os
import sys
import numpy as np
import warnings
import itertools
from scipy import stats

# Main function for loading in the ABCD data
def abcd_load(root_dir, measure_file):

    # Mapped measures
    mapped_meas = pd.read_excel(measure_file)

    # Load in the demographics file
    abcd_basic_demo = pd.read_csv(os.path.join(root_dir, r'abcd_p_demo.csv'), sep=',')

    # Copy the dataframe
    df_temp_sex = abcd_basic_demo.copy()

    # Assign sex to the following categories: 1: M (male), 2: F (female), 3: IM (intersex-male), 4: IF (intersex-female)
    df_temp_sex['sex'], _ = abcd_map_sex(df_temp_sex['demo_sex_v2'].to_numpy())

    # Only keep the subject ID and sex columns, and only where sex has a value
    df_temp_sex = df_temp_sex[['src_subject_id', 'sex']]
    df_temp_sex = df_temp_sex.loc[df_temp_sex['sex'] != 'nan']

    # Merge sex by src_subject_id to populate all eventnames
    abcd_basic_demo = pd.merge(abcd_basic_demo, df_temp_sex, how='left', on='src_subject_id')

    # Copy the dataframe
    df_temp_race = abcd_basic_demo.copy()

    # Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
    # 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
    df_temp_race['race'], missing_value_race = abcd_map_race(abcd_basic_demo[[col for col in abcd_basic_demo.columns if 'demo_race_a_p' in col]].to_numpy())

    # Only keep the subject ID and race columns, and only where race has a value
    df_temp_race = df_temp_race[['src_subject_id', 'race']]
    df_temp_race = df_temp_race.loc[df_temp_race['race'] != 'nan']

    # Merge race by src_subject_id to populate all eventnames
    abcd_basic_demo = pd.merge(abcd_basic_demo, df_temp_race, how='left', on='src_subject_id')

    # Copy the dataframe
    df_temp_eth = abcd_basic_demo.copy()

    # Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
    # not reported
    df_temp_eth['ethnicity'], missing_value_eth = abcd_map_ethnicity(abcd_basic_demo['demo_ethn_v2'].to_numpy())

    # Only keep the subject ID and ethnicity columns, and only where ethnicity has a value
    df_temp_eth = df_temp_eth[['src_subject_id', 'ethnicity']]
    df_temp_eth = df_temp_eth.loc[df_temp_eth['ethnicity'] != 'nan']

    # Merge race by src_subject_id to populate all eventnames
    abcd_basic_demo = pd.merge(abcd_basic_demo, df_temp_eth, how='left', on='src_subject_id')

    # Make a scan_id column
    abcd_basic_demo['scan_id'] = abcd_basic_demo['src_subject_id'] + '_' + abcd_basic_demo['eventname']

    # Copy the dataframe
    df_temp_inc = abcd_basic_demo.copy()

    # Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
    # answer. Do this for the baseline and longitudinal data
    df_temp_inc['income_base'], _ = abcd_map_income(abcd_basic_demo['demo_comb_income_v2'].to_numpy())
    df_temp_inc['income_fu'], _ = abcd_map_income(abcd_basic_demo['demo_comb_income_v2_l'].to_numpy())

    # Assign income
    df_temp_inc.loc[df_temp_inc['eventname'] == 'baseline_year_1_arm_1', 'income'] = df_temp_inc.loc[df_temp_inc['eventname'] == 'baseline_year_1_arm_1', 'income_base']
    df_temp_inc.loc[df_temp_inc['eventname'] != 'baseline_year_1_arm_1', 'income'] = df_temp_inc.loc[df_temp_inc['eventname'] != 'baseline_year_1_arm_1', 'income_fu']

    # Only keep the scan ID and income columns, and only where income has a value
    df_temp_inc = df_temp_inc[['scan_id', 'income']]
    df_temp_inc = df_temp_inc.loc[~pd.isna(df_temp_inc['income'])]

    # Merge race by scan_id to populate all eventnames
    abcd_basic_demo = pd.merge(abcd_basic_demo, df_temp_inc, how='left', on='scan_id')

    # Copy the dataframe
    df_temp_edu = abcd_basic_demo.copy()

    # Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
    # nan: Don't know/refuse to answer. Do this for
    df_temp_edu['education_base'], _ = abcd_map_education(abcd_basic_demo['demo_prnt_ed_v2'].to_numpy(), abcd_basic_demo['demo_prtnr_ed_v2'].to_numpy())
    df_temp_edu['education_fu'], _ = abcd_map_education(abcd_basic_demo['demo_prnt_ed_v2_l'].to_numpy(), abcd_basic_demo['demo_prtnr_ed_v2_l'].to_numpy())

    # Assign education
    df_temp_edu.loc[df_temp_edu['eventname'] == 'baseline_year_1_arm_1', 'education'] = df_temp_edu.loc[df_temp_edu['eventname'] == 'baseline_year_1_arm_1', 'education_base']
    df_temp_edu.loc[df_temp_edu['eventname'] != 'baseline_year_1_arm_1', 'education'] = df_temp_edu.loc[df_temp_edu['eventname'] != 'baseline_year_1_arm_1', 'education_fu']

    # Only keep the scan ID and education columns, and only where education has a value
    df_temp_edu = df_temp_edu[['scan_id', 'education']]
    df_temp_edu = df_temp_edu.loc[~pd.isna(df_temp_edu['education'])]

    # Merge race by src_subject_id to populate all eventnames
    abcd_basic_demo = pd.merge(abcd_basic_demo, df_temp_edu, how='left', on='scan_id')

    # Rename the subject id column
    abcd_basic_demo = abcd_basic_demo.rename(columns={'src_subject_id': 'subject_id'})

    # Only keep certain columns
    abcd_df = abcd_basic_demo[['subject_id', 'scan_id', 'sex', 'race', 'ethnicity', 'income', 'education']]

    # Load in the demographics file containing the age
    abcd_age = pd.read_csv(os.path.join(root_dir, r'abcd_y_lt.csv'), sep=',')
    
    # Assign scan_id
    abcd_age['scan_id'] = abcd_age['src_subject_id'] + '_' + abcd_age['eventname']

    # Assign age
    abcd_age['age'] = abcd_age['interview_age']/12
    
    # Only keep certain variables
    abcd_age = abcd_age[['scan_id', 'age']]
    
    # Merge
    abcd_df = pd.merge(abcd_df, abcd_age, how='left', on='scan_id')

    # Load in the scanner file containing the hash
    abcd_scanner = pd.read_csv(os.path.join(root_dir, r'mri_y_adm_info.csv'), sep=',')

    # Assign scan_id
    abcd_scanner['scan_id'] = abcd_scanner['src_subject_id'] + '_' + abcd_scanner['eventname']

    # Rename the hash to scanner
    abcd_scanner = abcd_scanner.rename(columns={'mri_info_deviceserialnumber': 'scanner'})

    # Add the ABCD prefix
    abcd_scanner['scanner'] = 'ABCD-' + abcd_scanner['scanner']

    # Only keep certain variables
    abcd_scanner = abcd_scanner[['scan_id', 'scanner']]

    # Merge
    abcd_df = pd.merge(abcd_df, abcd_scanner, how='left', on='scan_id')

    # Get the unique types of the mapped variables
    var_u, ind = np.unique(mapped_meas['type'], return_index=True)
    var_u = var_u[np.argsort(ind)]

    # Iterate over the types
    for var in var_u:

        # Extract the corresponding variables
        vars_cur = mapped_meas.loc[mapped_meas['type'] == var]

        # Load in the variable file
        abcd_var = pd.read_csv(os.path.normpath(os.path.join(root_dir, vars_cur['abcd_path'].to_list()[0])), sep=',', low_memory=False)

        # Create the scan_id column
        abcd_var['scan_id'] = abcd_var['src_subject_id'] + '_' + abcd_var['eventname']

        # Only keep the columns we need
        abcd_var = abcd_var[['scan_id'] + vars_cur['abcd'].to_list()]

        # Rename
        abcd_var.columns = ['scan_id'] + vars_cur['measure'].to_list()

        # Merge
        abcd_df = pd.merge(abcd_df, abcd_var, how='outer', on='scan_id')

    # Load in two QC files
    abcd_qc1 = pd.read_csv(os.path.join(root_dir, r'mri_y_qc_incl.csv'), sep=',')
    abcd_qc2 = pd.read_csv(os.path.join(root_dir, r'mri_y_qc_man_fsurf.csv'), sep=',')

    # Add the scan_id columns
    abcd_qc1['scan_id'] = abcd_qc1['src_subject_id'] + '_' + abcd_qc1['eventname']
    abcd_qc2['scan_id'] = abcd_qc2['src_subject_id'] + '_' + abcd_qc2['eventname']

    # Get rid of rows in QC2 (FreeSurfer) that were not reviewed
    abcd_qc2 = abcd_qc2.loc[abcd_qc2['fsqc_nrev'] > 0]

    # Merge
    abcd_qc = pd.merge(abcd_qc1, abcd_qc2, how='outer', on='scan_id')

    # Assign T1 QC to the following categories: 1: Passed (1 or 2), 2: Failed (0), 3: Unknown.
    abcd_qc['t1_qc'], _ = abcd_map_qc(abcd_qc['fsqc_qc'].to_numpy(), abcd_qc['imgincl_t1w_include'])

    # Only keep the scan ID and qc columns, and only where qc has a value
    abcd_qc = abcd_qc[['scan_id', 't1_qc']]
    abcd_qc = abcd_qc.loc[abcd_qc['t1_qc'] != 'nan']

    # Merge
    abcd_df = pd.merge(abcd_df, abcd_qc, how='outer', on='scan_id')

    # Add dataset
    abcd_df['dataset'] = 'ABCD'

    # Fill in missing demographic for new timepoints
    abcd_fix = abcd_df.copy()
    abcd_fix = abcd_fix.loc[pd.isna(abcd_fix['subject_id']), :]
    abcd_fix['subject_id'] = abcd_fix['scan_id'].str.split('_', expand=True)[0] + '_' + abcd_fix['scan_id'].str.split('_', expand=True)[1]
    scan_id = abcd_fix['scan_id'].to_list()
    abcd_fix = abcd_fix[['subject_id']]
    abcd_fix = abcd_fix.reset_index().merge(abcd_df.loc[~pd.isna(abcd_df['subject_id']) &  abcd_df['scan_id'].str.contains('baseline'), :], how='left', on='subject_id').set_index('index')
    abcd_fix['scan_id'] = scan_id
    abcd_df.loc[pd.isna(abcd_df['subject_id'])] = abcd_fix

    # Assign missing values
    abcd_df.loc[pd.isna(abcd_df['scanner']), 'scanner'] = 'ABCD-Unknown'
    abcd_df.loc[pd.isna(abcd_df['sex']), 'sex'] = 'Unknown'
    abcd_df.loc[pd.isna(abcd_df['race']), 'race'] = missing_value_race
    abcd_df.loc[pd.isna(abcd_df['ethnicity']), 'ethnicity'] = missing_value_eth

    # Get rid of rows that don't have imaging
    abcd_df = abcd_df.loc[~pd.isna(abcd_df['SubCortGrayVol'])]

    # Make sure we have the basics: age and sex
    abcd_df = abcd_df.loc[abcd_df['sex'] != 'Unknwon']
    abcd_df = abcd_df.loc[~pd.isna(abcd_df['age'])]

    # Drop duplicate rows
    abcd_df = abcd_df.drop_duplicates()

    return abcd_df

# Assign sex to the following categories: 1: M (male), 2: F (female), 3: IM (intersex-male), 4: IF (intersex-female)
def abcd_map_sex(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col = np.where(input_col == 1, 'M', output_col)
    output_col = np.where(input_col == 2, 'F', output_col)
    output_col = np.where(input_col == 3, 'IM', output_col)
    output_col = np.where(input_col == 4, 'IF', output_col)
    output_col = np.where((input_col < 1) | (input_col > 4) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val


# Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
# 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
def abcd_map_race(input_array):

    # Copy input to output
    output_col = np.zeros((np.shape(input_array)[0]))
    output_col.fill(np.nan)

    # Set the value for missing
    missing_val = 'Cat-7'

    # Re-assign
    output_col = np.where((np.sum(input_array[:, [2,3]], axis=1) >= 1) & (np.sum(input_array[:, np.r_[0:2, 4:16]], axis=1) == 0), 'Cat-1', output_col)
    output_col = np.where((np.sum(input_array[:, np.r_[8:15]], axis=1) >= 1) & (np.sum(input_array[:, np.r_[0:8, 15]], axis=1) == 0), 'Cat-2', output_col)
    output_col = np.where((input_array[:, 1] == 1) & (np.sum(input_array[:, 0:16], axis=1) == 1), 'Cat-3', output_col)
    output_col = np.where((input_array[:, 0] == 1) & (np.sum(input_array[:, 0:16], axis=1) == 1), 'Cat-4', output_col)
    output_col = np.where((input_array[:, 15] == 1) & (np.sum(input_array[:, np.r_[0:15]], axis=1) == 0), 'Cat-6', output_col)
    output_col = np.where((np.sum(input_array[:, np.r_[4:8]], axis=1) >= 1) & (np.sum(input_array[:, np.r_[0:4, 8:16]], axis=1) == 0),'Cat-6', output_col)
    output_col = np.where((np.sum(input_array[:, 0:16], axis=1) > 1) & (output_col == '0.0'), 'Cat-5', output_col)
    output_col = np.where(np.sum(input_array[:, np.r_[16:19]], axis=1) >= 1, 'Cat-7', output_col)

    # Return
    return output_col, missing_val

# Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
# not reported
def abcd_map_ethnicity(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Cat-3'

    # Re-assign
    output_col = np.where(input_col == 1, 'Cat-1', output_col)
    output_col = np.where(input_col == 2, 'Cat-2', output_col)
    output_col = np.where(input_col > 2, 'Cat-3', output_col)

    # Return
    return output_col, missing_val


# Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
# answer
def abcd_map_income(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col = np.where((input_col >= 1) & (input_col <= 6), 1, output_col)
    output_col = np.where((input_col >= 6) & (input_col <= 9), 2, output_col)
    output_col = np.where(input_col == 10, 3, output_col)
    output_col = np.where((input_col > 10) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
# nan: Don't know/refuse to answer. Do this for each parent, and take the average
def abcd_map_education(input_col1, input_col2):

    # Copy input to output
    output_col1 = input_col1.copy()
    output_col2 = input_col2.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col1 = np.where(input_col1 <= 12, 1, output_col1)
    output_col1 = np.where((input_col1  >= 13) & (input_col1 <= 15), 2, output_col1)
    output_col1 = np.where((input_col1  >= 16) & (input_col1 <= 18), 3, output_col1)
    output_col1 = np.where((input_col1  >= 19) & (input_col1 <= 21), 4, output_col1)
    output_col1 = np.where(input_col1 > 21, missing_val, output_col1)
    output_col2 = np.where(input_col2 <= 12, 1, output_col2)
    output_col2 = np.where((input_col2 >= 13) & (input_col2 <= 15), 2, output_col2)
    output_col2 = np.where((input_col2 >= 16) & (input_col2 <= 18), 3, output_col2)
    output_col2 = np.where((input_col2 >= 19) & (input_col2 <= 21), 4, output_col2)
    output_col2 = np.where(input_col2 > 21, missing_val, output_col2)

    # Take the mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_col = np.nanmean(np.concatenate((output_col1.reshape(-1, 1), output_col2.reshape(-1, 1)), axis=1), axis=1)

    # Return
    return output_col, missing_val

# Assign T1 QC to the following categories: 1: Passed (1 or 2), 2: Failed (0), 3: Unknown.
def abcd_map_qc(input_col1, input_col2):

    # Copy input to output
    output_col = input_col1.copy()
    output_col.fill(np.nan)

    # Set the value for missing
    missing_val = 'Unknown'

    # Re-assign
    output_col = np.where((input_col1 == 1) & (np.isnan(input_col2)), 'Passed', output_col)
    output_col = np.where((input_col2 == 1) & (np.isnan(input_col1)), 'Passed', output_col)
    output_col = np.where((input_col1 == 1) & (input_col2 == 1), 'Passed', output_col)
    output_col = np.where((input_col1 == 0) & (np.isnan(input_col2)), 'Failed', output_col)
    output_col = np.where((input_col2 == 0) & (np.isnan(input_col1)), 'Failed', output_col)
    output_col = np.where((input_col1 == 0) & (input_col2 == 0), 'Failed', output_col)

    # Return
    return output_col, missing_val

def abcd_choose(abcd_df):

    # Filter by age
    abcd_df = abcd_df.loc[(abcd_df['age'] >= 6) & (abcd_df['age'] < 19)]

    # Filter by passed QC
    abcd_df = abcd_df.loc[abcd_df['t1_qc'] == 'Passed']

    # Get rid of scans where we only have 1 on a site
    site_counts = abcd_df['scanner'].value_counts()
    site_counts = site_counts[site_counts < 2]
    for site in site_counts.index:
        abcd_df = abcd_df[abcd_df['scanner'] != site]

    # Find outliers
    abcd_df['is_outlier'] = 0
    count = 0
    for col in abcd_df.columns:
        if (col.startswith('lh_') or col.startswith('rh_')) and col.endswith('_thick'):
            count = count + 1
            z = np.abs(stats.zscore(abcd_df[col]))
            abcd_df.loc[z > 3, 'is_outlier'] = abcd_df.loc[z > 3, 'is_outlier'] + 1

    # Remove outliers
    abcd_df = abcd_df[abcd_df['is_outlier'] / count <= 0.50]

    # Split into singles and duplicates
    abcd_df_dup = abcd_df[abcd_df['subject_id'].duplicated(keep=False)]
    abcd_df_sing = pd.concat([abcd_df, abcd_df_dup]).drop_duplicates(keep=False)

    # Keep all the singles
    abcd_df_new = abcd_df_sing.copy()

    # Fewer 4-year, so include all of those
    abcd_df_new = pd.concat((abcd_df_new, abcd_df_dup.loc[abcd_df_dup['scan_id'].str.contains('4_year_follow_up')]))
    yr4_subs = abcd_df_dup.loc[abcd_df_dup['scan_id'].str.contains('4_year_follow_up'), 'subject_id']
    abcd_df_dup = abcd_df_dup[~abcd_df_dup['subject_id'].isin(yr4_subs)]

    # Get the unique subjects
    subs_u = np.unique(abcd_df_dup['subject_id'])

    # Take every second as baseline, starting at 0, and year-2, starting at 1
    bl_ind = list(itertools.islice(subs_u, 0, None, 2))
    yr2_ind = list(itertools.islice(subs_u, 1, None, 2))

    # Take the baseline
    abcd_df_new = pd.concat((abcd_df_new, abcd_df_dup[abcd_df_dup['subject_id'].isin(bl_ind) & abcd_df_dup['scan_id'].str.contains('baseline')]))

    # Take the 2 year followup
    abcd_df_new = pd.concat((abcd_df_new, abcd_df_dup[abcd_df_dup['subject_id'].isin(yr2_ind) & abcd_df_dup['scan_id'].str.contains('2_year_follow_up')]))

    return abcd_df_new



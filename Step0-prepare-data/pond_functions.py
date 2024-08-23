import pandas as pd
import os
import numpy as np
import warnings
from scipy import stats

# Main function for loading in the HBN data
def pond_load(root_dir, measure_file):

    # Mapped measures
    mapped_meas = pd.read_excel(measure_file)

    # Load in the demographics file
    pond_demo = pd.read_csv(os.path.join(root_dir,'POND', r'pond_extract_updated_31may2023_abcd.csv'), sep=',', low_memory=False)

    # Only keep certain columns
    pond_demo = pond_demo[['subject', 'DOB','PRIMARY_DIAGNOSIS', 'NSI_SEX'] + [col for col in pond_demo.columns if 'ETHNCTY_CDN' in col] +
                          ['EDUC_PMRY_CGVR_1_STD', 'EDUC_PMRY_CGVR_2_STD', 'HSHLD_INCOME_STD']]

    # Drop duplicates
    pond_demo = pond_demo.drop_duplicates()

    # Remove rows which are all NaNs in the important columns
    pond_demo = pond_demo.dropna(how='all', subset=['NSI_SEX'] + [col for col in pond_demo.columns if 'ETHNCTY_CDN' in col] + ['EDUC_PMRY_CGVR_1_STD', 'EDUC_PMRY_CGVR_2_STD', 'HSHLD_INCOME_STD'])
    pond_demo = pond_demo.dropna(how='all', subset=['DOB'])

    if not np.shape(np.unique(pond_demo['subject']))[0] == np.shape(pond_demo)[0]:
        Exception('Demographics contains duplicate participants! Need to address.')

    # Rename the subject id column
    pond_demo = pond_demo.rename(columns={'subject': 'subject_id'})
    pond_demo['subject_id'] = pond_demo['subject_id'].astype('str')

    # Assign sex
    pond_demo['sex'] = pond_demo['NSI_SEX']
    pond_demo.loc[pond_demo['sex'] == 'Female', 'sex'] = 'F'
    pond_demo.loc[pond_demo['sex'] == 'Male', 'sex'] = 'M'

    # Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
    # 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
    pond_demo['race'], missing_value_race = pond_map_race(pond_demo[[col for col in pond_demo.columns if 'ETHNCTY_CDN' in col]].to_numpy())

    # Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
    # not reported
    pond_demo['ethnicity'], missing_value_eth = pond_map_ethnicity(pond_demo['ETHNCTY_CDN_10'].to_numpy())

    # Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
    # answer
    pond_demo['income'], _ = pond_map_income(pond_demo['HSHLD_INCOME_STD'])

    # Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
    # nan: Don't know/refuse to answer. Do this for each parent, and take the average
    pond_demo['education'], _ = pond_map_education(pond_demo['EDUC_PMRY_CGVR_1_STD'].to_numpy(), pond_demo['EDUC_PMRY_CGVR_2_STD'].to_numpy())
    
    pond_demo['Dx'] = pond_demo.PRIMARY_DIAGNOSIS
    # Assign the subject id from the unique subjects
    pond_df = pond_demo[['subject_id', 'DOB', 'sex', 'race', 'ethnicity', 'income', 'education','Dx']]

    # Get the unique types of the mapped variables
    var_u, ind = np.unique(mapped_meas['type'], return_index=True)
    var_u = var_u[np.argsort(ind)]

    # Iterate over the types - if not CBCL
    for var in var_u:

        if var != 'cbcl':

            # Extract the corresponding variables
            vars_cur = mapped_meas.loc[mapped_meas['type'] == var]

            # Load in the variable file
            pond_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'POND', vars_cur['pond_path'].to_list()[0])), sep=r'\t', engine='python')

            # Get the scan_id
            pond_var['scan_id'] = pond_var[vars_cur['pond_subid'].to_list()[0]].str.split('/freesurfer/', expand=True)[1].str.replace('/', '')

            # Assign the subject_id
            pond_var['subject_id'] = pond_var[vars_cur['pond_subid'].to_list()[0]].str.split('/freesurfer/', expand=True)[1].str.split('_', expand=True)[0].str.replace('sub-', '')
            pond_var.loc[pond_var['subject_id'].str.startswith('088'), 'subject_id'] = pond_var.loc[pond_var['subject_id'].str.startswith('088'), 'subject_id'].str[1:]

            # Only keep the variables we need
            pond_var = pond_var[['subject_id', 'scan_id'] + vars_cur['pond'].to_list()]

            # Rename
            pond_var.columns = ['subject_id', 'scan_id'] + vars_cur['measure'].to_list()
            # Load in the CIVET file (to get the date)
            civet_var = pd.read_csv(os.path.normpath(os.path.join(root_dir, 'POND','CIVET', 'pond-neuroanatomy20230428_withmissingdates_MV.csv')), sep=',')

            # Calculated the T1_QC
            civet_var['t1_qc'], _ = pond_map_qc(civet_var['QC_LITE'].to_numpy())

            # Only keep certain columns
            civet_var = civet_var[['scan', 'scan_date', 'scanned_on', 't1_qc']]

            # Rename
            civet_var = civet_var.rename(columns={'scan': 'scan_id', 'scan_date': 'DOS', 'scanned_on': 'scanner'})

            # Fix scanner
            civet_var.loc[~pd.isna(civet_var['scanner']), 'scanner'] = 'POND-' + civet_var.loc[~pd.isna(civet_var['scanner']), 'scanner']

            # Delete duplicate scan_ids with different dates: these can't be trusted
            dup = civet_var['scan_id'].value_counts()
            dup = dup[dup > 1]
            dup = dup.index
            for sub in dup:
                civet_var = civet_var.drop(civet_var[civet_var['scan_id'] == sub].index)
                pond_var = pond_var.drop(pond_var[pond_var['scan_id'] == sub].index)

            # Merge
            pond_var = pd.merge(pond_var, civet_var, how='left', on='scan_id')

            # Merge with DOB
            pond_dob = pond_df[['subject_id', 'DOB']]
            pond_dob = pond_dob.drop_duplicates()
            pond_var = pd.merge(pond_var, pond_dob, how='left', on='subject_id')

            # Calculate age
            pond_var['age'] = (pd.to_datetime(pond_var['DOS']) - pd.to_datetime(pond_var['DOB'])).dt.days / 365

            # Only keep the variable we need
            pond_var = pond_var[['subject_id', 'scan_id', 'age', 'scanner', 't1_qc'] + vars_cur['measure'].to_list()]

            # If scan_id (and thus age) doesn't exist, match on subject_id
            if 'scan_id' in pond_df.columns:

                # Delete the subject id column
                pond_var = pond_var[['scan_id'] + vars_cur['measure'].to_list()]

                # Merge
                pond_df = pond_df.reset_index().merge(pond_var, how='outer', on='scan_id').set_index('index').sort_index()

            else:

                # Merge
                pond_df = pd.merge(pond_df, pond_var, how='outer', on='subject_id')

    # To fix the missing ages, first check if another run exists that has an age
    missing_ages = pond_df.loc[pd.isna(pond_df['age']) & ~pd.isna(pond_df['scan_id'])]
    for ind, row in missing_ages.iterrows():
        session = row['scan_id'].split('_')[0] + '_' + row['scan_id'].split('_')[1]
        m = pond_df.loc[(~pd.isna(pond_df['scan_id'])) & (pond_df['scan_id'].str.contains(session)) & (pond_df['scan_id'] != row['scan_id']) & (~pd.isna(pond_df['age']))]
        if np.shape(m)[0] > 0:
            pond_df.loc[pond_df['scan_id'] == row['scan_id'], 'age'] = m['age'].to_list()[0]

    # Now need to do CBCL
    var = 'cbcl'

    # Extract the corresponding variables
    vars_cur = mapped_meas.loc[mapped_meas['type'] == var]

    # Load
    pond_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'POND', vars_cur['pond_path'].to_list()[0])), sep=',',
                           low_memory=False)

    # Rename the subject id column
    pond_var = pond_var.rename(columns={'subject': 'subject_id'})
    pond_var['subject_id'] = pond_var['subject_id'].astype('str')

    # Assign date of assessment
    pond_var['DOS'] = pond_var['CB68DATE2'].str.replace('T00:00:00Z', '')

    # Delete the columns with NaN as DOS
    pond_var = pond_var.loc[~pd.isna(pond_var['DOS'])]

    # Calculate age
    pond_var['age'] = (pd.to_datetime(pond_var['DOS']) - pd.to_datetime(pond_var['DOB'])).dt.days / 365

    # Only keep the columns we need
    pond_var = pond_var[['subject_id', 'age', 'DOS'] + vars_cur['pond'].to_list()]

    # Rename
    pond_var.columns = ['subject_id', 'age', 'DOS'] + vars_cur['measure'].to_list()

    # Delete duplicate rows
    pond_var = pond_var.drop_duplicates()

    # Separate into duplicates and singles
    pond_var_dup = pond_var[pond_var['subject_id'].duplicated(keep=False)]
    pond_var_sing = pd.concat([pond_var, pond_var_dup]).drop_duplicates(keep=False)

    # Assign the scan_id by merging
    pond_var_sing = pond_var_sing.merge(pond_df[['subject_id', 'scan_id']], how='left', on='subject_id')

    # Assign the missing scan_ids
    pond_var_sing.loc[pd.isna(pond_var_sing['scan_id']), 'scan_id'] = pond_var_sing.loc[pd.isna(pond_var_sing['scan_id']), 'subject_id'] +'_CBCL_' + pond_var_sing.loc[pd.isna(pond_var_sing['scan_id']), 'DOS']

    # Need to iterate over the duplicate subjects
    dup_subjs = np.unique(pond_var_dup['subject_id'])
    pond_var_dup_merge = pd.DataFrame(columns=pond_var_dup.columns.to_list() + ['scan_id'])
    for subj in dup_subjs:

        # Get the CBCL rows
        var_match = pond_var_dup.loc[pond_var_dup['subject_id'] == subj]

        # Initialze logical to determine if a row is selected
        var_match_select = [False for i in range(np.shape(var_match)[0])]

        # Find the matching
        df_match = pond_df.loc[(pond_df['subject_id'] == subj) & (~pd.isna(pond_df['scan_id']))]

        # Iterate over the matches in the actual data frame
        for ind, row in df_match.iterrows():

            # Find the difference between age
            diff = abs(row['age'] - var_match['age'])

            # Get the minimum
            diff_ind = np.argmin(diff)
            var_match_select[diff_ind] = True

            # Extract this row
            var_match_row = var_match.copy()
            var_match_row = var_match_row.iloc[[diff_ind]]

            # Add the scan ID so we match
            var_match_row['scan_id'] = row['scan_id']

            # Add to dataframe
            pond_var_dup_merge = pd.concat((pond_var_dup_merge, var_match_row))

        # Add the remaining non-selected CBCL rows
        if not all(var_match_select):

            # Extract the usnelected rows
            var_match_rows = var_match.copy()
            var_match_rows = var_match_rows.loc[np.array(var_match_select) == False]

            # Assign scan_id
            var_match_rows['scan_id'] = var_match_rows['subject_id'] + '_CBCL_' + var_match_rows['DOS']

            # Add to dataframe
            pond_var_dup_merge = pd.concat((pond_var_dup_merge, var_match_rows))

    # Merge the singles and duplicates
    pond_var = pd.concat((pond_var_sing, pond_var_dup_merge))

    # Drop age/DOS to prepare for merge
    pond_var = pond_var[['subject_id', 'scan_id', 'age'] + vars_cur['measure'].to_list()].sort_index()

    # First, merge the CBCL scans (no match for an MRI)
    pond_var_na = pond_var.loc[pond_var['scan_id'].str.contains('CBCL')]
    pond_df_na = pd.merge(pond_df.loc[pd.isna(pond_df['scan_id'])].drop(['scan_id', 'age'], axis=1), pond_var_na, how='outer', on='subject_id')

    # Now, merge the ones there was matches for
    pond_var_nna = pond_var.loc[~pond_var['scan_id'].str.contains('CBCL'), ['scan_id'] + vars_cur['measure'].to_list()]
    pond_df_nna = pd.merge(pond_df.loc[~pd.isna(pond_df['scan_id'])], pond_var_nna, how='outer', on='scan_id').drop_duplicates()

    # Concatenate the two
    pond_df = pd.concat((pond_df_na, pond_df_nna)).sort_values('subject_id').reset_index().drop(['index'], axis=1)

    # Fill in the demographics (DOB, sex, race, ethnicity, income, education) for the cases where there was an unselected row
    missing_demo = pond_df.copy()
    missing_demo = missing_demo.loc[pd.isna(pond_df['DOB'])]
    for ind, row in missing_demo.iterrows():
        # Find the matching
        df_match = pond_df.loc[(pond_df['subject_id'] == row['subject_id']) & (~pd.isna(pond_df['DOB']))]
        if np.shape(df_match)[0] > 0:
            df_match = df_match.iloc[[0]]
            pond_df.loc[[ind], ['DOB', 'sex', 'race', 'ethnicity', 'income', 'education']] = df_match[['DOB', 'sex', 'race', 'ethnicity', 'income', 'education']].to_numpy()

    # Drop DOB
    pond_df = pond_df.drop(['DOB'], axis=1)

    # Add dataset
    pond_df['dataset'] = 'POND'

    # Assign missing values
    pond_df.loc[pd.isna(pond_df['scanner']), 'scanner'] = 'POND-Unknown'
    pond_df.loc[pd.isna(pond_df['sex']), 'sex'] = 'Unknown'
    pond_df.loc[pond_df['sex'] == 'nan', 'sex'] = 'Unknown'
    pond_df.loc[pd.isna(pond_df['race']), 'race'] = missing_value_race
    pond_df.loc[pond_df['race'] == 'nan', 'race'] = missing_value_race
    pond_df.loc[pd.isna(pond_df['ethnicity']), 'ethnicity'] = missing_value_eth
    pond_df.loc[pond_df['ethnicity'] == 'nan', 'ethnicity'] = missing_value_eth

    # Get rid of rows that don't have either imaging
    pond_df = pond_df.loc[~pd.isna(pond_df['SubCortGrayVol'])]

    # Make sure we have the basics: age and sex
    pond_df = pond_df.loc[pond_df['sex'] != 'Unknwon']
    pond_df = pond_df.loc[~pd.isna(pond_df['age'])]

    return pond_df

# Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
# 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
def pond_map_race(input_array):

    # Copy input to output
    output_col = np.zeros((np.shape(input_array)[0]))
    output_col.fill(np.nan)

    # Set the value for missing
    missing_val = 'Cat-7'

    # Get rid of the "Jewish" (ind=7) and "Hispanic" (ind=9)
    input_array[~np.isnan(input_array[:, 0]), 7] = 0
    input_array[~np.isnan(input_array[:, 0]), 9] = 0

    # Re-assign
    output_col = np.where((input_array[:, 0] == 1) & (np.sum(input_array[:, np.r_[1:15]], axis=1) == 0), 'Cat-1', output_col)
    output_col = np.where((np.sum(input_array[:, np.r_[1,3:7,8,10:13]], axis=1) >= 1) & (np.sum(input_array[:, np.r_[0,2,13,14]], axis=1) == 0), 'Cat-2', output_col)
    output_col = np.where((input_array[:, 2] == 1) & (np.sum(input_array[:, np.r_[0:2, 3:15]], axis=1) == 0), 'Cat-3', output_col)
    output_col = np.where((input_array[:, 13] == 1) & (np.sum(input_array[:, np.r_[0:13, 14]], axis=1) == 0), 'Cat-4', output_col)
    output_col = np.where((input_array[:, 14] == 1) & (np.sum(input_array[:, np.r_[0:14]], axis=1) == 0), 'Cat-6', output_col)
    output_col = np.where((np.sum(input_array[:, 0:15], axis=1) > 1) & (output_col == 'nan'), 'Cat-5', output_col)
    output_col = np.where((np.sum(input_array[:, np.r_[0:7, 8, 10:15]], axis=1) == 0), 'Cat-7', output_col)

    # Return
    return output_col, missing_val

# Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
# not reported
def pond_map_ethnicity(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Cat-3'

    # Re-assign
    output_col = np.where(input_col == 1, 'Cat-1', output_col)
    output_col = np.where(input_col == 0, 'Cat-2', output_col)

    # Return
    return output_col, missing_val

# Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
# answer
def pond_map_income(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col = np.where((input_col >= 1) & (input_col <= 3), 1, output_col)
    output_col = np.where((input_col >= 4) & (input_col <= 7), 2, output_col)
    output_col = np.where(input_col == 8, 3, output_col)
    output_col = np.where((input_col > 10) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
# nan: Don't know/refuse to answer. Do this for each parent, and take the average
def pond_map_education(input_col1, input_col2):

    # Copy input to output
    output_col1 = input_col1.copy()
    output_col2 = input_col2.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col1 = np.where(input_col1 <= 13, 1, output_col1)
    output_col1 = np.where((input_col1  >= 14) & (input_col1 <= 16), 2, output_col1)
    output_col1 = np.where((input_col1  >= 17) & (input_col1 <= 19), 3, output_col1)
    output_col1 = np.where((input_col1  >= 20) & (input_col1 <= 22), 4, output_col1)
    output_col1 = np.where(input_col1 > 22, missing_val, output_col1)
    output_col2 = np.where(input_col2 <= 13, 1, output_col2)
    output_col2 = np.where((input_col2 >= 14) & (input_col2 <= 16), 2, output_col2)
    output_col2 = np.where((input_col2 >= 17) & (input_col2 <= 19), 3, output_col2)
    output_col2 = np.where((input_col2 >= 20) & (input_col2 <= 22), 4, output_col2)
    output_col2 = np.where(input_col2 > 22, missing_val, output_col2)

    # Take the mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_col = np.nanmean(np.concatenate((output_col1.reshape(-1, 1), output_col2.reshape(-1, 1)), axis=1), axis=1)

    # Return
    return output_col, missing_val

# Assign T1 QC to the following categories: 1: Passed (1 or 2), 2: Failed (0), 3: Unknown.
def pond_map_qc(input_col):

    # Copy input to output
    output_col = input_col.copy()
    output_col.fill(np.nan)

    # Set the value for missing
    missing_val = 'Unknown'

    # Re-assign
    output_col = np.where(input_col == True, 'Passed', output_col)
    output_col = np.where(input_col == False, 'Failed', output_col)
    output_col = np.where(np.isnan(input_col), missing_val, output_col)

    # Return
    return output_col, missing_val

def pond_choose(root_dir, pond_df):

    # Filter by age
    pond_df = pond_df.loc[(pond_df['age'] >= 6) & (pond_df['age'] < 19)]

    # Filter by passed QC
    pond_df = pond_df.loc[pond_df['t1_qc'] == 'Passed']

    # Get rid of scans where we only have 1 on a site
    site_counts = pond_df['scanner'].value_counts()
    site_counts = site_counts[site_counts < 2]
    for site in site_counts.index:
        pond_df = pond_df[pond_df['scanner'] != site]

    # Find outliers
    pond_df['is_outlier'] = 0
    count = 0
    for col in pond_df.columns:
        if (col.startswith('lh_') or col.startswith('rh_')) and col.endswith('_thick'):
            count = count + 1
            z = np.abs(stats.zscore(pond_df[col]))
            pond_df.loc[z > 3, 'is_outlier'] = pond_df.loc[z > 3, 'is_outlier'] + 1

    # Remove outliers
    pond_df = pond_df[pond_df['is_outlier']/count <= 0.50]

    # Load in the CIVET file (to get the date)
    civet_var = pd.read_csv(
        os.path.normpath(os.path.join(root_dir, 'POND','CIVET', 'pond-neuroanatomy20230428_withmissingdates_MV.csv')), sep=',')

    # Only keep certain columns
    civet_var = civet_var[['scan', 'best_of_subject']]

    # Rename
    civet_var = civet_var.rename(columns={'scan': 'scan_id'})

    # Delte NaNs
    civet_var = civet_var.loc[~pd.isna(civet_var['best_of_subject'])]

    # Merge
    pond_df = pd.merge(pond_df, civet_var, how='left', on='scan_id')

    # Initialize new df
    pond_df_new = pd.DataFrame(columns=pond_df.columns)

    # Iterate over the unique subjects
    subj_u = np.unique(pond_df['subject_id'])
    for subj in subj_u:

        # Extract the rows
        subj_df = pond_df.loc[pond_df['subject_id'] == subj]

        # If only one, we're good to go
        if np.shape(subj_df)[0] == 1:

            # Assign
            pond_df_new = pd.concat((pond_df_new, subj_df))

        else:

            # If there's one best of subject, use that one
            subj_df_new = subj_df.copy()
            subj_df_new = subj_df_new.loc[subj_df['best_of_subject'] == True]
            if np.shape(subj_df_new)[0] == 1:
                pond_df_new = pd.concat((pond_df_new, subj_df_new))
            else:

                # If everything is NaN, choose the last run
                if all(pd.isna(subj_df['best_of_subject'])):
                    pond_df_new = pd.concat((pond_df_new, subj_df.iloc[[0]]))

    # Delete best of subject
    pond_df_new = pond_df_new.drop(['best_of_subject'], axis=1)

    return pond_df_new

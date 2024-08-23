import pandas as pd
import os
import numpy as np
import collections
import warnings
from scipy import stats

# Main function for loading in the HBN data
def hbn_load(root_dir, measure_file):

    # Mapped measures
    mapped_meas = pd.read_excel(measure_file)

    # Initialize HCP dataframe with: subject_id, age, sex, race/ethnicity, income, and education
    hbn_df = pd.DataFrame()

    # Load in the demographics file
    hbn_basic_demo = pd.read_csv(os.path.join(root_dir,'HBN', r'9994_Basic_Demos_20220728.csv'), sep=',', skiprows=[1])

    # Delete rows of all NaNs (last row)
    hbn_basic_demo = hbn_basic_demo.dropna(how='all')

    # Get rid of spaces in EID column
    hbn_basic_demo['EID'] = hbn_basic_demo['EID'].str.replace(' ', '')

    # Only keep certain columns
    hbn_basic_demo = hbn_basic_demo[['EID', 'Age', 'Sex', 'Participant_Status']]

    # Drop duplicates
    hbn_basic_demo = hbn_basic_demo.drop_duplicates()

    # In case of multiple visits (currently only 1), take the "Complete"
    dup = [item for item, count in collections.Counter(hbn_basic_demo['EID']).items() if count > 1]
    for sub in dup:
        hbn_basic_demo = hbn_basic_demo.drop(hbn_basic_demo[(hbn_basic_demo['EID'] == sub) & ~(hbn_basic_demo['Participant_Status'].str.contains('Complete', na=False))].index)

    # Assign the easy demogaphics
    hbn_df[['subject_id', 'age', 'sex']] = hbn_basic_demo[['EID', 'Age', 'Sex']]

    # Remape sex
    hbn_df.loc[hbn_df['sex'] == 0, 'sex'] = 'M'
    hbn_df.loc[hbn_df['sex'] == 1, 'sex'] = 'F'

    # Load in the family demographics
    hbn_demo = pd.read_csv(os.path.join(root_dir,'HBN', r'9994_PreInt_Demos_Fam_20220728.csv'), sep=',', skiprows=[1])

    # Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
    # 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
    hbn_demo['race'], missing_value_race = hbn_map_race(hbn_demo['Child_Race'].to_numpy())

    # Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
    # not reported
    hbn_demo['ethnicity'], missing_value_eth = hbn_map_ethnicity(hbn_demo['Child_Ethnicity'].to_numpy())

    # Rename the subject id column
    hbn_demo = hbn_demo.rename(columns={'EID': 'subject_id'})

    # Only keep certain columns
    hbn_demo = hbn_demo[['subject_id', 'race', 'ethnicity']]

    # Get rid of spaces in EID column
    hbn_demo['subject_id'] = hbn_demo['subject_id'].str.replace(' ', '')

    # Delete duplicate rows
    hbn_demo = hbn_demo.drop_duplicates()

    # Merge
    hbn_df = pd.merge(hbn_df, hbn_demo, how='outer', on='subject_id')

    # Load in the .csv containing the income information
    hbn_inc = pd.read_csv(os.path.join(root_dir,'HBN', r'9994_FSQ_20220728.csv'), sep=',', skiprows=[1])

    # Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
    # answer
    hbn_inc['income'], _ = hbn_map_income(hbn_inc['FSQ_04'].to_numpy())

    # Rename the subject id column
    hbn_inc = hbn_inc.rename(columns={'EID': 'subject_id'})

    # Get rid of spaces in EID column
    hbn_inc['subject_id'] = hbn_inc['subject_id'].str.replace(' ', '')

    # Only keep certain columns
    hbn_inc = hbn_inc[['subject_id', 'income']]

    # Delete duplicate rows
    hbn_inc = hbn_inc.drop_duplicates()

    # Merge
    hbn_df = pd.merge(hbn_df, hbn_inc, how='outer', on='subject_id')

    # Load in the .csv containing the education information
    hbn_edu = pd.read_csv(os.path.join(root_dir,'HBN', r'9994_Barratt_20220728.csv'), sep=',', skiprows=[1])

    # Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
    # nan: Don't know/refuse to answer. Do this for each parent.
    hbn_edu['education'], _ = hbn_map_education(hbn_edu['Barratt_P1_Edu'].to_numpy(), hbn_edu['Barratt_P2_Edu'].to_numpy())

    # Rename the subject id column
    hbn_edu = hbn_edu.rename(columns={'EID': 'subject_id'})

    # Get rid of spaces in EID column
    hbn_edu['subject_id'] = hbn_edu['subject_id'].str.replace(' ', '')

    # Only keep certain columns
    hbn_edu = hbn_edu[['subject_id', 'education']]

    # Delete duplicate rows
    hbn_edu = hbn_edu.drop_duplicates()

    # Merge
    hbn_df = pd.merge(hbn_df, hbn_edu, how='outer', on='subject_id')

    # Load in the .csv containing the MRI scanner information
    hbn_scanner = pd.read_csv(os.path.join(root_dir,'HBN', r'9994_MRI_Track_20220728_mv.csv'), sep=',', skiprows=[1])

    # Assign scanner to the data
    hbn_scanner['scanner'], _ = hbn_map_scanner(hbn_scanner['Scan_Location'].to_numpy())

    # Rename the subject id column
    hbn_scanner = hbn_scanner.rename(columns={'EID': 'subject_id'})

    # Get rid of spaces in EID column
    hbn_scanner['subject_id'] = hbn_scanner['subject_id'].str.replace(' ', '')

    # Only keep certain columns
    hbn_scanner = hbn_scanner[['subject_id', 'scanner']]

    # Delete duplicate rows
    hbn_scanner = hbn_scanner.drop_duplicates()

    # Merge
    hbn_df = pd.merge(hbn_df, hbn_scanner, how='outer', on='subject_id')

    # Get the unique types of the mapped variables
    var_u, ind = np.unique(mapped_meas['type'], return_index=True)
    var_u = var_u[np.argsort(ind)]

    # Iterate over the types
    for var in var_u:

        # Extract the corresponding variables
        vars_cur = mapped_meas.loc[mapped_meas['type'] == var]

        # Load in the variable file
        if var == 'cbcl':
            hbn_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'HBN', vars_cur['hbn_path'].to_list()[0])), sep=',', skiprows=[1])
        else:
            hbn_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'HBN', vars_cur['hbn_path'].to_list()[0])), sep='\t')

        # If CBCL, easy
        if var == 'cbcl':

            # Assign subject ID
            hbn_var['subject_id'] = hbn_var[vars_cur['hbn_subid'].to_list()[0]]

            # Get rid of spaces in EID column
            hbn_var['subject_id'] = hbn_var['subject_id'].str.replace(' ', '')

            # Only keep the columns we need
            hbn_var = hbn_var[['subject_id'] + vars_cur['hbn'].to_list()]

            # Rename
            hbn_var.columns = ['subject_id'] + vars_cur['measure'].to_list()

            # Merge
            hbn_df = pd.merge(hbn_df, hbn_var, how='outer', on='subject_id')

        else:

            # Assign subject ID
            hbn_var['subject_id'] = [x.split('_')[0].split('-')[1] for x in hbn_var[vars_cur['hbn_subid'].to_list()[0]]]

            # Get scan ID
            hbn_var['scan_id'] = hbn_var[vars_cur['hbn_subid'].to_list()[0]].str.replace('_ses-HBN', '').str.replace('siteCBIC', '').str.replace('siteRU', '').str.replace('siteCUNY', '').str.replace('siteSI', '')

            # Save the scanner
            hbn_scanner = hbn_var.copy()
            hbn_scanner = hbn_scanner[['scan_id'] + [vars_cur['hbn_subid'].to_list()[0]]]
            scanner_fix = hbn_scanner[vars_cur['hbn_subid'].to_list()[0]].str.split('_', expand=True)[1].str.replace('ses-', '').str.replace('site', '-').str.replace('acq-HCP', 'nan')
            hbn_scanner['scanner_fix'] = scanner_fix
            hbn_scanner.loc[hbn_scanner['scanner_fix'] == 'nan'] = pd.NA
            hbn_scanner = hbn_scanner.dropna()
            hbn_scanner = hbn_scanner[['scan_id', 'scanner_fix']]

            # Only keep the columns we need
            hbn_var = hbn_var[['subject_id', 'scan_id'] + vars_cur['hbn'].to_list()]

            # Rename
            hbn_var.columns = ['subject_id', 'scan_id'] + vars_cur['measure'].to_list()

            # If scan_id doesn't exist, match on subject_id
            if 'scan_id' in hbn_df.columns:

                # Delete the subject id column
                hbn_var = hbn_var[['scan_id'] + vars_cur['measure'].to_list()]

                # Merge
                hbn_df = pd.merge(hbn_df, hbn_var, how='outer', on='scan_id')

            else:

                # Merge
                hbn_df = pd.merge(hbn_df, hbn_var, how='outer', on='subject_id')

                # Try to fix scanner
                hbn_df = pd.merge(hbn_df, hbn_scanner, how='left', on='scan_id')
                hbn_df.loc[pd.isna(hbn_df['scanner']) & ~pd.isna(hbn_df['scanner_fix']), 'scanner'] = hbn_df.loc[pd.isna(hbn_df['scanner']) & ~pd.isna(hbn_df['scanner_fix']), 'scanner_fix']

                # Drop column
                hbn_df = hbn_df.drop(['scanner_fix'], axis=1)

    # Load in the .xlsx containing the T1 QC
    hbn_qc = pd.read_csv(os.path.join(root_dir,'HBN', r'T1scans_lerch_extrasMV.csv'))

    # Assign T1 QC to the following categories: 1: Passed (1 or 2), 2: Failed (0), 3: Unknown.
    hbn_qc['t1_qc'], _ = hbn_map_qc(hbn_qc['qc_1'].to_numpy())

    # Get rid of the QC in dir
    hbn_qc['dir'] = hbn_qc['dir'].str.replace('_QC', '')

    # Rename the side id an
    hbn_qc = hbn_qc.rename(columns={'dir': 'scan_id'})

    # Only keep certain columns
    hbn_qc = hbn_qc[['scan_id', 't1_qc']]

    # Delete duplicate rows
    hbn_qc = hbn_qc.drop_duplicates()

    # Merge - this time using a left merge, since we don't need the QC for those we don't have data for
    hbn_df = pd.merge(hbn_df, hbn_qc, how='left', on='scan_id')

    # Add dataset
    hbn_df['dataset'] = 'HBN'

    # Assign missing values
    hbn_df.loc[pd.isna(hbn_df['scanner']), 'scanner'] = 'HBN-Unknown'
    hbn_df.loc[pd.isna(hbn_df['sex']), 'sex'] = 'Unknown'
    hbn_df.loc[pd.isna(hbn_df['race']), 'race'] = missing_value_race
    hbn_df.loc[pd.isna(hbn_df['ethnicity']), 'ethnicity'] = missing_value_eth

    # Get rid of rows that don't have either CBCL or imaging
    hbn_df = hbn_df.loc[~pd.isna(hbn_df['scan_id'])]

    # Make sure we have the basics: age and sex
    hbn_df = hbn_df.loc[hbn_df['sex'] != 'Unknwon']
    hbn_df = hbn_df.loc[~pd.isna(hbn_df['age'])]


    # Drop any duplicates
    hbn_df = hbn_df.drop_duplicates()

    return hbn_df


# Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
# 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
def hbn_map_race(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Cat-7'

    # Re-assign
    output_col = np.where((input_col == 5) | (input_col == 6), 'Cat-1', output_col)
    output_col = np.where((input_col == 3) | (input_col == 4), 'Cat-2', output_col)
    output_col = np.where(input_col == 1, 'Cat-3', output_col)
    output_col = np.where(input_col == 0, 'Cat-4', output_col)
    output_col = np.where(input_col == 8, 'Cat-5', output_col)
    output_col = np.where(input_col == 9, 'Cat-6', output_col)
    output_col = np.where(input_col == 7, 'Cat-6', output_col)
    output_col = np.where((input_col == 2) | (input_col >= 10) | np.isnan(input_col), missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
# not reported
def hbn_map_ethnicity(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Cat-3'

    # Re-assign
    output_col = np.where(input_col == 1, 'Cat-1', output_col)
    output_col = np.where(input_col == 0, 'Cat-2', output_col)
    output_col = np.where((input_col >= 2) | (np.isnan(input_col)), 'Cat-3', output_col)

    # Return
    return output_col, missing_val

# Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
# answer
def hbn_map_income(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col = np.where(input_col <= 4, 1, output_col)
    output_col = np.where((input_col >= 5) & (input_col < 11), 2, output_col)
    output_col = np.where(input_col == 11, 3, output_col)
    output_col = np.where((input_col == 12) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
# nan: Don't know/refuse to answer. Do this for each parent, and take the average
def hbn_map_education(input_col1, input_col2):

    # Copy input to output
    output_col1 = input_col1.copy()
    output_col2 = input_col2.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col1 = np.where(input_col1 <= 9, 1, output_col1)
    output_col1 = np.where((input_col1  >= 12) & (input_col1 <= 15), 2, output_col1)
    output_col1 = np.where(input_col1  == 18, 3, output_col1)
    output_col1 = np.where(input_col1 == 21, 4, output_col1)
    output_col1 = np.where(np.isnan(input_col1), missing_val, output_col1)
    output_col2 = np.where(input_col2 <= 9, 1, output_col2)
    output_col2 = np.where((input_col2 >= 12) & (input_col2 <= 15), 2, output_col2)
    output_col2 = np.where(input_col2 == 18, 3, output_col2)
    output_col2 = np.where(input_col2 == 21, 4, output_col2)
    output_col2 = np.where(np.isnan(input_col2), missing_val, output_col2)

    # Take the mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_col = np.nanmean(np.concatenate((output_col1.reshape(-1, 1), output_col2.reshape(-1, 1)), axis=1), axis=1)

    # Return
    return output_col, missing_val

# Assign scanner to the following categories: HBN-SI, HBN-RU, HBN-CBIC, HBN-CUNY, HBN-Unknown
def hbn_map_scanner(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'HBN-Unknown'

    # Re-assign
    output_col = np.where(input_col == 1, 'HBN-SI', output_col)
    output_col = np.where(input_col == 2, 'HBN-RU', output_col)
    output_col = np.where(input_col == 3, 'HBN-CBIC', output_col)
    output_col = np.where(input_col == 4, 'HBN-CUNY', output_col)
    output_col = np.where((input_col == 0) | (input_col >= 5) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign T1 QC to the following categories: 1: Passed (1 or 2), 2: Failed (0), 3: Unknown.
def hbn_map_qc(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Unknown'

    # Re-assign
    output_col = np.where(input_col == 1, 'Passed', output_col)
    output_col = np.where(input_col == 2, 'Passed', output_col)
    output_col = np.where(input_col == 0, 'Failed', output_col)
    output_col = np.where((input_col < 0) | (input_col > 2) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val

def hbn_choose(root_dir, hbn_df):

    # Filter by age
    hbn_df = hbn_df.loc[(hbn_df['age'] >= 6) & (hbn_df['age'] < 19)]

    # Filter by passed QC
    hbn_df = hbn_df.loc[hbn_df['t1_qc'] == 'Passed']

    # Find outliers
    hbn_df['is_outlier'] = 0
    count = 0
    for col in hbn_df.columns:
        if (col.startswith('lh_') or col.startswith('rh_')) and col.endswith('_thick'):
            count = count + 1
            z = np.abs(stats.zscore(hbn_df[col]))
            hbn_df.loc[z > 3, 'is_outlier'] = hbn_df.loc[z > 3, 'is_outlier'] + 1

    # Remove outliers
    hbn_df = hbn_df[hbn_df['is_outlier'] / count <= 0.50]

    # Load in the CIVET file (to get the date)
    civet_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'HBN', 'CIVET', 'hbn-neuroanatomy20220908.csv')), sep=',')

    # Only keep certain columns
    civet_var = civet_var[['scan', 'best_of_subject','Dx']]

    # Rename
    civet_var = civet_var.rename(columns={'scan': 'scan_id'})

    # Delte NaNs
    civet_var = civet_var.loc[~pd.isna(civet_var['best_of_subject'])]

    # Merge
    hbn_df = pd.merge(hbn_df, civet_var, how='left', on='scan_id')

    # Initialize new df
    hbn_df_new = pd.DataFrame(columns=hbn_df.columns)

    # Iterate over the unique subjects
    subj_u = np.unique(hbn_df['subject_id'])
    for subj in subj_u:

        # Extract the rows
        subj_df = hbn_df.loc[hbn_df['subject_id'] == subj]

        # If only one, we're good to go
        if np.shape(subj_df)[0] == 1:

            # Assign
            hbn_df_new = pd.concat((hbn_df_new, subj_df))

        else:

            # If there's one best of subject, use that one
            subj_df_new = subj_df.copy()
            subj_df_new = subj_df_new.loc[subj_df['best_of_subject'] == True]
            if np.shape(subj_df_new)[0] == 1:
                hbn_df_new = pd.concat((hbn_df_new, subj_df_new))
            else:

                # If everything is NaN, choose the last run
                if all(pd.isna(subj_df['best_of_subject'])):
                    hbn_df_new = pd.concat((hbn_df_new, subj_df.iloc[[0]]))

    # Delete best of subject
    hbn_df_new = hbn_df_new.drop(['best_of_subject'], axis=1)

    return hbn_df_new
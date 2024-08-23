import pandas as pd
import os
import numpy as np
import warnings
from scipy import stats

# Main function for loading in the HCP-D data
def hcpd_load(root_dir, measure_file):

    # Mapped measures
    mapped_meas = pd.read_excel(measure_file)

    # Initialize HCP dataframe with: subject_id, age, sex, race/ethnicity, income, and education
    hcpd_df = pd.DataFrame()

    # Load in the demographics file
    hcpd_demo = pd.read_csv(os.path.join(root_dir, r'ndar_subject01.txt'), sep='\t', skiprows=[1])

    # Only keep certain columns
    hcpd_demo = hcpd_demo[['src_subject_id', 'interview_age', 'sex', 'site', 'race', 'ethnic_group']]

    # Drop duplicates
    hcpd_demo = hcpd_demo.drop_duplicates()

    # Delete rows of all NaNs (last row)
    hcpd_demo = hcpd_demo.dropna(how='all')

    # Assign the easy demogaphics
    hcpd_df[['subject_id', 'age', 'sex']] = hcpd_demo[['src_subject_id', 'interview_age', 'sex']]

    # Convert age to years (currently in months)
    hcpd_df['age'] = hcpd_df['age'] / 12

    # Assign scanner by appending "HCPD" to the scanner names
    hcpd_df['scanner'] = 'HCPD-' + hcpd_demo['site']

    # Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
    # 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
    hcpd_df['race'], missing_val = hcpd_map_race(hcpd_demo['race'].to_numpy())
    hcpd_df.loc[pd.isna(hcpd_df['race']), 'race'] = missing_val

    # Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
    # not reported
    hcpd_df['ethnicity'], missing_val = hcpd_map_ethnicity(hcpd_demo['ethnic_group'].to_numpy())
    hcpd_df.loc[pd.isna(hcpd_df['ethnicity']), 'ethnicity'] = missing_val

    # Load in the .csv containing SES information
    hcpd_ses = pd.read_csv(os.path.join(root_dir, r'socdem01.txt'), sep='\t', skiprows=[1])

    # Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
    # answer
    hcpd_ses['income'], _ = hcpd_map_income(hcpd_ses['annual_fam_inc'].to_numpy())

    # Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
    # nan: Don't know/refuse to answer. Do this for each parent.
    hcpd_ses['education'], _ = hcpd_map_education(hcpd_ses['mother_edu_cat'].to_numpy(), hcpd_ses['father_edu_cat'].to_numpy())

    # Rename the subject id column
    hcpd_ses = hcpd_ses.rename(columns={'src_subject_id': 'subject_id'})

    # Only keep the columns we need
    hcpd_ses = hcpd_ses[['subject_id', 'income', 'education']]

    # Merge
    hcpd_df = pd.merge(hcpd_df, hcpd_ses, how='outer', on='subject_id')

    # Get the unique types of the mapped variables
    var_u, ind = np.unique(mapped_meas['type'], return_index=True)
    var_u = var_u[np.argsort(ind)]

    # Iterate over the types
    for var in var_u:

        # Extract the corresponding variables
        vars_cur = mapped_meas.loc[mapped_meas['type'] == var]

        # Load in the variable file
        if var == 'cbcl':
            hcpd_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'HC', vars_cur['hcpd_path'].to_list()[0])), sep='\t', skiprows=[1])
        else:
            hcpd_var = pd.read_csv(os.path.normpath(os.path.join(root_dir,'HC', vars_cur['hcpd_path'].to_list()[0])), sep='\t')

        # Remove the suffix for the MRI subject ID measure
        hcpd_var['subject_id'] = hcpd_var[vars_cur['hcpd_subid'].to_list()[0]].replace('_V1_MR', '', regex=True)

        # Only keep the columns we need
        hcpd_var = hcpd_var[['subject_id'] + vars_cur['hcpd'].to_list()]

        # Rename
        hcpd_var.columns = ['subject_id'] + vars_cur['measure'].to_list()

        # Merge
        hcpd_df = pd.merge(hcpd_df, hcpd_var, how='outer', on='subject_id')

    # Only one MRI for each subject for HCP-D, so assign scan_id as the same as the subject
    hcpd_df['scan_id'] = hcpd_demo['src_subject_id']

    # All pass QC for HCP-D
    hcpd_df['t1_qc'] = 'Passed'

    # Add dataset
    hcpd_df['dataset'] = 'HCPD'

    # Make sure it has both
    hcpd_df = hcpd_df.loc[~pd.isna(hcpd_df['scan_id'])]

    # Make sure we have the basics: age and sex
    hcpd_df = hcpd_df.loc[hcpd_df['sex'] != 'Unknwon']
    hcpd_df = hcpd_df.loc[~pd.isna(hcpd_df['age'])]

    return hcpd_df

# Assign race to the following categories: 1: Native American Indian/American Indian/Alaska Native, 2: Asian,
# 3: Black/African American, 4: White, 5: More than one race, 6: Other race, 7: Unknown or not reported
def hcpd_map_race(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Cat-7'

    # Re-assign
    output_col = np.where(input_col == 'American Indian/Alaska Native', 'Cat-1', output_col)
    output_col = np.where(input_col == 'Asian', 'Cat-2', output_col)
    output_col = np.where(input_col == 'Black or African American', 'Cat-3', output_col)
    output_col = np.where(input_col == 'White', 'Cat-4', output_col)
    output_col = np.where(input_col == 'More than one race', 'Cat-5', output_col)
    output_col = np.where(input_col == 'Hawaiian or Pacific Islander', 'Cat-6', output_col)
    output_col = np.where(input_col == 'Unknown or not reported', missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign ethnicity to the following categories: 1: Hispanic/Latino/Latina, 2: Not Hispanic/Latino/Latina, 3: Unknown or
# not reported
def hcpd_map_ethnicity(input_col):

    # Copy input to output
    output_col = input_col.copy()

    # Set the value for missing
    missing_val = 'Cat-3'

    # Re-assign
    output_col = np.where(input_col == 'Hispanic or Latino', 'Cat-1', output_col)
    output_col = np.where(input_col == 'Not Hispanic or Latino', 'Cat-2', output_col)
    output_col = np.where(input_col == 'Unknown or not reported', 'Cat-3', output_col)

    # Return
    return output_col, missing_val


# Assign family income to the following (rough) categories: 1: <$50k, 2: $50k-$199k, 3: >=$200k, nan: Don't know/refuce to
# answer
def hcpd_map_income(input_col):

    # Copy input to output
    output_col = input_col.copy()
    output_col.fill(np.nan)

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col = np.where((input_col < 50000) & (input_col > 0), 1, output_col)
    output_col = np.where((input_col >= 50000) & (input_col < 200000), 2, output_col)
    output_col = np.where(input_col >= 200000, 3, output_col)
    output_col = np.where((input_col <= 0) | (np.isnan(input_col)), missing_val, output_col)

    # Return
    return output_col, missing_val

# Assign education to the following categories: 1: <High school, 2: High school, 3: Undergraduate-like, 4: Graduate,
# nan: Don't know/refuse to answer. Do this for each parent, and take the average
def hcpd_map_education(input_col1, input_col2):

    # Copy input to output
    output_col1 = input_col1.copy()
    output_col2 = input_col2.copy()

    # Set the value for missing
    missing_val = np.nan

    # Re-assign
    output_col1 = np.where(input_col1 <= 12, 1, output_col1)
    output_col1 = np.where((input_col1  >= 13) & (input_col1 <= 16), 2, output_col1)
    output_col1 = np.where((input_col1  >= 17) & (input_col1 <= 21), 3, output_col1)
    output_col1 = np.where((input_col1  >= 22) & (input_col1 <= 25), 4, output_col1)
    output_col1 = np.where(input_col1 >= 26, missing_val, output_col1)
    output_col2 = np.where(input_col2 <= 12, 1, output_col2)
    output_col2 = np.where((input_col2 >= 13) & (input_col2 <= 16), 2, output_col2)
    output_col2 = np.where((input_col2 >= 17) & (input_col2 <= 21), 3, output_col2)
    output_col2 = np.where((input_col2 >= 22) & (input_col2 <= 25), 4, output_col2)
    output_col2 = np.where(input_col2 >= 26, missing_val, output_col2)

    # Take the mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_col = np.nanmean(np.concatenate((output_col1.reshape(-1, 1), output_col2.reshape(-1, 1)), axis=1), axis=1)

    # Return
    return output_col, missing_val

def hcpd_choose(hcpd_df):
    
    # Filter by age
    hcpd_df = hcpd_df.loc[(hcpd_df['age'] >= 6) & (hcpd_df['age'] < 19)]

    # Filter by passed QC
    hcpd_df = hcpd_df.loc[hcpd_df['t1_qc'] == 'Passed']

    # Find outliers
    hcpd_df['is_outlier'] = 0
    count = 0
    for col in hcpd_df.columns:
        if (col.startswith('lh_') or col.startswith('rh_')) and col.endswith('_thick'):
            count = count + 1
            z = np.abs(stats.zscore(hcpd_df[col]))
            hcpd_df.loc[z > 3, 'is_outlier'] = hcpd_df.loc[z > 3, 'is_outlier'] + 1

    # Remove outliers
    hcpd_df = hcpd_df[hcpd_df['is_outlier'] / count <= 0.50]
    
    hcpd_df = hcpd_df.drop_duplicates()

    return hcpd_df
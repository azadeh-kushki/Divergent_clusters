import os
os.chdir("E:\Step0-prepare-data")
from hbn_functions import hbn_load, hbn_choose
from abcd_functions import abcd_load, abcd_choose
from pond_functions import pond_load, pond_choose
from hcpd_functions import hcpd_load, hcpd_choose
import pandas as pd

measure_file = r'mapped_measures.xlsx'
# Load the ABCD data
abcd_df = abcd_load(os.path.normpath(r'Data'), measure_file)
abcd_df = abcd_choose(abcd_df)
# Load the HBN data
hbn_df = hbn_load(os.path.normpath(r'Data'), measure_file)
hbn_df = hbn_choose(r'Data', hbn_df)
# Load the HCP-D data
hcpd_df = hcpd_load(os.path.normpath(r'Data'), measure_file)
hcpd_df = hcpd_choose(hcpd_df)
hcpd_df['dataset'] = 'ABCD'
# Load the POND data
pond_df = pond_load(os.path.normpath(r'Data'), measure_file)
pond_df = pond_choose(r'Data', pond_df)
# List of acceptable Dx values
dx_values = ['ASD', 'ADHD', 'Typically Developing', 'OCD','Sub-threshold ADHD','Tourette Syndrome','Sub-threshold OCD','Intellectual Disability only']
# Filter df_pond to keep only rows where Dx is in dx_values
pond_df = pond_df[pond_df['Dx'].isin(dx_values)]
# List of Dx values to be changed to "Other"
change_to_other = ['Sub-threshold ADHD', 'Tourette Syndrome', 'Sub-threshold OCD', 'Intellectual Disability only']
# Update 'Dx' column
pond_df['Dx'] = pond_df['Dx'].apply(lambda x: "Other" if x in change_to_other else x)

# Concatenate all the data
df = pd.concat((abcd_df, hcpd_df,hbn_df, pond_df))

df.to_csv("alldata-processed.csv",index=False)
pond_df.to_csv("pond-processed.csv",index=False)
hbn_df.to_csv("hbn-processed.csv",index=False)
abcd_df.to_csv("abcd_df-processed.csv",index=False)
hcpd_df.to_csv("hcpd_df-processed.csv",index=False)


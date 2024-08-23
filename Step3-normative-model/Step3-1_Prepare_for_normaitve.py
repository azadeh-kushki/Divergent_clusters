import pandas as pd
import numpy as np
from scipy.stats import norm
import os 
os.chdir("E:/Step3-normative-model")
   
def split_regions(df):
    # Select columns based on specific keywords
    thickness_cols= df.columns[df.columns.get_loc('lh_bankssts_thick_combatted'):df.columns.get_loc('rh_insula_thick_combatted') + 1 ]

    area_cols  = df.columns[df.columns.get_loc('lh_bankssts_area_combatted'):df.columns.get_loc('rh_insula_area_combatted') + 1 ]

    volume_cols = df.columns[df.columns.get_loc('lh_bankssts_vol_combatted'):df.columns.get_loc('rh_insula_vol_combatted') + 1 ]


    # Identify columns that don't fall into the previous categories
    other_cols = df.columns[df.columns.get_loc('Left-Thalamus-Proper_combatted'):df.columns.get_loc('Right-VentralDC_combatted') + 1 ]


    # Create dataframes for each category
    brain_area = df[area_cols]
    brain_thickness = df[thickness_cols]
    brain_volume = df[volume_cols]
    brain_subcort = df[other_cols]

    return brain_area, brain_thickness, brain_volume, brain_subcort

def normalize_data(df):   
   
    [N, M] = df.shape
    z = df.copy() * 0
    for i in range(0, M):
        z.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean()) / (df.iloc[:, i].std())
    return (z)


df_pond_male = pd.read_csv('E:/Step2-correct-data/zresult_pond_male.csv')
df_pond_female = pd.read_csv('E:/Step2-correct-data/zresult_pond_female.csv')
df_hbn_male = pd.read_csv('E:/Step2-correct-data/zresult_hbn_male.csv')
df_hbn_female = pd.read_csv('E:/Step2-correct-data/zresult_hbn_female.csv')
df_abcd_male = pd.read_csv('E:/Step2-correct-data/zresult_abcd_male.csv')
df_abcd_female = pd.read_csv('E:/Step2-correct-data/zresult_abcd_female.csv')

df_pond_male_brain =  df_pond_male[df_pond_male.columns[df_pond_male.columns.str.contains('combatted')]]
df_pond_female_brain =  df_pond_female[df_pond_female.columns[df_pond_female.columns.str.contains('combatted')]]
df_hbn_male_brain =  df_hbn_male[df_hbn_male.columns[df_hbn_male.columns.str.contains('combatted')]]
df_hbn_female_brain =  df_hbn_female[df_hbn_female.columns[df_hbn_female.columns.str.contains('combatted')]]
df_abcd_male_brain =  df_abcd_male[df_abcd_male.columns[df_abcd_male.columns.str.contains('combatted')]]
df_abcd_female_brain =  df_abcd_female[df_abcd_female.columns[df_abcd_male.columns.str.contains('combatted')]]

df_pond_male_brain = df_pond_male_brain.dropna()
df_pond_female_brain = df_pond_female_brain.dropna()
df_hbn_male_brain = df_hbn_male_brain.dropna()
df_hbn_female_brain = df_hbn_female_brain.dropna()
df_abcd_male_brain = df_abcd_male_brain.dropna()
df_abcd_female_brain = df_abcd_female_brain.dropna()


df_pond_male = df_pond_male.loc[df_pond_male_brain.index]
df_pond_female = df_pond_female.loc[df_pond_female_brain.index]

df_hbn_male = df_hbn_male.loc[df_hbn_male_brain.index]
df_hbn_female = df_hbn_female.loc[df_hbn_female_brain.index]


df_abcd_male = df_abcd_male.loc[df_abcd_male_brain.index]
df_abcd_female = df_abcd_female.loc[df_abcd_female_brain.index]

df1_area,  df1_thickness, df1_volume, df1_subcort = split_regions(df_pond_male_brain)
df1_area_fe,  df1_thickness_fe, df1_volume_fe, df1_subcort_fe = split_regions(df_pond_female_brain)

df2_area, df2_thickness,  df2_volume, df2_subcort = split_regions(df_hbn_male_brain)
df2_area_fe, df2_thickness_fe,  df2_volume_fe, df2_subcort_fe = split_regions(df_hbn_female_brain)

df3_area, df3_thickness,  df3_volume, df3_subcort = split_regions(df_abcd_male_brain)
df3_area_fe, df3_thickness_fe,  df3_volume_fe, df3_subcort_fe = split_regions(df_abcd_female_brain)


##################### normalize ###############################

df1_thicknessx = normalize_data(df1_thickness)
df1_areax = normalize_data(df1_area)
df1_volumex = normalize_data(df1_volume)
df1_subcortx = normalize_data(df1_subcort)


df1_thicknessx_fe = normalize_data(df1_thickness_fe)
df1_areax_fe = normalize_data(df1_area_fe)
df1_volumex_fe = normalize_data(df1_volume_fe)
df1_subcortx_fe = normalize_data(df1_subcort_fe)

df2_thicknessx = normalize_data(df2_thickness)
df2_areax = normalize_data(df2_area)
df2_volumex = normalize_data(df2_volume)
df2_subcortx = normalize_data(df2_subcort)

df2_thicknessx_fe = normalize_data(df2_thickness_fe)
df2_areax_fe = normalize_data(df2_area_fe)
df2_volumex_fe = normalize_data(df2_volume_fe)
df2_subcortx_fe = normalize_data(df2_subcort_fe)


df3_thicknessx = normalize_data(df3_thickness)
df3_areax = normalize_data(df3_area)
df3_volumex = normalize_data(df3_volume)
df3_subcortx = normalize_data(df3_subcort)


df3_thicknessx_fe = normalize_data(df3_thickness_fe)
df3_areax_fe = normalize_data(df3_area_fe)
df3_volumex_fe = normalize_data(df3_volume_fe)
df3_subcortx_fe = normalize_data(df3_subcort_fe)


# perpare covariate_normsample for sex and age
def store_covariate(data,column,name):
    covariate_normsample = data[column]
    namex = 'covariate_normsample_'+name+'.txt'
    covariate_normsample.to_csv(namex,
                            sep = ' ',
                            header = False,
                            index = False)

df_abcd_male['age_x']=df_abcd_male.age
df_abcd_female['age_x'] = df_abcd_female.age

dadaz = [[df_pond_male,'df_pond_male'],[df_pond_female,'df_pond_female'],[df_hbn_male,'df_hbn_male'],
        [df_hbn_female,'df_hbn_female'],[df_abcd_male,'df_abcd_male'],[df_abcd_female,'df_abcd_female']]
for data in dadaz: 
    print(data[1])
    store_covariate(data[0],'age',data[1])

normativz = [[df3_areax,'df3_areax'],[df3_thicknessx,'df3_thicknessx'],
             [df3_volumex,'df3_volumex'],[df3_subcortx,'df3_subcortx'],
             [df3_areax_fe,'df3_areax_fe'],[df3_thicknessx_fe,'df3_thicknessx_fe'],
                          [df3_volumex_fe,'df3_volumex_fe'],[df3_subcortx_fe,'df3_subcortx_fe'],]

for normz in normativz:
    nam  = 'features_normsample_'+normz[1]+'.txt'
    normz[0].to_csv(nam,sep = ' ',
                           header = False,
                           index = False)

nono_normativz = [[df1_areax,'df1_areax'],[df1_thicknessx,'df1_thicknessx'],
             [df1_volumex,'df1_volumex'],[df1_subcortx,'df1_subcortx'],
             [df2_areax,'df2_areax'],[df2_thicknessx,'df2_thicknessx'],
                          [df2_volumex,'df2_volumex'],[df2_subcortx,'df2_subcortx'],
                          
            [df1_areax_fe,'df1_areaxFE'],[df1_thicknessx_fe,'df1_thicknessxFE'],
                         [df1_volumex_fe,'df1_volumexFE'],[df1_subcortx_fe,'df1_subcortxFE'],
                         [df2_areax_fe,'df2_areaxFE'],[df2_thicknessx_fe,'df2_thicknessxFE'],
                                      [df2_volumex_fe,'df2_volumexFE'],[df2_subcortx_fe,'df2_subcortxFE']]

for nonnorm in nono_normativz:
    nam  = 'feature_'+nonnorm[1]+'.txt'
    nonnorm[0].to_csv(nam,sep = ' ',
                           header = False,
                           index = False)


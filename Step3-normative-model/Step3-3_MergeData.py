import pandas as pd
import numpy as np
import os
 

os.chdir("E:/Step3-normative-model")


df_pond_male = pd.read_csv('E:/Step2-correct-data/zresult_pond_male.csv')
df_pond_female = pd.read_csv('E:/Step2-correct-data/zresult_pond_female.csv')
df_hbn_male = pd.read_csv('E:/Step2-correct-data/zresult_hbn_male.csv')
df_hbn_female = pd.read_csv('E:/Step2-correct-data/zresult_hbn_female.csv')
df_abcd_male = pd.read_csv('E:/Step2-correct-data/zresult_abcd_male.csv')
df_abcd_female = pd.read_csv('E:/Step2-correct-data/zresult_abcd_female.csv')

df1_thickness = pd.read_csv("Z_df1_thickness_prob.txt", sep=" ", header=None)
df1_area = pd.read_csv("Z_df1_area_prob.txt", sep=" ", header=None)
df1_volume = pd.read_csv("Z_df1_volume_prob.txt", sep=" ", header=None)
df1_subcort = pd.read_csv("Z_df1_subcortical_prob.txt", sep=" ", header=None)
df2_thickness = pd.read_csv("Z_df2_thickness_prob.txt", sep=" ", header=None)
df2_area = pd.read_csv("Z_df2_area_prob.txt", sep=" ", header=None)
df2_volume = pd.read_csv("Z_df2_volume_prob.txt", sep=" ", header=None)
df2_subcort = pd.read_csv("Z_df2_subcortical_prob.txt", sep=" ", header=None)

df1_thickness_fe = pd.read_csv("Z_df1_thickness_prob_fe.txt", sep=" ", header=None)
df1_area_fe = pd.read_csv("Z_df1_area_prob_fe.txt", sep=" ", header=None)
df1_volume_fe = pd.read_csv("Z_df1_volume_prob_fe.txt", sep=" ", header=None)
df1_subcort_fe = pd.read_csv("Z_df1_subcortical_prob_fe.txt", sep=" ", header=None)
df2_thickness_fe = pd.read_csv("Z_df2_thickness_prob_fe.txt", sep=" ", header=None)
df2_area_fe = pd.read_csv("Z_df2_area_prob_fe.txt", sep=" ", header=None)
df2_volume_fe = pd.read_csv("Z_df2_volume_prob_fe.txt", sep=" ", header=None)
df2_subcort_fe = pd.read_csv("Z_df2_subcortical_prob_fe.txt", sep=" ", header=None)


df1_thickness.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_thick_combatted'):df_pond_male.columns.get_loc('rh_insula_thick_combatted') + 1 ]
df1_thickness_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_thick_combatted'):df_pond_male.columns.get_loc('rh_insula_thick_combatted') + 1 ]
df2_thickness.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_thick_combatted'):df_pond_male.columns.get_loc('rh_insula_thick_combatted') + 1 ]
df2_thickness_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_thick_combatted'):df_pond_male.columns.get_loc('rh_insula_thick_combatted') + 1 ]


df1_area.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_area_combatted'):df_pond_male.columns.get_loc('rh_insula_area_combatted') + 1 ]
df1_area_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_area_combatted'):df_pond_male.columns.get_loc('rh_insula_area_combatted') + 1 ]
df2_area.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_area_combatted'):df_pond_male.columns.get_loc('rh_insula_area_combatted') + 1 ]
df2_area_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_area_combatted'):df_pond_male.columns.get_loc('rh_insula_area_combatted') + 1 ]


df1_volume.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_vol_combatted'):df_pond_male.columns.get_loc('rh_insula_vol_combatted') + 1 ]
df1_volume_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_vol_combatted'):df_pond_male.columns.get_loc('rh_insula_vol_combatted') + 1 ]
df2_volume.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_vol_combatted'):df_pond_male.columns.get_loc('rh_insula_vol_combatted') + 1 ]
df2_volume_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('lh_bankssts_vol_combatted'):df_pond_male.columns.get_loc('rh_insula_vol_combatted') + 1 ]


df1_subcort.columns = df_pond_male.columns[df_pond_male.columns.get_loc('Left-Thalamus-Proper_combatted'):df_pond_male.columns.get_loc('Right-VentralDC_combatted') + 1 ]
df1_subcort_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('Left-Thalamus-Proper_combatted'):df_pond_male.columns.get_loc('Right-VentralDC_combatted') + 1 ]
df2_subcort.columns = df_pond_male.columns[df_pond_male.columns.get_loc('Left-Thalamus-Proper_combatted'):df_pond_male.columns.get_loc('Right-VentralDC_combatted') + 1 ]
df2_subcort_fe.columns = df_pond_male.columns[df_pond_male.columns.get_loc('Left-Thalamus-Proper_combatted'):df_pond_male.columns.get_loc('Right-VentralDC_combatted') + 1 ]



df_pond_male[list(df1_thickness.columns)] = df1_thickness
df_pond_male[list(df1_area.columns)] = df1_area
df_pond_male[list(df1_volume.columns)] = df1_volume
df_pond_male[list(df1_subcort.columns)] = df1_subcort

df_pond_female[list(df1_thickness.columns)] = df1_thickness_fe
df_pond_female[list(df1_area.columns)] = df1_area_fe
df_pond_female[list(df1_volume.columns)] = df1_volume_fe
df_pond_female[list(df1_subcort.columns)] = df1_subcort_fe

df_hbn_male[list(df2_thickness.columns)] = df2_thickness
df_hbn_male[list(df2_area.columns)] = df2_area
df_hbn_male[list(df2_volume.columns)] = df2_volume
df_hbn_male[list(df2_subcort.columns)] = df2_subcort


df_hbn_female[list(df2_thickness.columns)] = df2_thickness_fe
df_hbn_female[list(df2_area.columns)] = df2_area_fe
df_hbn_female[list(df2_volume.columns)] = df2_volume_fe
df_hbn_female[list(df2_subcort.columns)] = df2_subcort_fe


df_pond_female = df_pond_female.loc[df1_area_fe.index]



df_pond_male.to_csv("df_pond_male_final.csv")
df_pond_female.to_csv("df_pond_female_final.csv")

df_hbn_male.to_csv("df_hbn_male_final.csv")
df_hbn_female.to_csv("df_hbn_female_final.csv")
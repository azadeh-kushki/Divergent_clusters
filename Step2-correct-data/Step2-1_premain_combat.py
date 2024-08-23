import os
os.chdir("E:/Step2-correct-data")
import csv
from run_combat import run_combat_bootstrap
import pandas as pd



phenofinal_df_pond= pd.read_csv("E:/Step1-prepare-pheno/pheno_pond_df.csv")
phenofinal_df_hbn=  pd.read_csv("E:/Step1-prepare-pheno/pheno_hbn_df.csv")
df = pd.read_csv('E:/Step0-prepare-data/alldata-processed.csv')

# Load the names of the brain measures that we're using
brain_file = open('brain_measures-all.csv', 'r')
brain_vars = csv.reader(brain_file, delimiter=',')
brain_vars = [item for sublist in brain_vars for item in sublist]
brain_file.close()

## Data  
df_pond = phenofinal_df_pond
df_hbn = phenofinal_df_hbn
df_abcd = df.loc[df['dataset'] == 'ABCD']
df_hbn = df_hbn.dropna(subset=['Dx'])


# Set the clobber
clobber = 1

# # Run ComBat
df_pond_male = df_pond[df_pond.sex=="M"].reset_index(drop=True)
df_pond_female =  df_pond[df_pond.sex=="F"].reset_index(drop=True)
df_hbn_male = df_hbn[df_hbn.sex=="M"].reset_index(drop=True)
df_hbn_female =  df_hbn[df_hbn.sex=="F"].reset_index(drop=True)

df_abcd_male = df_abcd[df_abcd.sex=="M"].reset_index(drop=True)
df_abcd_female =  df_abcd[df_abcd.sex=="F"].reset_index(drop=True)




df_pond_malex = run_combat_bootstrap(df_pond_male,'scanner',[], brain_vars, 100, 1)
df_pond_femalex = run_combat_bootstrap(df_pond_female,'scanner',[], brain_vars, 100, 1)
df_hbn_malex = run_combat_bootstrap(df_hbn_male,'scanner',[], brain_vars, 100, 1)
df_hbn_femalex = run_combat_bootstrap(df_hbn_female,'scanner',[], brain_vars, 100, 1)

df_abcd_malex = run_combat_bootstrap(df_abcd_male,'scanner',[], brain_vars, 100, 1)
df_abcd_femalex = run_combat_bootstrap(df_abcd_female,'scanner',[], brain_vars, 100, 1)


df_pond_malex.dropna()
df_pond_femalex.dropna()
df_hbn_malex.dropna()
df_hbn_femalex.dropna()
df_abcd_malex.dropna()
df_abcd_femalex.dropna()

df_pond_male = df_pond_male.loc[df_pond_malex.index]
df_pond_female = df_pond_female.loc[df_pond_femalex.index]

df_hbn_male = df_hbn_male.loc[df_hbn_malex.index]
df_hbn_female = df_hbn_female.loc[df_hbn_femalex.index]


df_abcd_male = df_abcd_male.loc[df_abcd_malex.index]
df_abcd_female = df_abcd_female.loc[df_abcd_femalex.index]


full_df = pd.concat([df_pond_malex, df_pond_femalex, df_hbn_malex,df_hbn_femalex , df_abcd_malex,df_abcd_femalex ], axis=0)


# Save to CSV
df_pond_malex.to_csv('zresult_pond_male.csv', index=False)
df_pond_femalex.to_csv('zresult_pond_female.csv', index=False)
df_hbn_malex.to_csv('zresult_hbn_male.csv', index=False)
df_hbn_femalex.to_csv('zresult_hbn_female.csv', index=False)


full_df.to_csv('ZallDataCombat.csv', index=False)


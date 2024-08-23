import pandas as pd
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
import os 
os.chdir("E:/Step2-correct-data")

############################################################## ATTENTION ###################################################
### IN THIS FINAL VERSION THIS CODE DOES NOTHING AS WE DONT CORRECT FOR ANYTHING. IT JUST RENAME AND STORES THE DATA
############################################################################################################################
def normalize_data(df):
    [N, M] = df.shape
    z = df.copy() * 0
    for i in range(0, M):
        z.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean()) / (df.iloc[:, i].std())
    return (z)

####I mport combated dataframe 
df = pd.read_csv('ZallDataCombat.csv')
#### Separate by dataset'
brain_columns = [col for col in df.columns if '_combatted' in col]

df = df.dropna(subset=brain_columns)

df_pond = df.loc[df['dataset'] == 'POND']
df_hbn = df.loc[df['dataset'] == 'HBN']
df_abcd = df.loc[df['dataset'] == 'ABCD']


df_pond['subject_id'] = df_pond['subject_id'].astype(str)  
df_hbn['subject_id'] = df_hbn['subject_id'].astype(str)    

#Separate Brains
brain_columns = [col for col in df.columns if '_combatted' in col]
df_pond_brains = df_pond[brain_columns]
df_hbn_brains = df_hbn[brain_columns]
df_abcd_brains = df_abcd[brain_columns]


df_pond_male = df_pond[df_pond.sex=="M"]
df_pond_female = df_pond[df_pond.sex=="F"]

df_hbn_male = df_hbn[df_hbn.sex=="M"]
df_hbn_female = df_hbn[df_hbn.sex=="F"]

df_abcd_male = df_abcd[df_abcd.sex=="M"]
df_abcd_female = df_abcd[df_abcd.sex=="F"]


df_pond_brains_male = df_pond_brains[df_pond.sex=="M"]
df_pond_brains_female = df_pond_brains[df_pond.sex=="F"]

df_hbn_brains_male = df_hbn_brains[df_hbn.sex=="M"]
df_hbn_brains_female = df_hbn_brains[df_hbn.sex=="F"]

df_abcd_brains_male = df_abcd_brains[df_abcd.sex=="M"]
df_abcd_brains_female = df_abcd_brains[df_abcd.sex=="F"]


# re-indexing before correction
df_pond_brains_male = df_pond_brains_male.reset_index(drop=True)
df_pond_brains_female = df_pond_brains_female.reset_index(drop=True)
df_hbn_brains_male = df_hbn_brains_male.reset_index(drop=True)
df_hbn_brains_female = df_hbn_brains_female.reset_index(drop=True)
df_abcd_brains_male = df_abcd_brains_male.reset_index(drop=True)
df_abcd_brains_female = df_abcd_brains_female.reset_index(drop=True)

df_pond_male = df_pond_male.reset_index(drop=True)
df_pond_female = df_pond_female.reset_index(drop=True)
df_hbn_male = df_hbn_male.reset_index(drop=True)
df_hbn_female = df_hbn_female.reset_index(drop=True)
df_abcd_male = df_abcd_male.reset_index(drop=True)
df_abcd_female = df_abcd_female.reset_index(drop=True)



df_pond_male = df_pond_male.reset_index(drop=True)
df_pond_female = df_pond_female.reset_index(drop=True)
df_hbn_male = df_hbn_male.reset_index(drop=True)
df_hbn_female = df_hbn_female.reset_index(drop=True)
df_abcd_male = df_abcd_male.reset_index(drop=True)
df_abcd_female = df_abcd_female.reset_index(drop=True)


df_pond_male.to_csv("zresult_pond_male.csv",index=False)
df_pond_female.to_csv("zresult_pond_female.csv",index=False)

df_hbn_male.to_csv("zresult_hbn_male.csv",index=False)
df_hbn_female.to_csv("zresult_hbn_female.csv",index=False)

df_abcd_male.to_csv("zresult_abcd_male.csv",index=False)
df_abcd_female.to_csv("zresult_abcd_female.csv",index=False)



# # Scatter plot after correction
# plt.subplot(1, 2, 2)
# plt.scatter(df_pond['age'], POND_brainx.iloc[:,-1], alpha=0.5)
# plt.title(f"After Correction: {brain_region}")
# plt.xlabel("Age")
# plt.ylabel("Brain Region Value")

# plt.tight_layout()
# plt.show()

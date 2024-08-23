import pandas as pd
import numpy as np
import os 
os.chdir("\Step1-prepare-pheno")
def add_Dx_pond(df_pond):
    pond_diag = pd.read_csv("pond_extract_updated_31may2023_abcd.csv")
    pond_diag = pond_diag[pond_diag['PRIMARY_DIAGNOSIS'].isin(['ASD', 'ADHD', 'OCD', 'Typically Developing','Intellectual Disability only', 'Tourette Syndrome',
                                                               'Sub-threshold OCD', 'Sub-threshold ADHD','Social Communication Disorders', 'Intellectual Disability Only'])]
    pond_diag = pond_diag[pond_diag['subject'].isin(df_pond['subject_id'])]
    pond_diag = pond_diag.drop_duplicates(subset=['subject'], keep='first')
    df_pond['subject_id'] = df_pond['subject_id'].astype(str)
    pond_diag['subject'] = pond_diag['subject'].astype(str)  
    df_pond = pd.merge(df_pond, pond_diag[['subject', 'PRIMARY_DIAGNOSIS']], 
                       left_on='subject_id', 
                       right_on='subject', 
                       how='left')
    df_pond.rename(columns={'PRIMARY_DIAGNOSIS': 'DxAll'}, inplace=True)
    df_pond.drop('subject', axis=1, inplace=True)
    df_pond = df_pond.dropna(subset=['DxAll'])

    return(df_pond)

def most_frequent_diagnosis(row):
    diagnoses = row.dropna()
    if diagnoses.empty:
        return np.nan
    return diagnoses.mode()[0]

def add_Dx_hbn(df_hbn):
    HBN_diag = pd.read_csv("9994_ConsensusDx_20220728.csv")
    HBN_diag = HBN_diag.iloc[1:,:]
    diagnosis_columns = ['DX_01', 'DX_02', 'DX_03', 'DX_04', 'DX_05', 'DX_06', 'DX_07', 'DX_08', 'DX_09', 'DX_10']
    allowed_values = [
        'ADHD-Combined Type', 'Intellectual Disability-Mild', 'Other Specified Attention-Deficit/Hyperactivity Disorder','No Diagnosis Given',
        'Autism Spectrum Disorder','ADHD-Hyperactive/Impulsive Type',  'Unspecified Attention-Deficit/Hyperactivity Disorder', 'ADHD-Inattentive Type',
        'Provisional Tic Disorder', 'Borderline Intellectual Functioning', 'Intellectual Disability-Moderate','Obsessive-Compulsive Disorder','Tourettes Disorder',
        'Intellectual Disability-Severe', 'Persistent (Chronic) Motor or Vocal Tic Disorder','Other Specified Tic Disorder','Social (Pragmatic) Communication Disorder']
    new_df_columns = ['EID'] + diagnosis_columns
    HBN_Digz = HBN_diag[new_df_columns]
    subset_without_EID = HBN_Digz.drop(columns=['EID'])
    subset_without_EID = subset_without_EID.dropna(how='all')
    HBN_Digz = HBN_Digz.loc[subset_without_EID.index]
    
    # Apply the function to each row
    HBN_Digz['Consensus'] = HBN_Digz.iloc[:,1:].apply(most_frequent_diagnosis, axis=1)
    finalHBN_diag = HBN_Digz[['EID','Consensus']]
    finalHBN_diag = finalHBN_diag.drop_duplicates(subset=['EID'], keep='first')
    finalHBN_diag = finalHBN_diag[finalHBN_diag['EID'].isin(df_hbn['subject_id'])]
    df_hbn['subject_id'] = df_hbn['subject_id'].astype(str)
    finalHBN_diag['EID'] = finalHBN_diag['EID'].astype(str)
    df_hbn = pd.merge(df_hbn, finalHBN_diag[['EID', 'Consensus']], 
                       left_on='subject_id', 
                       right_on='EID', 
                       how='left')

    df_hbn.rename(columns={'Consensus': 'DxAll'}, inplace=True)
    df_hbn.drop('EID', axis=1, inplace=True)
    df_hbn = df_hbn.dropna(subset=['DxAll'])
    return(df_hbn)

def add_IQ_pond(df_pond):
    pheno = pd.read_csv("pond_extract_updated_31may2023_abcd.csv")
    pheno["FULL_IQ"] = pheno["WASI_II_FSIQ_4"]
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['WASI_FSIQ_4'])
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['WISC_V_FSIQ'])
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['WISC_IV_FSIQ'])
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['SBFULLIQ'])
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['WPPSI_IV47_FSIQ_CS'])
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['WASI_II_FSIQ_2'])
    pheno["FULL_IQ"] = pheno["FULL_IQ"].fillna(pheno['WASI_FSIQ_2'])
    df_pond['subject_id'] = df_pond['subject_id'].astype(str)  
    pheno = pheno.drop_duplicates(subset=['subject'], keep='first')

    pheno['subject'] = pheno['subject'].astype(str)    
    df_pond = pd.merge(df_pond, pheno[['subject', 'FULL_IQ']], 
                       left_on='subject_id', 
                       right_on='subject', 
                       how='left')
    
    df_pond.rename(columns={'FULL_IQ': 'FULL_IQ'}, inplace=True)
    df_pond.drop('subject', axis=1, inplace=True)
    return(df_pond)        

def add_IQ_hbn(df_hbn):
    WISCIQ = pd.read_csv("assessment_data/9994_WISC_20220728.csv").iloc[1:,:]
    WISCIQ = WISCIQ.drop_duplicates(subset=['EID'], keep='first')
    
    df_hbn['subject_id'] = df_hbn['subject_id'].astype(str)  
    WISCIQ['EID'] = WISCIQ['EID'].astype(str)    
    df_hbn = pd.merge(df_hbn, WISCIQ[['EID', 'WISC_FSIQ']], 
                       left_on='subject_id', 
                       right_on='EID', 
                       how='left')
    
    df_hbn.rename(columns={'WISC_FSIQ': 'FULL_IQ'}, inplace=True)
    df_hbn.drop('EID', axis=1, inplace=True)
    return(df_hbn)

def add_SCQ_pond(df_pond):
    pondSCQ = pd.read_csv("pond_extract_updated_31may2023_abcd.csv")
    
    df_pond['subject_id'] = df_pond['subject_id'].astype(str)  
    pondSCQ = pondSCQ.drop_duplicates(subset=['subject'], keep='first')

    pondSCQ['subject'] = pondSCQ['subject'].astype(str)    
    df_pond = pd.merge(df_pond, pondSCQ[['subject', 'SCQTOT']], 
                       left_on='subject_id', 
                       right_on='subject', 
                       how='left')
    
    df_pond.rename(columns={'SCQTOT': 'SCQTOT'}, inplace=True)
    df_pond.drop('subject', axis=1, inplace=True)
    return(df_pond)

def add_SCQ_hbn(df_hbn):
    SCQ = pd.read_csv("assessment_data/9994_SCQ_20220728.csv")
    SCQ = SCQ.rename(columns={"SCQ_Total": "SCQTOT"})
    SCQ = SCQ.drop_duplicates(subset=['EID'], keep='first')
    
    df_hbn['subject_id'] = df_hbn['subject_id'].astype(str)  
    SCQ['EID'] = SCQ['EID'].astype(str)    
    df_hbn = pd.merge(df_hbn, SCQ[['EID', 'SCQTOT']], 
                       left_on='subject_id', 
                       right_on='EID', 
                       how='left')
    
    df_hbn.rename(columns={'SCQTOT': 'SCQTOT'}, inplace=True)
    df_hbn.drop('EID', axis=1, inplace=True)
    return(df_hbn)

def add_SWAN_pond(df_pond):
    pheno = pd.read_csv("pond_extract_updated_31may2023_abcd.csv")
    
    df_pond['subject_id'] = df_pond['subject_id'].astype(str)  
    pheno = pheno.drop_duplicates(subset=['subject'], keep='first')

    pheno['subject'] = pheno['subject'].astype(str)    
    df_pond = pd.merge(df_pond, pheno[['subject', 'ADHD_I_SUB','ADHD_HI_SUB','CB68IPTS','CB68EPTS']], 
                       left_on='subject_id', 
                       right_on='subject', 
                       how='left')
    
    df_pond.rename(columns={'ADHD_I_SUB': 'ADHD_I_SUB'}, inplace=True)
    df_pond.rename(columns={'ADHD_HI_SUB': 'ADHD_HI_SUB'}, inplace=True)
    df_pond.rename(columns={'CB68IPTS': 'CBCL_Int'}, inplace=True)
    df_pond.rename(columns={'CB68EPTS': 'CBCL_Ext'}, inplace=True)

    df_pond.drop('subject', axis=1, inplace=True)
    return(df_pond)
def newSWAN(SWAN):
    SWAN_IN = SWAN[['SWAN_01','SWAN_02','SWAN_03','SWAN_04','SWAN_05','SWAN_06','SWAN_07','SWAN_08','SWAN_09']]
    SWAN_HI = SWAN[['SWAN_10','SWAN_11','SWAN_12','SWAN_13','SWAN_14','SWAN_15','SWAN_16','SWAN_17','SWAN_18']]
    ADHD_I_SUB = np.zeros([1,SWAN.shape[0]])[0]
    ADHD_HI_SUB = np.zeros([1,SWAN.shape[0]])[0]
    for i in range(SWAN_IN.shape[1]):
        temp = (SWAN_IN.iloc[:,i].astype(float)==2).astype(int) + (SWAN_IN.iloc[:,i].astype(float)==3).astype(int)
        ADHD_I_SUB = temp+ADHD_I_SUB
    for j in range(SWAN_HI.shape[1]):
        temp = (SWAN_HI.iloc[:,j].astype(float)==2).astype(int) + (SWAN_HI.iloc[:,j].astype(float)==3).astype(int)
        ADHD_HI_SUB = temp+ADHD_HI_SUB

    ADHD_I_SUB[SWAN_IN.isna().all(axis=1)] = np.nan
    ADHD_HI_SUB[SWAN_HI.isna().all(axis=1)] = np.nan
    return(ADHD_I_SUB, ADHD_HI_SUB)

def add_SWAN_hbn(df_hbn):
    SWAN = pd.read_csv("assessment_data/9994_SWAN_20220728.csv")
    SWAN = SWAN.iloc[1:,:]
    ADHD_I, ADHD_HI = newSWAN(SWAN)  
    SWAN['ADHD_I_SUB'] = ADHD_I
    SWAN['ADHD_HI_SUB'] = ADHD_HI
    SWAN = SWAN.drop_duplicates(subset=['EID'], keep='first')
    
    df_hbn['subject_id'] = df_hbn['subject_id'].astype(str)  
    SWAN['EID'] = SWAN['EID'].astype(str)    
    df_hbn = pd.merge(df_hbn, SWAN[['EID', 'ADHD_I_SUB','ADHD_HI_SUB']], 
                       left_on='subject_id', 
                       right_on='EID', 
                       how='left')
    
    df_hbn.rename(columns={'ADHD_I_SUB': 'ADHD_I_SUB'}, inplace=True)
    df_hbn.rename(columns={'ADHD_HI_SUB': 'ADHD_HI_SUB'}, inplace=True)
    df_hbn.drop('EID', axis=1, inplace=True)
    return(df_hbn)

def add_cbcl_hbn(df_hbn):
    CBCL =pd.read_csv("assessment_data/9994_CBCL_20220728.csv")
    CBCL = CBCL.drop_duplicates(subset=['EID'], keep='first')
    
    df_hbn['subject_id'] = df_hbn['subject_id'].astype(str)  
    CBCL['EID'] = CBCL['EID'].astype(str)    
    df_hbn = pd.merge(df_hbn, CBCL[['EID', 'CBCL_Int','CBCL_Ext']], 
                       left_on='subject_id', 
                       right_on='EID', 
                       how='left')
    
    df_hbn.rename(columns={'CBCL_Int': 'CBCL_Int'}, inplace=True)
    df_hbn.rename(columns={'CBCL_Ext': 'CBCL_Ext'}, inplace=True)

    df_hbn.drop('EID', axis=1, inplace=True)
    return(df_hbn)

    
df_pond = pd.read_csv('E:Step0-prepare-data/pond-processed.csv')
df_hbn = pd.read_csv('E:Step0-prepare-data/hbn-processed.csv')
##################### Add Diagnosis ##################################
D_df_pond = add_Dx_pond(df_pond) #having diagnosis is must
D_df_hbn =  add_Dx_hbn(df_hbn) #having diagnosis is must
##################### Add IQ ################a##################
Q_df_pond = add_IQ_pond(D_df_pond)  
Q_df_hbn = add_IQ_hbn(D_df_hbn)  
 ##################### Add SCQ ##################################
SC_df_pond = add_SCQ_pond(Q_df_pond)  
SC_df_hbn = add_SCQ_hbn(Q_df_hbn)  
##################### Add ADHD_I_SUB AND HI_SUB and CBCLs ##################################
final_df_pond = add_SWAN_pond(SC_df_pond)  
ADHDI_df_hbn = add_SWAN_hbn(SC_df_hbn)  
final_df_hbn = add_cbcl_hbn(ADHDI_df_hbn)

final_df_pond.to_csv("pheno_pond_df.csv",index=False)
final_df_hbn.to_csv("pheno_hbn_df.csv",index=False)

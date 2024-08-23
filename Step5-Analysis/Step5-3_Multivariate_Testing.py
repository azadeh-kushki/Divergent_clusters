import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

def xgb_classifier(pond, brain_regions, label_name, hbn):
    # Ensure the input labels and brain regions exist in both datasets
    common_brain_regions = [region for region in brain_regions if region in pond.columns and region in hbn.columns]
    if not common_brain_regions or label_name not in pond.columns or label_name not in hbn.columns:
        print("Error: Missing labels or brain regions in the datasets.")
        return None

    # Remove rows with missing data in the common brain regions or the label column
    pond_cleaned = pond.dropna(subset=common_brain_regions + [label_name])
    hbn_cleaned = hbn.dropna(subset=common_brain_regions + [label_name])

    # Prepare the label encoder to convert text labels to numeric
    label_encoder = LabelEncoder()

    # Preparing the training data from POND
    X_train = pond_cleaned[common_brain_regions]
    y_train = label_encoder.fit_transform(pond_cleaned[label_name])

    # Preparing the testing data from HBN
    X_test = hbn_cleaned[common_brain_regions]
    y_test = label_encoder.transform(hbn_cleaned[label_name])

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Initialize the classifier
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train_scaled, y_train)

    # Best parameters and best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_}")

    # Predict with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("-------------------")
    print(f"Model tested Accuracy: {accuracy:.4f}")
    print("-------------------")
    # Cross-validation scores
    cross_val_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores on training set: {cross_val_scores}")
    print(f"Mean cross-validation score: {cross_val_scores.mean()}")

    # Confusion matrix and ROC-AUC score
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n {conf_matrix}")

    # Remove "_combatted" suffix from feature names if it exists
    clean_feature_names = [region.replace('_combatted', '') for region in common_brain_regions]

    # Initialize the explainer
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_scaled)

    # Class encoding and direction printing
    class_encoding = label_encoder.classes_
    print("Class encoding: ", class_encoding)
    print("Positive SHAP values push towards: ", class_encoding[1])
    print("Negative SHAP values push towards: ", class_encoding[0])

    # Summary plot
    shap.summary_plot(shap_values, X_train, feature_names=clean_feature_names, plot_type="bar")
    plt.show()

    # Feature importance plot
    shap.summary_plot(shap_values, X_train, feature_names=clean_feature_names)
    plt.show()

    # Tree plot
    shap.decision_plot(explainer.expected_value, shap_values[0], feature_names=clean_feature_names)
    plt.show()

    return accuracy

def One_out_classification(df1, regions1,regions2, df2,class_of_interest):
    dfx = df1.copy()
    dfx2 = df2.copy()
    dfx['areal_labels'] = dfx['areal_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    dfx2['areal_labels'] = dfx2['areal_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    
    
    dfx['thickness_labels'] = dfx['thickness_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    dfx2['thickness_labels'] = dfx2['thickness_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    
    
    dfx['volume_labels'] = dfx['volume_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    dfx2['volume_labels'] = dfx2['volume_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    
    
    dfx['subcort_labels'] = dfx['subcort_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    dfx2['subcort_labels'] = dfx2['subcort_labels'].apply(lambda x: 'Reference' if x != class_of_interest else x)
    print("*****************************")
    print("Pond predict HBN: Area")
    xgb_classifier(dfx, regions1[0], 'areal_labels', dfx2)
    print("*****************************")
    print("Pond predict HBN: Thickness")
    xgb_classifier(dfx, regions1[1], 'thickness_labels', dfx2)
    print("*****************************")
    print("Pond predict HBN: Volume")
    xgb_classifier(dfx, regions1[2], 'volume_labels', dfx2)
    print("*****************************")
    print("Pond predict HBN: Subcortical")
    xgb_classifier(dfx, regions1[3], 'subcort_labels', dfx2)
    
    print("*****************************")
    print("HBN predict POND: Area")
    xgb_classifier(dfx2, regions2[0], 'areal_labels', dfx)
    print("*****************************")
    print("HBN predict POND: Thickness")
    xgb_classifier(dfx2, regions2[1], 'thickness_labels', dfx)
    print("*****************************")
    print("HBN predict POND: Volume")
    xgb_classifier(dfx2, regions2[2], 'volume_labels', dfx)
    print("*****************************")
    print("HBN predict POND: Subcortical")
    xgb_classifier(dfx2, regions2[3], 'subcort_labels', dfx)
    
    

os.chdir("E:/Step5-Analysis")

df_pond_male = pd.read_csv( 'E:/Step4-Clustering/POND_MALE_FINAL.csv')
df_pond_female = pd.read_csv('E:/POND_FEMALE_FINAL.csv')
df_hbn_male = pd.read_csv('E:/HBN_MALE_FINAL.csv')
df_hbn_female = pd.read_csv('E:/HBN_FEMALE_FINAL.csv')
  

os.chdir("E:/Step4-Clustering")
regions1 = [ pd.read_csv('POND_area_Cluster1.csv',index_col=0).columns.tolist(),
          pd.read_csv('POND_Thickness_Cluster1.csv',index_col=0).columns.tolist(),
          pd.read_csv('POND_volume_Cluster1.csv',index_col=0).columns.tolist(),
          pd.read_csv('POND_subcortical_Cluter1.csv',index_col=0).columns.tolist()]

regions2 = [ pd.read_csv('HBN_area_Cluster1.csv',index_col=0).columns.tolist(),
          pd.read_csv('HBN_Thickness_Cluster1.csv',index_col=0).columns.tolist(),
          pd.read_csv('HBN_volume_Cluster1.csv',index_col=0).columns.tolist(),
          pd.read_csv('HBN_subcortical_Cluster1.csv',index_col=0).columns.tolist()]



One_out_classification(df_pond_male, regions1,regions2, df_hbn_male,'Cluster1')
One_out_classification(df_pond_male, regions1,regions2, df_hbn_male,'Cluster2')

One_out_classification(df_pond_male, regions1,regions2, df_hbn_male,'Cluster3')


regions1x = [ pd.read_csv('POND_area_Cluster1_fe.csv',index_col=0).columns.tolist(),
         [],[],
          pd.read_csv('POND_subcortical_Cluster1_fe.csv',index_col=0).columns.tolist()]

regions2x = [ pd.read_csv('HBN_area_Cluster1_fe.csv',index_col=0).columns.tolist(),[],[],
          pd.read_csv('HBN_subcortical_Cluster1_fe.csv',index_col=0).columns.tolist()]


One_out_classification(df_pond_female, regions1x,regions2x, df_hbn_female,'Cluster1')


One_out_classification(df_pond_female, regions1x,regions2x, df_hbn_female,'Cluster2')



########## MAIN XGBOOST PHENO


df_pond_male = pd.read_csv( 'E:/Step4-Clustering/POND_MALE_FINAL.csv')
df_pond_female = pd.read_csv('E:/Step4-Clustering/POND_FEMALE_FINAL.csv')
df_hbn_male = pd.read_csv('E:/Step4-Clustering/HBN_MALE_FINAL.csv')
df_hbn_female = pd.read_csv('E:/Step4-Clustering/HBN_FEMALE_FINAL.csv')



features = ["age", "SCQTOT", "ADHD_I_SUB", "ADHD_HI_SUB", "CBCL_Int", "CBCL_Ext", "FULL_IQ"]
categorical_features = ["income", "education","race"]
labels = ['areal_labels', 'thickness_labels', 'volume_labels', 'subcort_labels']


def preprocess_race_feature(df, race_column):
    # Replace 'Cat-4' with 'WHITE' and others with 'minorities', and exclude 'Cat-7'
    df = df[df[race_column] != 'Cat-7']  # Exclude 'Cat-7'
    df[race_column] = df[race_column].replace({'Cat-4': 'WHITE'})
    df[race_column] = df[race_column].apply(lambda x: 'minorities' if x != 'WHITE' else x)
    return df

def preprocess_education(df, education_column):
    # Round up the education levels
    df[education_column] = df[education_column].apply(np.ceil)
    # Remove rows with NaN values
    df = df.dropna(subset=[education_column])
    # Convert to integer type
    df[education_column] = df[education_column].astype(int)
    return df


def process_income(df, in_column):
    # Round up the education levels
    # Remove rows with NaN values
    df = df.dropna(subset=[in_column])
    # Convert to integer type
    return df

def preprocess_categorical_features(df, categorical_features):
    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    
    # Create a DataFrame with encoded categorical features
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    # Drop the original categorical features from the original DataFrame
    df = df.drop(columns=categorical_features)
    
    # Concatenate the encoded categorical features with the original DataFrame
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    return df, encoder


df_pond_male = preprocess_race_feature(df_pond_male, 'race')
df_hbn_male = preprocess_race_feature(df_hbn_male, 'race')
df_pond_female = preprocess_race_feature(df_pond_female, 'race')
df_hbn_female = preprocess_race_feature(df_hbn_female, 'race')

df_pond_male = preprocess_education(df_pond_male, 'education')
df_pond_male = preprocess_education(df_pond_male, 'education')
df_pond_female = preprocess_education(df_pond_female, 'education')
df_hbn_female = preprocess_education(df_hbn_female, 'education')

df_pond_male = process_income(df_pond_male, 'income')
df_pond_male = process_income(df_pond_male, 'income')
df_pond_female = process_income(df_pond_female, 'income')
df_hbn_female = process_income(df_hbn_female, 'income')




df_pond_malez, encoder = preprocess_categorical_features(df_pond_male, categorical_features)
df_hbn_malez, _ = preprocess_categorical_features(df_hbn_male, categorical_features)

df_pond_femalez, encoder = preprocess_categorical_features(df_pond_female, categorical_features)
df_hbn_femalez, _ = preprocess_categorical_features(df_hbn_female, categorical_features)

phenoregions = features+encoder.get_feature_names_out(categorical_features).tolist()
allphen =  [phenoregions,phenoregions,phenoregions,phenoregions]

One_out_classification(df_pond_malez,allphen,allphen, df_hbn_malez,'Cluster1')
One_out_classification(df_pond_malez,allphen,allphen, df_hbn_malez,'Cluster2')
One_out_classification(df_pond_malez,allphen,allphen, df_hbn_malez,'Cluster3')


One_out_classification(df_hbn_femalez,allphen,allphen, df_hbn_femalez,'Cluster1')

One_out_classification(df_hbn_femalez,allphen,allphen, df_hbn_femalez,'Cluster2')

import pandas as pd

def One_vs_Reference_classification(df1, regions1, regions2, df2, class_of_interest):
    # Copy the dataframes to avoid modifying the original ones
    dfx = df1.copy()
    dfx2 = df2.copy()
    
    # Filter to include only the class of interest and "Reference"
    dfx = dfx[dfx['areal_labels'].isin([class_of_interest, 'Reference'])]
    dfx2 = dfx2[dfx2['areal_labels'].isin([class_of_interest, 'Reference'])]
    
    dfx = dfx[dfx['thickness_labels'].isin([class_of_interest, 'Reference'])]
    dfx2 = dfx2[dfx2['thickness_labels'].isin([class_of_interest, 'Reference'])]
    
    dfx = dfx[dfx['volume_labels'].isin([class_of_interest, 'Reference'])]
    dfx2 = dfx2[dfx2['volume_labels'].isin([class_of_interest, 'Reference'])]
    
    dfx = dfx[dfx['subcort_labels'].isin([class_of_interest, 'Reference'])]
    dfx2 = dfx2[dfx2['subcort_labels'].isin([class_of_interest, 'Reference'])]
    
    print("*****************************")
    print("Pond predict HBN: Area")
    xgb_classifier(dfx, regions1[0], 'areal_labels', dfx2)
    print("*****************************")
    print("Pond predict HBN: Thickness")
    xgb_classifier(dfx, regions1[1], 'thickness_labels', dfx2)
    print("*****************************")
    print("Pond predict HBN: Volume")
    xgb_classifier(dfx, regions1[2], 'volume_labels', dfx2)
    print("*****************************")
    print("Pond predict HBN: Subcortical")
    xgb_classifier(dfx, regions1[3], 'subcort_labels', dfx2)
    
    print("*****************************")
    print("HBN predict POND: Area")
    xgb_classifier(dfx2, regions2[0], 'areal_labels', dfx)
    print("*****************************")
    print("HBN predict POND: Thickness")
    xgb_classifier(dfx2, regions2[1], 'thickness_labels', dfx)
    print("*****************************")
    print("HBN predict POND: Volume")
    xgb_classifier(dfx2, regions2[2], 'volume_labels', dfx)
    print("*****************************")
    print("HBN predict POND: Subcortical")
    xgb_classifier(dfx2, regions2[3], 'subcort_labels', dfx)


One_vs_Reference_classification(df_pond_malez,allphen,allphen, df_hbn_malez,'Cluster1')
One_vs_Reference_classification(df_pond_malez,allphen,allphen, df_hbn_malez,'Cluster2')

One_vs_Reference_classification(df_pond_malez,allphen,allphen, df_hbn_malez,'Cluster3')
One_vs_Reference_classification(df_hbn_femalez,allphen,allphen, df_hbn_femalez,'Cluster1')
One_vs_Reference_classification(df_hbn_femalez,allphen,allphen, df_hbn_femalez,'Cluster2')


xgb_classifier(df_pond_malez, features+encoder.get_feature_names_out(categorical_features).tolist(), 'thickness_labels', df_hbn_malez)
xgb_classifier(df_pond_malez, features+encoder.get_feature_names_out(categorical_features).tolist(), 'subcort_labels', df_hbn_malez)

xgb_classifier(df_hbn_malez, features+encoder.get_feature_names_out(categorical_features).tolist(), 'thickness_labels', df_pond_malez)
xgb_classifier(df_hbn_malez, features+encoder.get_feature_names_out(categorical_features).tolist(), 'subcort_labels', df_pond_malez)
 


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

os.chdir("E:/Step4-Clustering")


df_pond_male = pd.read_csv('E:/Step3-normative-model/df_pond_male_final.csv')
df_pond_female = pd.read_csv('E:/Step3-normative-model/df_pond_female_final.csv')
df_hbn_male = pd.read_csv('E:/Step3-normative-model/df_hbn_male_final.csv')
df_hbn_female = pd.read_csv('E:/Step3-normative-model/df_hbn_female_final.csv')

################################# Split by Measure  ###############################
df1_thickness = df_pond_male.loc[:,
                                 'lh_bankssts_thick_combatted':'rh_insula_thick_combatted']
df1_thickness_fe = df_pond_female.loc[:,
                                      'lh_bankssts_thick_combatted':'rh_insula_thick_combatted']
df2_thickness = df_hbn_male.loc[:,
                                'lh_bankssts_thick_combatted':'rh_insula_thick_combatted']
df2_thickness_fe = df_hbn_female.loc[:,
                                     'lh_bankssts_thick_combatted':'rh_insula_thick_combatted']

df1_area = df_pond_male.loc[:,
                            'lh_bankssts_area_combatted':'rh_insula_area_combatted']
df1_area_fe = df_pond_female.loc[:,
                                 'lh_bankssts_area_combatted':'rh_insula_area_combatted']
df2_area = df_hbn_male.loc[:,
                           'lh_bankssts_area_combatted':'rh_insula_area_combatted']
df2_area_fe = df_hbn_female.loc[:,
                                'lh_bankssts_area_combatted':'rh_insula_area_combatted']

df1_volume = df_pond_male.loc[:,
                              'lh_bankssts_vol_combatted':'rh_insula_vol_combatted']
df1_volume_fe = df_pond_female.loc[:,
                                   'lh_bankssts_vol_combatted':'rh_insula_vol_combatted']
df2_volume = df_hbn_male.loc[:,
                             'lh_bankssts_vol_combatted':'rh_insula_vol_combatted']
df2_volume_fe = df_hbn_female.loc[:,
                                  'lh_bankssts_vol_combatted':'rh_insula_vol_combatted']

df1_subcort = df_pond_male.loc[:,
                               'Left-Thalamus-Proper_combatted':'Right-VentralDC_combatted']
df1_subcort_fe = df_pond_female.loc[:,
                                    'Left-Thalamus-Proper_combatted':'Right-VentralDC_combatted']
df2_subcort = df_hbn_male.loc[:,
                              'Left-Thalamus-Proper_combatted':'Right-VentralDC_combatted']
df2_subcort_fe = df_hbn_female.loc[:,
                                   'Left-Thalamus-Proper_combatted':'Right-VentralDC_combatted']

######################## Thresholding based on absolute value ########################
import pandas as pd

def process_brain_measure(df_full, df_measure, measure_name):
    nameG = 'group_type_'+measure_name
    # 1. Identify rows where all values are less than 2 (Reference)
    df_full[nameG] = df_measure.apply(lambda row: 'Reference' if all(row.abs() < 2 ) else 'Outlier', axis=1)

    # 2. Identify columns where all values are less than 2
    reference_columns = df_measure.columns[df_measure.apply(lambda col: all(col.abs() < 2 ))]
    
    # 3. Create output dataframes
    reference_rows = df_full[df_full[nameG] == 'Reference']
    outlier_rows = df_full[df_full[nameG] == 'Outlier']

    reference_df = reference_rows[reference_columns]
    outlier_df = outlier_rows[df_measure.columns.difference(reference_columns)]

    # Print the counts
    print(f"Reference: {reference_rows.shape[0]} participants, {len(reference_columns)} columns")
    print(f"Outliers: {outlier_rows.shape[0]} participants, {len(df_measure.columns.difference(reference_columns))} columns")

    return df_full, reference_df, outlier_df

# Example usage for POND male thickness

df1_female = pd.read_csv("E:/Step4-Clustering/POND_FEMALE_FINAL.csv")
thicks = df1_female.loc[:, 'lh_bankssts_thick_combatted':'rh_insula_thick_combatted']
df1_femalex, reference_pond_thickness_fe, outlier_pond_thickness_fe = process_brain_measure(df1_female, thicks, "thickness")

vols = df1_female.loc[:,'lh_bankssts_vol_combatted':'rh_insula_vol_combatted']
df1_femalex, reference_pond_thickness_fe, outlier_pond_thickness_fe = process_brain_measure(df1_femalex, vols, "volume")


df_pond_male = pd.read_csv('E:/df_pond_male_final.csv')
df1_thickness = df_pond_male.loc[:, 'lh_bankssts_thick_combatted':'rh_insula_thick_combatted']

df_pond_malex, reference_pond_thickness, outlier_pond_thickness = process_brain_measure(df_pond_male, df1_thickness, "thickness")
df_pond_femalex, reference_pond_thickness_fe, outlier_pond_thickness_fe = process_brain_measure(df_pond_female, df1_thickness_fe, "thickness")

df_hbn_malex, reference_hbn_thickness, outlier_hbn_thickness = process_brain_measure(df_hbn_male, df2_thickness, "thickness")
df_hbn_femalex, reference_hbn_thickness_fe, outlier_hbn_thickness_fe = process_brain_measure(df_hbn_female, df2_thickness_fe, "thickness")

df_pond_malex, reference_pond_area, outlier_pond_area = process_brain_measure(df_pond_malex, df1_area, "area")
df_pond_femalex, reference_pond_area_fe, outlier_pond_area_fe = process_brain_measure(df_pond_femalex, df1_area_fe, "area")
df_hbn_malex, reference_hbn_area, outlier_hbn_area = process_brain_measure(df_hbn_malex, df2_area, "area")
df_hbn_femalex, reference_hbn_area_fe, outlier_hbn_area_fe= process_brain_measure(df_hbn_femalex, df2_area_fe, "area")

df_pond_malex, reference_pond_volume, outlier_pond_volume = process_brain_measure(df_pond_malex, df1_volume, "volume")
df_pond_femalex, reference_pond_volume_fe, outlier_pond_volume_fe = process_brain_measure(df_pond_femalex, df1_volume_fe, "volume")
df_hbn_malex, reference_hbn_volume, outlier_hbn_volume = process_brain_measure(df_hbn_malex, df2_volume, "volume")
df_hbn_female, reference_hbn_volume_fe, outlier_hbn_volume_fe = process_brain_measure(df_hbn_femalex, df2_volume_fe, "volume")

df_pond_malex, reference_pond_subcort, outlier_pond_subcort = process_brain_measure(df_pond_malex, df1_subcort, "subcortical")
df_pond_femalex, reference_pond_subcort_fe, outlier_pond_subcort_fe = process_brain_measure(df_pond_femalex, df1_subcort_fe, "subcortical")
df_hbn_malex, reference_hbn_subcort, outlier_hbn_subcort = process_brain_measure(df_hbn_malex, df2_subcort, "subcortical")
df_hbn_femalex, reference_hbn_subcort_fe, outlier_hbn_subcort_fe = process_brain_measure(df_hbn_femalex, df2_subcort_fe, "subcortical")

brain_outlier_df_pond_male_thickness = outlier_pond_thickness
brain_outlier_df_pond_female_thickness = outlier_pond_thickness_fe
brain_outlier_df_hbn_male_thickness = outlier_hbn_thickness
brain_outlier_df_hbn_female_thickness = outlier_hbn_thickness_fe

brain_outlier_df_pond_male_area = outlier_pond_area
brain_outlier_df_pond_female_area = outlier_pond_area_fe
brain_outlier_df_hbn_male_area = outlier_hbn_area
brain_outlier_df_hbn_female_area = outlier_hbn_area_fe

brain_outlier_df_pond_male_volume = outlier_pond_volume
brain_outlier_df_pond_female_volume =outlier_pond_volume_fe
brain_outlier_df_hbn_male_volume = outlier_hbn_volume
brain_outlier_df_hbn_female_volume = outlier_hbn_volume_fe

brain_outlier_df_pond_male_subcort = outlier_pond_subcort
brain_outlier_df_pond_female_subcort = outlier_pond_subcort_fe
brain_outlier_df_hbn_male_subcort =outlier_hbn_subcort
brain_outlier_df_hbn_female_subcort = outlier_hbn_subcort_fe


######################## Final Optimal Number of Clusters ########################
def evaluate_clustering_metrics(df, max_clusters=10):
    silhouette_scores = []
    calinski_scores = []
    range_clusters = range(2, max_clusters + 1)

    for n_clusters in range_clusters:
        model = SpectralCoclustering(n_clusters=n_clusters, random_state=42)
        model.fit(df)
        if n_clusters == 1:
            # Silhouette score is not defined for a single cluster
            silhouette_scores.append(-1)
            calinski_scores.append(
                calinski_harabasz_score(df, model.row_labels_))
        else:
            silhouette_scores.append(silhouette_score(df, model.row_labels_))
            calinski_scores.append(
                calinski_harabasz_score(df, model.row_labels_))

    return range_clusters, silhouette_scores, calinski_scores


def plot_clustering_metrics(range_clusters, silhouette_scores, calinski_scores, title):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters', fontsize=18)
    ax1.set_ylabel('Silhouette Score', color=color, fontsize=18)
    ax1.plot(range_clusters, silhouette_scores, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Calinski-Harabasz Score', color=color, fontsize=18)
    ax2.plot(range_clusters, calinski_scores, marker='o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    plt.show()
    fig.savefig(f"{title}_New.svg", format='svg')


# List of your dataframes
dataframes = {
    "POND_Thickness_Male": brain_outlier_df_pond_male_thickness,
    "POND_Thickness_Female": brain_outlier_df_pond_female_thickness,
    "HBN_Thickness_Male": brain_outlier_df_hbn_male_thickness,
    "HBN_Thickness_Female": brain_outlier_df_hbn_female_thickness,
    "POND_Area_Male": brain_outlier_df_pond_male_area,
    "POND_Area_Female": brain_outlier_df_pond_female_area,
    "HBN_Area_Male": brain_outlier_df_hbn_male_area,
    "HBN_Area_Female": brain_outlier_df_hbn_female_area,
    "POND_Volume_Male": brain_outlier_df_pond_male_volume,
    "POND_Volume_Female": brain_outlier_df_pond_female_volume,
    "HBN_Volume_Male": brain_outlier_df_hbn_male_volume,
    "HBN_Volume_Female": brain_outlier_df_hbn_female_volume,
    "POND_Subcort_Male": brain_outlier_df_pond_male_subcort,
    "POND_Subcort_Female": brain_outlier_df_pond_female_subcort,
    "HBN_Subcort_Male": brain_outlier_df_hbn_male_subcort,
    "HBN_Subcort_Female": brain_outlier_df_hbn_female_subcort
}



# Plot metrics for each dataframe
for title, df in dataframes.items():
    range_clusters, silhouette_scores, calinski_scores = evaluate_clustering_metrics(
        df)

    plot_clustering_metrics(range_clusters, silhouette_scores,
                            calinski_scores, title=f'Clustering Metrics for {title}_New')


######################## Final Optimal Number of Clusters ########################

# Function to evaluate clustering metrics
def evaluate_clustering_metrics(df, max_clusters=10, n_trials=100):
    range_clusters = range(2, max_clusters + 1)
    results = {
        "n_clusters": [],
        "silhouette_mean": [],
        "silhouette_std": [],
        "calinski_mean": [],
        "calinski_std": [],
        "davies_mean": [],
        "davies_std": []
    }

    for n_clusters in range_clusters:
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []

        for _ in range(n_trials):
            model = SpectralCoclustering(n_clusters=n_clusters, random_state=np.random.randint(10000))
            model.fit(df)
            row_labels = model.row_labels_

            silhouette_scores.append(silhouette_score(df, row_labels))
            calinski_scores.append(calinski_harabasz_score(df, row_labels))
            davies_scores.append(davies_bouldin_score(df, row_labels))

        results["n_clusters"].append(n_clusters)
        results["silhouette_mean"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_mean"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["davies_mean"].append(np.mean(davies_scores))
        results["davies_std"].append(np.std(davies_scores))

    return pd.DataFrame(results)

# List of your dataframes
dataframes = {
    "POND_Thickness_Male": brain_outlier_df_pond_male_thickness,
    "POND_Thickness_Female": brain_outlier_df_pond_female_thickness,
    "HBN_Thickness_Male": brain_outlier_df_hbn_male_thickness,
    "HBN_Thickness_Female": brain_outlier_df_hbn_female_thickness,
    "POND_Area_Male": brain_outlier_df_pond_male_area,
    "POND_Area_Female": brain_outlier_df_pond_female_area,
    "HBN_Area_Male": brain_outlier_df_hbn_male_area,
    "HBN_Area_Female": brain_outlier_df_hbn_female_area,
    "POND_Volume_Male": brain_outlier_df_pond_male_volume,
    "POND_Volume_Female": brain_outlier_df_pond_female_volume,
    "HBN_Volume_Male": brain_outlier_df_hbn_male_volume,
    "HBN_Volume_Female": brain_outlier_df_hbn_female_volume,
    "POND_Subcort_Male": brain_outlier_df_pond_male_subcort,
    "POND_Subcort_Female": brain_outlier_df_pond_female_subcort,
    "HBN_Subcort_Male": brain_outlier_df_hbn_male_subcort,
    "HBN_Subcort_Female": brain_outlier_df_hbn_female_subcort
}

# Evaluate metrics for each dataframe and store results in a dictionary
results_dict = {title: evaluate_clustering_metrics(df) for title, df in dataframes.items()}

# Combine all results into a single dataframe
combined_results = pd.concat(results_dict, axis=0)
combined_results.index.names = ['Dataset', 'Row']
combined_results.reset_index(level='Dataset', inplace=True)


######################## Compute 2 Cluster Solution [OPTIMAL] ########################
def cocluster_stable(df, n_clusters=2, n_runs=100):
    models = []
    labels_list = []

    for i in range(n_runs):
        model = SpectralCoclustering(n_clusters=n_clusters, random_state=i)
        model.fit(df)

        if np.any(np.sum(model.rows_, axis=1) == 0) or np.any(np.sum(model.columns_, axis=1) == 0):  # Check for empty clusters
            continue

        models.append(model)
        labels_list.append(model.row_labels_)

    if not models:
        return None, pd.DataFrame(), pd.DataFrame()  # No valid model was found

    consensus_scores = np.zeros((len(models), len(models)))
    try:
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i != j:
                    consensus_scores[i, j] = consensus_score(
                        model_a.biclusters_, model_b.biclusters_)
    except ValueError as e:
        print("Error calculating consensus scores:", e)
        return None, pd.DataFrame(), pd.DataFrame()
    
    # Find the model with the highest average consensus score
    average_consensus = consensus_scores.mean(axis=1)
    best_model_index = np.argmax(average_consensus)
    best_model = models[best_model_index]

    row_labels = best_model.row_labels_
    col_labels = best_model.column_labels_

    # Extract clusters
    cluster1_df = df.iloc[row_labels == 0, :].iloc[:, col_labels == 0]
    cluster2_df = df.iloc[row_labels == 1, :].iloc[:, col_labels == 1]
    if n_clusters==3:
        cluster3_df = df.iloc[row_labels == 2, :].iloc[:, col_labels == 2]
        return row_labels, cluster1_df, cluster2_df, cluster3_df
    else:    
        if cluster1_df.empty or cluster2_df.empty:
            return None, pd.DataFrame(), pd.DataFrame()
    
        return row_labels, cluster1_df, cluster2_df

n_clusters = 2
n_runs = 100

# Male
labels_p_t, pond_c1_t, pond_c2_t= cocluster_stable(brain_outlier_df_pond_male_thickness, n_clusters, n_runs)
labels_h_t, hbn_c1_t, hbn_c2_t = cocluster_stable(brain_outlier_df_hbn_male_thickness, n_clusters, n_runs)

labels_p_a, pond_c1_a, pond_c2_a ,pond_c3_a= cocluster_stable(brain_outlier_df_pond_male_area, 3, n_runs)
labels_h_a, hbn_c1_a, hbn_c2_a , hbn_c3_a= cocluster_stable(brain_outlier_df_hbn_male_area, 3, n_runs)

labels_p_v, pond_c1_v, pond_c2_v = cocluster_stable(brain_outlier_df_pond_male_volume, n_clusters, n_runs)
labels_h_v, hbn_c1_v, hbn_c2_v = cocluster_stable(brain_outlier_df_hbn_male_volume, n_clusters, n_runs)
labels_p_s, pond_c1_s, pond_c2_s = cocluster_stable(brain_outlier_df_pond_male_subcort, n_clusters, n_runs)
labels_h_s, hbn_c1_s, hbn_c2_s = cocluster_stable(brain_outlier_df_hbn_male_subcort, n_clusters, n_runs)

# Female
labels_p_t_fe, pond_c1_t_fe, pond_c2_t_fe = cocluster_stable(brain_outlier_df_pond_female_thickness, n_clusters, n_runs)

labels_h_t_fe, hbn_c1_t_fe, hbn_c2_t_fe = cocluster_stable(brain_outlier_df_hbn_female_thickness, n_clusters, n_runs)
labels_p_a_fe, pond_c1_a_fe, pond_c2_a_fe = cocluster_stable(brain_outlier_df_pond_female_area, n_clusters, n_runs)
labels_h_a_fe, hbn_c1_a_fe, hbn_c2_a_fe = cocluster_stable(brain_outlier_df_hbn_female_area, n_clusters, n_runs)

labels_p_v_fe, pond_c1_v_fe, pond_c2_v_fe = cocluster_stable(brain_outlier_df_pond_female_volume, n_clusters, n_runs)

labels_h_v_fe, hbn_c1_v_fe, hbn_c2_v_fe = cocluster_stable(brain_outlier_df_hbn_female_volume, n_clusters, n_runs)
labels_p_s_fe, pond_c1_s_fe, pond_c2_s_fe = cocluster_stable(brain_outlier_df_pond_female_subcort, n_clusters, n_runs)
labels_h_s_fe, hbn_c1_s_fe, hbn_c2_s_fe = cocluster_stable(brain_outlier_df_hbn_female_subcort, 2, n_runs)


######################## Overlapping by Jaccard  ########################
def compute_overlap(df1, df2):
    features_df1 = set(df1.columns)
    features_df2 = set(df2.columns)
    intersection = features_df1.intersection(features_df2)
    union = features_df1.union(features_df2)
    return len(intersection) / len(union) if union else 0



# Prepare the data for the heatmap
data = {
    'C1 POND vs C1 HBN': [
        compute_overlap(pond_c1_a, hbn_c1_a),
        compute_overlap(pond_c1_t, hbn_c1_t),
        compute_overlap(pond_c1_v, hbn_c1_v),
        compute_overlap(pond_c1_s, hbn_c1_s)
    ],
    'C1 POND vs C2 HBN': [
        compute_overlap(pond_c1_a, hbn_c2_a),
        compute_overlap(pond_c1_t, hbn_c2_t),
        compute_overlap(pond_c1_v, hbn_c2_v),
        compute_overlap(pond_c1_s, hbn_c2_s)
    ],
    'C2 POND vs C1 HBN': [
        compute_overlap(pond_c2_a, hbn_c1_a),
        compute_overlap(pond_c2_t, hbn_c1_t),
        compute_overlap(pond_c2_v, hbn_c1_v),
        compute_overlap(pond_c2_s, hbn_c1_s)
    ],
    'C2 POND vs C2 HBN': [
        compute_overlap(pond_c2_a, hbn_c2_a),
        compute_overlap(pond_c2_t, hbn_c2_t),
        compute_overlap(pond_c2_v, hbn_c2_v),
        compute_overlap(pond_c2_s, hbn_c2_s) 
    ],
    'C3 POND vs C3 HBN': [
        compute_overlap(pond_c3_a, hbn_c3_a),
    ],
    'C3 POND vs C2 HBN': [
        compute_overlap(pond_c3_a, hbn_c2_a),
    ],
    'C3 POND vs C1 HBN': [
        compute_overlap(pond_c3_a, hbn_c1_a),
    ],
    'C1 POND vs C3 HBN': [
        compute_overlap(pond_c1_a, hbn_c3_a),
    ],
    'C2 POND vs C3 HBN': [
        compute_overlap(pond_c2_a, hbn_c3_a),
    ]

}


# Create a DataFrame from the dictionary
jaccard_df = pd.DataFrame(
    data, index=['Area', 'Thickness', 'Volume', 'Subcortical'])
fig, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(jaccard_df, annot=True, cmap='Blues', fmt=".2f")
title = "Male-JacardMatch_new"  # Define your title or pass it as a variable
fig.savefig(f"{title}.svg", format='svg')


data_fe = {
    'C1 POND vs C1 HBN': [
        compute_overlap(pond_c1_a_fe, hbn_c1_a_fe),
        compute_overlap(pond_c1_t_fe, hbn_c1_t_fe),
        compute_overlap(pond_c1_v_fe, hbn_c1_v_fe),
        compute_overlap(pond_c1_s_fe, hbn_c1_s_fe)
    ],
    'C1 POND vs C2 HBN': [
        compute_overlap(pond_c1_a_fe, hbn_c2_a_fe),
        compute_overlap(pond_c1_t_fe, hbn_c2_t_fe),
        compute_overlap(pond_c1_v_fe, hbn_c2_v_fe),
        compute_overlap(pond_c1_s_fe, hbn_c2_s_fe)
    ],
    'C2 POND vs C1 HBN': [
        compute_overlap(pond_c2_a_fe, hbn_c1_a_fe),
        compute_overlap(pond_c2_t_fe, hbn_c1_t_fe),
        compute_overlap(pond_c2_v_fe, hbn_c1_v_fe),
        compute_overlap(pond_c2_s_fe, hbn_c1_s_fe)
    ],
    'C2 POND vs C2 HBN': [
        compute_overlap(pond_c2_a_fe, hbn_c2_a_fe),
        compute_overlap(pond_c2_t_fe, hbn_c2_t_fe),
        compute_overlap(pond_c2_v_fe, hbn_c2_v_fe),
        compute_overlap(pond_c2_s_fe, hbn_c2_s_fe)
    ]
}

# Create a DataFrame from the dictionary
jaccard_df_fe = pd.DataFrame(
    data_fe, index=['Area', 'Thickness', 'Volume', 'Subcortical'])
# Create the heatmap
fig, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(jaccard_df_fe, annot=True, cmap='Blues', fmt=".2f")
title = "Female-JacardMatch_nEW"  # Define your title or pass it as a variable
fig.savefig(f"{title}.svg", format='svg')


# COMPARING MALE AND FEMALE
def compute_overlap(df1, df2):
    features_df1 = set(df1.columns)
    features_df2 = set(df2.columns)
    intersection = features_df1.intersection(features_df2)
    union = features_df1.union(features_df2)
    return len(intersection) / len(union) if union else 0


# Prepare the data for the heatmap
data_p = {
    'C1 POND-MALE vs C1 POND-FEMALE': [
        compute_overlap(pond_c1_a, pond_c1_a_fe),
        compute_overlap(pond_c1_t, pond_c1_t_fe),
        compute_overlap(pond_c1_v, pond_c1_v_fe),
        compute_overlap(pond_c1_s, pond_c1_s_fe)
    ],
    'C1 POND-MALE vs C2 POND-FEMALE': [
        compute_overlap(pond_c1_a, pond_c2_a_fe),
        compute_overlap(pond_c1_t, pond_c2_t_fe),
        compute_overlap(pond_c1_v, pond_c2_v_fe),
        compute_overlap(pond_c1_s, pond_c2_s_fe)
    ],
    'C2 POND_MALE vs C1 POND_FEMALE': [
        compute_overlap(pond_c2_a, pond_c1_a_fe),
        compute_overlap(pond_c2_t, pond_c1_t_fe),
        compute_overlap(pond_c2_v, pond_c1_v_fe),
        compute_overlap(pond_c2_s, pond_c1_s_fe)
    ],
    'C2 POND-MALE vs C2 POND-FEMALE': [
        compute_overlap(pond_c2_a, pond_c2_a_fe),
        compute_overlap(pond_c2_t, pond_c2_t_fe),
        compute_overlap(pond_c2_v, pond_c2_v_fe),
        compute_overlap(pond_c2_s, pond_c2_s_fe)]
}


# Prepare the data for the heatmap
data_h = {
    'C1 HBN-MALE vs C1 HBN-FEMALE': [
        compute_overlap(hbn_c1_a, hbn_c1_a_fe),
        compute_overlap(hbn_c1_t, hbn_c1_t_fe),
        compute_overlap(hbn_c1_v, hbn_c1_v_fe),
        compute_overlap(hbn_c1_s, hbn_c1_s_fe)
    ],
    'C1 HBN-MALE vs C2 HBN-FEMALE': [
        compute_overlap(hbn_c1_a, hbn_c2_a_fe),
        compute_overlap(hbn_c1_t, hbn_c2_t_fe),
        compute_overlap(hbn_c1_v, hbn_c2_v_fe),
        compute_overlap(hbn_c1_s, hbn_c2_s_fe)
    ],
    'C2 HBN_MALE vs C1 HBN_FEMALE': [
        compute_overlap(hbn_c2_a, hbn_c1_a_fe),
        compute_overlap(hbn_c2_t, hbn_c1_t_fe),
        compute_overlap(hbn_c2_v, hbn_c1_v_fe),
        compute_overlap(hbn_c2_s, hbn_c1_s_fe)
    ],
    'C2 HBN-MALE vs C2 HBN-FEMALE': [
        compute_overlap(hbn_c2_a, hbn_c2_a_fe),
        compute_overlap(hbn_c2_t, hbn_c2_t_fe),
        compute_overlap(hbn_c2_v, hbn_c2_v_fe),
        compute_overlap(hbn_c2_s, hbn_c2_s_fe)]
}

# Create a DataFrame from the dictionary
jaccard_df_p = pd.DataFrame(
    data_p, index=['Area', 'Thickness', 'Volume', 'Subcortical'])
fig, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(jaccard_df_p, annot=True, cmap='Blues', fmt=".2f")
title = "POND-JacardMatch_NEW"  # Define your title or pass it as a variable
fig.savefig(f"{title}.svg", format='svg')


jaccard_df_h = pd.DataFrame(
    data_h, index=['Area', 'Thickness', 'Volume', 'Subcortical'])

fig, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(jaccard_df_h, annot=True, cmap='Blues', fmt=".2f")
title = "HBN-JacardMatch_NEW"  # Define your title or pass it as a variable
fig.savefig(f"{title}.svg", format='svg')


##################### Outlier vs Conformer Match ######################

labels_p_t = np.where(labels_p_t == 0, 'Cluster1', 'Cluster2')
labels_p_a = np.select([labels_p_a == 0,labels_p_a == 1,labels_p_a == 2], ['Cluster1', 'Cluster2','Cluster3'])
labels_p_v = np.where(labels_p_v == 0, 'Cluster1', 'Cluster2')
labels_p_s = np.where(labels_p_s == 0, 'Cluster1', 'Cluster2')

labels_h_t = np.where(labels_h_t == 0, 'Cluster2', 'Cluster1')
labels_h_a = np.select([labels_h_a == 0,labels_h_a == 1,labels_h_a == 2], ['Cluster3', 'Cluster1','Cluster2'])
labels_h_v = np.where(labels_h_v == 0, 'Cluster1', 'Cluster2')
labels_h_s = np.where(labels_h_s == 0, 'Cluster2', 'Cluster1')

labels_p_a_fe = np.where(labels_p_a_fe == 0, 'Cluster1', 'Cluster2')
labels_h_a_fe = np.where(labels_h_a_fe == 0, 'Cluster2', 'Cluster1')
labels_h_t_fe = np.where(labels_h_t_fe == 0, 'Cluster1', 'Cluster2')
labels_h_v_fe = np.where(labels_h_v_fe == 0, 'Cluster1', 'Cluster2')

labels_p_s_fe = np.where(labels_p_s_fe == 0, 'Cluster1', 'Cluster2')
labels_h_s_fe = np.where(labels_h_s_fe == 0, 'Cluster1', 'Cluster2')



df_pond_malex['thickness_labels'] = df_pond_malex['group_type_thickness']
df_pond_malex['thickness_labels'][df_pond_malex['thickness_labels'] != "thickness_labels"] = labels_p_t
df_pond_malex['areal_labels'] = df_pond_malex['group_type_area']
df_pond_malex['areal_labels'][df_pond_malex['areal_labels'] != "areal_labels"] = labels_p_a
df_pond_malex['volume_labels'] = df_pond_malex['group_types_volume']
df_pond_malex['volume_labels'][df_pond_malex['volume_labels'] != "volume_labels"] = labels_p_v
df_pond_malex['subcort_labels'] = df_pond_malex['group_type_subcortical']
df_pond_malex['subcort_labels'][df_pond_malex['volume_labels'] != "subcort_labels"] = labels_p_s

df_pond_femalex['thickness_labels'] = df_pond_femalex['group_type_thickness']
df_pond_femalex['thickness_labels'][df_pond_femalex['thickness_labels'] != "thickness_labels"] = labels_p_t_fe
df_pond_femalex['areal_labels'] = df_pond_femalex['group_type_area']
df_pond_femalex['areal_labels'][df_pond_femalex['areal_labels'] != "areal_labels"] = labels_p_a_fe
df_pond_femalex['volume_labels'] = df_pond_femalex['group_types_volume']
df_pond_femalex['volume_labels'][df_pond_femalex['volume_labels'] != "volume_labels"] = labels_p_v_fe
df_pond_femalex['subcort_labels'] = df_pond_femalex['group_type_subcortical']
df_pond_femalex['subcort_labels'][df_pond_femalex['volume_labels'] != "subcort_labels"] = labels_p_s_fe




df_hbn_malex['thickness_labels'] = df_hbn_malex['group_type_thickness']
df_hbn_malex['thickness_labels'][df_hbn_malex['thickness_labels'] != "thickness_labels"] = labels_h_t
df_hbn_malex['areal_labels'] = df_hbn_malex['group_type_area']
df_hbn_malex['areal_labels'][df_hbn_malex['areal_labels'] != "areal_labels"] = labels_h_a
df_hbn_malex['volume_labels'] = df_hbn_malex['group_types_volume']
df_hbn_malex['volume_labels'][df_hbn_malex['volume_labels'] != "volume_labels"] = labels_h_v
df_hbn_malex['subcort_labels'] = df_hbn_malex['group_type_subcortical']
df_hbn_malex['subcort_labels'][df_hbn_malex['volume_labels'] != "subcort_labels"] = labels_h_s

df_hbn_femalex['thickness_labels'] = df_hbn_femalex['group_type_thickness']
df_hbn_femalex['thickness_labels'][df_hbn_femalex['thickness_labels'] != "thickness_labels"] = labels_h_t_fe
df_hbn_femalex['areal_labels'] = df_hbn_femalex['group_type_area']
df_hbn_femalex['areal_labels'][df_hbn_femalex['areal_labels'] != "areal_labels"] = labels_h_a_fe
df_hbn_femalex['volume_labels'] = df_hbn_femalex['group_types_volume']
df_hbn_femalex['volume_labels'][df_hbn_femalex['volume_labels'] != "volume_labels"] = labels_h_v_fe
df_hbn_femalex['subcort_labels'] = df_hbn_femalex['group_type_subcortical']
df_hbn_femalex['subcort_labels'][df_hbn_femalex['volume_labels'] != "subcort_labels"] = labels_h_s_fe


######################### MAIN FILE EXPORTS ##########################
df_pond_malex.to_csv("POND_MALE_FINAL.csv")
df_pond_femalex.to_csv("POND_FEMALE_FINAL.csv")
df_hbn_femalex.to_csv("HBN_FEMALE_FINAL.csv")
######################## EXPORTS  ########################
pond_c1_t.to_csv("POND_Thickness_Cluster1.csv")
pond_c2_t.to_csv("POND_Thickness_Cluster2.csv")
hbn_c1_t.to_csv("HBN_Thickness_Cluster2.csv")
hbn_c2_t.to_csv("HBN_Thickness_Cluster1.csv")
pond_c1_a.to_csv("POND_area_Cluster1.csv")
pond_c2_a.to_csv("POND_area_Cluster2.csv")
pond_c3_a.to_csv("POND_area_Cluster3.csv")
hbn_c3_a.to_csv("HBN_area_Cluster1.csv")
hbn_c1_a.to_csv("HBN_area_Cluster2.csv")
hbn_c2_a.to_csv("HBN_area_Cluster3.csv")
pond_c1_v.to_csv("POND_volume_Cluster1.csv")
pond_c2_v.to_csv("POND_volume_Cluster2.csv")
hbn_c2_v.to_csv("HBN_volume_Cluster2.csv")
hbn_c1_v.to_csv("HBN_volume_Cluster1.csv")
pond_c1_s.to_csv("POND_subcortical_Cluter1.csv")
pond_c2_s.to_csv("POND_subcortical_Cluter2.csv")
hbn_c2_s.to_csv("HBN_subcortical_Cluster1.csv")
hbn_c1_s.to_csv("HBN_subcortical_Cluster2.csv")

# FEMALE
hbn_c2_t_fe.to_csv("HBN_Thickness_Cluster2_fe.csv")
hbn_c1_t_fe.to_csv("HBN_Thickness_Cluster1_fe.csv")
pond_c1_a_fe.to_csv("POND_area_Cluster1_fe.csv")
pond_c2_a_fe.to_csv("POND_area_Cluster2_fe.csv")
hbn_c2_a_fe.to_csv("HBN_area_Cluster1_fe.csv")
hbn_c1_a_fe.to_csv("HBN_area_Cluster2_fe.csv")
hbn_c1_v_fe.to_csv("HBN_volume_Cluster1_fe.csv")
hbn_c2_v_fe.to_csv("HBN_volume_Cluster2_fe.csv")
pond_c1_s_fe.to_csv("POND_subcortical_Cluster1_fe.csv")
pond_c2_s_fe.to_csv("POND_subcortical_Cluster2_fe.csv")
hbn_c2_s_fe.to_csv("HBN_subcortical_Cluster1_fe.csv")
hbn_c1_s_fe.to_csv("HBN_subcortical_Cluster2_fe.csv")




# List of DataFrames and their corresponding CSV names
dfs_and_names = [
    ("POND_Thickness_Cluster1.csv", pond_c1_t),
    ("POND_Thickness_Cluster2.csv", pond_c2_t),
    ("HBN_Thickness_Cluster1.csv", hbn_c2_t),
    ("HBN_Thickness_Cluster2.csv", hbn_c1_t),
    ("POND_area_Cluster1.csv", pond_c1_a),
    ("POND_area_Cluster2.csv", pond_c2_a),
    ("POND_area_Cluster3.csv", pond_c3_a),
    ("HBN_area_Cluster1.csv", hbn_c3_a),
    ("HBN_area_Cluster2.csv", hbn_c1_a),
    ("HBN_area_Cluster3.csv", hbn_c2_a),
    ("POND_volume_Cluster1.csv", pond_c1_v),
    ("POND_volume_Cluster2.csv", pond_c2_v),
    ("HBN_volume_Cluster2.csv", hbn_c2_v),
    ("HBN_volume_Cluster1.csv", hbn_c1_v),
    ("POND_subcortical_Cluster1.csv", pond_c1_s),
    ("POND_subcortical_Cluster2.csv", pond_c2_s),
    ("HBN_subcortical_Cluster1.csv", hbn_c2_s),
    ("HBN_subcortical_Cluster2.csv", hbn_c1_s),
    ("HBN_Thickness_Cluster2_fe.csv", hbn_c2_t_fe),
    ("HBN_Thickness_Cluster1_fe.csv", hbn_c1_t_fe),
    ("POND_area_Cluster1_fe.csv", pond_c1_a_fe),
    ("POND_area_Cluster2_fe.csv", pond_c2_a_fe),
    ("HBN_area_Cluster1_fe.csv", hbn_c2_a_fe),
    ("HBN_area_Cluster2_fe.csv", hbn_c1_a_fe),
    ("HBN_volume_Cluster1_fe.csv", hbn_c1_v_fe),
    ("HBN_volume_Cluster2_fe.csv", hbn_c2_v_fe),
    ("POND_subcortical_Cluster1_fe.csv", pond_c1_s_fe),
    ("POND_subcortical_Cluster2_fe.csv", pond_c2_s_fe),
    ("HBN_subcortical_Cluster1_fe.csv", hbn_c2_s_fe),
    ("HBN_subcortical_Cluster2_fe.csv", hbn_c1_s_fe),
]


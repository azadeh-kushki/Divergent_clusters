# Load necessary libraries# Load necessary libraries
library(dplyr)
library(broom)  # for tidy() function
library(purrr)  # for map functions
library(stats)  # for p.adjust function
library(effsize)  # for cohen.d function
library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(car)
library("factoextra")
library("FactoMineR")
library("factoextra")

setwd("E:/Step5-Analysis")

options(digits=3)
options(max.print=1000)

######################### IMPORT DATA ##########################                                                          
df1_male <- read.csv(file = "E:/Step4-Clustering/POND_MALE_FINAL.csv")
df1_female <- read.csv(file = "E:/Step4-Clustering/POND_FEMALE_FINAL.csv")

df2_male <- read.csv(file = "E:/Step4-Clustering/HBN_MALE_FINAL.csv")
df2_female <- read.csv(file = "E:/Step4-Clustering/HBN_FEMALE_FINAL.csv")

df3_male <- read.csv(file = "E:/Step2-correct-data/zresult_abcd_male.csv")
df3_female <- read.csv(file = "E:/Step2-correct-data/zresult_abcd_female.csv")


abcd_df<- read.csv(file ="F:/Step0-prepare-data/abcd_df-processed.csv")
hcp_df<- read.csv(file ="F:/hcpd_df-processed.csv")

pond <- rbind(df1_male, df1_female)
hbn <- rbind(df2_male, df2_female)
abcd <- rbind(df3_male, df3_female)
normative <- rbind(abcd_df,hcp_df)


# Function to perform t-test and return a tidy data frame with Cohen's d and p-value
perform_t_test <- function(data, cluster_col, cluster_value, reference_value, region) {
  subset_data <- data %>%
    filter(!!sym(cluster_col) %in% c(cluster_value, reference_value))
  
  t_test_result <- t.test(subset_data[[region]] ~ subset_data[[cluster_col]], 
                          subset = subset_data[[cluster_col]] %in% c(cluster_value, reference_value))
  
  cohen_d_result <- cohen.d(subset_data[[region]], subset_data[[cluster_col]], na.rm = TRUE, 
                            pooled = TRUE, hedges.correction = TRUE)
  cohen_d_value <- cohen_d_result$estimate
  tidy_result <- tidy(t_test_result)
  p_value <- tidy_result$p.value
  data.frame(
    region = region,
    effect_size = cohen_d_value,
    p_value = p_value
  )
}

# Function to process multiple regions for a given dataset and cluster comparison
process_brain_measures <- function(data, cluster_col, cluster_value, reference_value, columns_of_interest) {
  t_test_results <- map_dfr(columns_of_interest, ~perform_t_test(data, cluster_col, cluster_value, reference_value, .x))
  t_test_results <- t_test_results %>%
    mutate(adjusted_p_value = p.adjust(p_value, method = "fdr"),
           value = ifelse(adjusted_p_value < 0.05, effect_size, 0))
  final_results <- t_test_results %>%
    select(region, effect_size, adjusted_p_value, value)

  return(final_results)
}

# Function to read CSV, process data, and save results
process_and_save_results <- function(csv_file, data, cluster_col, cluster_value, reference_value, output_file) {
  columns_of_interest <- gsub("^X", "", colnames(read.csv(csv_file))[-1])  # Remove "X" from column names and start from the second element
  results <- process_brain_measures(data, cluster_col, cluster_value, reference_value, columns_of_interest)
  write.csv(results, output_file, row.names = FALSE)
}

# Process and save results for various combinations
process_and_save_results("POND_Thickness_Cluster1.csv", df1_male, "thickness_labels", "Cluster1", "Reference", "POND_male_thickness_cluster1_vs_reference.csv")
process_and_save_results("POND_Thickness_Cluster2.csv", df1_male, "thickness_labels", "Cluster2", "Reference", "POND_male_thickness_cluster2_vs_reference.csv")
process_and_save_results("HBN_Thickness_Cluster1.csv", df2_male, "thickness_labels", "Cluster1", "Reference", "HBN_male_thickness_cluster1_vs_reference.csv")
process_and_save_results("HBN_Thickness_Cluster2.csv", df2_male, "thickness_labels", "Cluster2", "Reference", "HBN_male_thickness_cluster2_vs_reference.csv")
process_and_save_results("POND_area_Cluster1.csv", df1_male, "areal_labels", "Cluster1", "Reference", "POND_male_area_cluster1_vs_reference.csv")
process_and_save_results("POND_area_Cluster2.csv", df1_male, "areal_labels", "Cluster2", "Reference", "POND_male_area_cluster2_vs_reference.csv")
process_and_save_results("POND_area_Cluster3.csv", df1_male, "areal_labels", "Cluster3", "Reference", "POND_male_area_cluster3_vs_reference.csv")
process_and_save_results("HBN_area_Cluster1.csv", df2_male, "areal_labels", "Cluster1", "Reference", "HBN_male_area_cluster1_vs_reference.csv")
process_and_save_results("HBN_area_Cluster2.csv", df2_male, "areal_labels", "Cluster2", "Reference", "HBN_male_area_cluster2_vs_reference.csv")
process_and_save_results("HBN_area_Cluster3.csv", df2_male, "areal_labels", "Cluster3", "Reference", "HBN_male_area_cluster3_vs_reference.csv")
process_and_save_results("POND_volume_Cluster1.csv", df1_male, "volume_labels", "Cluster1", "Reference", "POND_male_volume_cluster1_vs_reference.csv")
process_and_save_results("POND_volume_Cluster2.csv", df1_male, "volume_labels", "Cluster2", "Reference", "POND_male_volume_cluster2_vs_reference.csv")
process_and_save_results("HBN_volume_Cluster1.csv", df2_male, "volume_labels", "Cluster1", "Reference", "HBN_male_volume_cluster1_vs_reference.csv")
process_and_save_results("HBN_volume_Cluster2.csv", df2_male, "volume_labels", "Cluster2", "Reference", "HBN_male_volume_cluster2_vs_reference.csv")
process_and_save_results("POND_subcortical_Cluter1.csv", df1_male, "subcort_labels", "Cluster1", "Reference", "POND_male_subcortical_cluster1_vs_reference.csv")
process_and_save_results("POND_subcortical_Cluter2.csv", df1_male, "subcort_labels", "Cluster2", "Reference", "POND_male_subcortical_cluster2_vs_reference.csv")
process_and_save_results("HBN_subcortical_Cluster1.csv", df2_male, "subcort_labels", "Cluster1", "Reference", "HBN_male_subcortical_cluster1_vs_reference.csv")
process_and_save_results("HBN_subcortical_Cluster2.csv", df2_male, "subcort_labels", "Cluster2", "Reference", "HBN_male_subcortical_cluster2_vs_reference.csv")
process_and_save_results("POND_area_Cluster1_fe.csv", df1_female, "areal_labels", "Cluster1", "Reference", "POND_female_area_cluster1_vs_reference_fe.csv")
process_and_save_results("POND_area_Cluster2_fe.csv", df1_female, "areal_labels", "Cluster2", "Reference", "POND_female_area_cluster2_vs_reference_fe.csv")
process_and_save_results("HBN_area_Cluster1_fe.csv", df2_female, "areal_labels", "Cluster1", "Reference", "HBN_female_area_cluster1_vs_reference_fe.csv")
process_and_save_results("HBN_area_Cluster2_fe.csv", df2_female, "areal_labels", "Cluster2", "Reference", "HBN_female_area_cluster2_vs_reference_fe.csv")
process_and_save_results("POND_subcortical_Cluster1_fe.csv", df1_female, "subcort_labels", "Cluster1", "Reference", "POND_female_subcortical_cluster1_vs_reference_fe.csv")
process_and_save_results("POND_subcortical_Cluster2_fe.csv", df1_female, "subcort_labels", "Cluster2", "Reference", "POND_female_subcortical_cluster2_vs_reference_fe.csv")
process_and_save_results("HBN_subcortical_Cluster1_fe.csv", df2_female, "subcort_labels", "Cluster1", "Reference", "HBN_female_subcortical_cluster1_vs_reference_fe.csv")
process_and_save_results("HBN_subcortical_Cluster2_fe.csv", df2_female, "subcort_labels", "Cluster2", "Reference", "HBN_female_subcortical_cluster2_vs_reference_fe.csv")
process_and_save_results("HBN_Thickness_Cluster1_fe.csv", df2_female, "thickness_labels", "Cluster1", "Reference", "HBN_female_thickness_cluster1_vs_reference_fe.csv")
process_and_save_results("HBN_Thickness_Cluster2_fe.csv", df2_female, "thickness_labels", "Cluster2", "Reference", "HBN_female_thickness_cluster2_vs_reference_fe.csv")
process_and_save_results("HBN_volume_Cluster1_fe.csv", df2_female, "volume_labels", "Cluster1", "Reference", "HBN_female_volume_cluster1_vs_reference_fe.csv")
process_and_save_results("HBN_volume_Cluster2_fe.csv", df2_female, "volume_labels", "Cluster2", "Reference", "HBN_female_volume_cluster2_vs_reference_fe.csv")

######################## Phenotypes vs Diagnostic groups  #######################
main_stats <- function(d,p) {
  medi <- tapply(d,p,median,na.rm = TRUE)
  IQR <- tapply(d,p,IQR,na.rm = TRUE)
  meanz <-  tapply(d, p, mean, na.rm = TRUE)
  stdz <- tapply(d, p, sd, na.rm = TRUE)
  if (shapiro.test(d)$p<0.05){
    pk <- kruskal.test(d~p)
  } else{
    pk <- pairwise.wilcox.test(d,p)
    print("Normally Distributed")
  }
  if (pk$p.value < 0.05) {
    # Perform pairwise comparisons if significant
    pairwise_results <- pairwise.wilcox.test(d, p, p.adjust.method = "fdr")
    print("--- PAIRWISE ----")
    print(pairwise_results)
  }
  else {
    print("Normally Distributed")
    # Use ANOVA for normally distributed data
    pk <- aov(d ~ p)
    if (summary(pk)[[1]][["Pr(>F)"]][1] < 0.05) {
      # Perform Tukey's HSD test for pairwise comparisons if significant
      pairwise_results <- TukeyHSD(pk)
      print("--- PAIRWISE ----")
      print(pairwise_results)
    }}
  #  pk <- ks.test(d,p)
  print("Median: " )
  print(medi)
  print("__________________________________________________________________________________________")
  print("IQR")
  print(IQR)
  print("__________________________________________________________________________________________")
  print("Mean")
  print(meanz)
  print("__________________________________________________________________________________________")
  print("Std")
  print(stdz)
  print("__________________________________________________________________________________________")
  print("p-valuF:")
  print(pk)
  
  medi_formatted <- sprintf("%.2f", medi)
  IQR_formatted <- sprintf("%.2f", IQR)
  meanz_formatted <- sprintf("%.2f", meanz)
  stdz_formatted <- sprintf("%.2f", stdz)
  
  # Concatenate the formatted strings
  result_string <- paste0(medi_formatted, "(", IQR_formatted, ")-", meanz_formatted, "(", stdz_formatted, ")")
  
  # Print the result
  print(result_string)
  
}



table(df2_female$Dx,df2_female$thickness_labels)
table(df2_female$Dx,df2_female$volume_labels)
main_stats(df1_female$age,df1_female$group_type_thickness)
table(df1_female$Dx,df1_female$group_type_volume)
main_stats(df1_female$age,df1_female$group_type_volume)

contingency_table <- table(df1_male$Dx, df1_male$cluster_vs_all)
prop.test(contingency_table)
table(df1_female$Dx,df1_female$group_type_thickness)
main_stats(df1_female$age,df1_female$group_type_thickness)


table(df1_female$Dx,df1_female$group_type_volume)
main_stats(df1_female$age,df1_female$group_type_volume)

############################# BASIC STATS ############################################
table(hbn$Dx)
table(pond$Dx)
table(hbn$DxAllxx)
main_stats(pond$age,pond$Dx)
table(pond$Dx,pond$sex)
main_stats(hbn$age,hbn$Dx)

# Extracting DxAll values where Dx is "Other"
hbnother <- hbn$DxAll[hbn$Dx == "Other"]
table(hbnother)

pondother <- pond$DxAll[pond$Dx == "Other"]
table(pondother)
print("Diagnostics Counts POND:")
table(as.factor(factor(pond$DxAllx))) #Count per diagnosis
print("Diagnostics Counts HBN:")
table(as.factor(factor(hbn$DxAllx))) #Count per diagnosis
print("Phenotypic Measure (Age) POND:")
main_stats(pond$age,pond$DxAllx)
print("Phenotypic Measure (Age) HBN:")
main_stats(hbn$age,hbn$DxAllx)
print("Phenotypic Measure (sex) for POND:")
table(pond$DxAllx,pond$sex)
prop.test(table(pond$DxAllx,pond$sex),table( as.factor(factor(pond$DxAllx))))
print("Phenotypic Measure (sex) for HBN:")
table(hbn$DxAllx,hbn$sex)
prop.test(table(hbn$DxAllx,hbn$sex),table( as.factor(factor(hbn$DxAllx))))
print("Phenotypic Measure (SCQ) POND:")
main_stats(pond$SCQTOT,pond$DxAllx)
print("Phenotypic Measure (SCQ) HBN:")
main_stats(hbn$SCQTOT,hbn$DxAllx)
print("Phenotypic Measure (ADHD_I_SUB) POND:")
main_stats(pond$ADHD_I_SUB,pond$DxAllx)
print("Phenotypic Measure (ADHD_I_SUB) HBN:")
main_stats(hbn$ADHD_I_SUB,hbn$DxAllx)
print("Phenotypic Measure (ADHD_HI_SUB) POND:")
main_stats(pond$ADHD_HI_SUB,pond$DxAllx)
print("Phenotypic Measure (ADHD_HI_SUB) HBN:")
main_stats(hbn$ADHD_HI_SUB,hbn$DxAllx)
print("Phenotypic Measure (FULL_IQ) POND:")
main_stats(pond$FULL_IQ,pond$DxAllx)
print("Phenotypic Measure (FULL_IQ) HBN:")
main_stats(hbn$FULL_IQ,hbn$DxAllx)
print("Phenotypic Measure (FULL_IQ) POND:")
main_stats(pond$CBCL_I,pond$DxAllx)
print("Phenotypic Measure (FULL_IQ) HBN:")
main_stats(hbn$CBCL_Int_T,hbn$Dx)
main_stats(hbn$CBCL_Ext_T,hbn$Dx)
main_stats(pond$CBCL_E,pond$DxAllx)
main_stats(hbn$CBCL_E,hbn$DxAllx)
table(pond$income,pond$DxAllx)
table(hbn$income,hbn$DxAllx)
table(pond$education,pond$DxAllx)
table(hbn$education,hbn$DxAllx)
table(pond$race,pond$DxAllx)
table(hbn$education,hbn$DxAllx)
pond <- rbind(df1_male, df1_female)
hbn <- rbind(df2_male, df2_female)
abcd <- rbind(df3_male, df3_female)
normative <- rbind(abcd_df,hcp_df)

overall_stats <- function(dfx) {

  median_age <- median(dfx, na.rm = TRUE)
  iqr_age <- IQR(dfx, na.rm = TRUE)
  mean_age <- mean(dfx, na.rm = TRUE)
  std_age <- sd(dfx, na.rm = TRUE)
  median_age_f <- sprintf("%.2f", median_age)
  iqr_age_f <- sprintf("%.2f", iqr_age)
  mean_age_f <- sprintf("%.2f", mean_age)
  std_age_f <- sprintf("%.2f", std_age)
  result_string <- paste0(median_age_f, "(", iqr_age_f, ")-", mean_age_f, "(", std_age_f, ")")
  print(result_string)
  
}

overall_stats(abcd_df$cbcl_syn_tot)
table(pond$race)

######################## Charactrizing Clusters ##########################
perform_tests_multiple_labels <- function(df, label_columns) {
  # Continuous variables for t-test
  continuous_vars <- c("age", "SCQTOT", "ADHD_I_SUB", "ADHD_HI_SUB", "FULL_IQ", "CBCL_Int", "CBCL_Ext")
  # Categorical variables for chi-square test
  categorical_vars <- c("Dx", "race", "income", "education")
  # Initialize result list
  results_list <- list()
  # Collect p-values
  p_values <- list()
  for (label_column in label_columns) {
    results <- data.frame(matrix(ncol = length(continuous_vars) + length(categorical_vars), nrow = 1))
    colnames(results) <- c(continuous_vars, categorical_vars)
    # Perform t-tests
    for (var in continuous_vars) {
      df_test <- df[!is.na(df[[var]]) & !is.na(df[[label_column]]), ]
      formula <- as.formula(paste(var, "~", label_column))
      anova_result <- Anova(lm(formula, data = df_test))
      p_value <- anova_result$`Pr(>F)`[1]
      p_values[[paste(var, label_column, sep = "_")]] <- p_value
    }
    
    # Perform chi-square tests for categorical variables
    for (var in categorical_vars) {
      df_test_cat <- df[!is.na(df[[var]]) & !is.na(df[[label_column]]), ]
      table_data <- table(df_test_cat[[var]], df_test_cat[[label_column]])
      if (min(dim(table_data)) > 1) {
        chi_result <- chisq.test(table_data)
        p_value <- chi_result$p.value
        p_values[[paste(var, label_column, sep = "_")]] <- p_value
      } else {
        p_values[[paste(var, label_column, sep = "_")]] <- NA
      }
    }
    
    results_list[[label_column]] <- results
  }
  
  # Adjust all p-values together using Bonferroni correction
  all_p_values <- unlist(p_values)
  adjusted_p_values <- p.adjust(all_p_values, method = "holm")
  
  # Update results with adjusted p-values
  idx <- 1
  final_results <- data.frame(matrix(ncol = length(continuous_vars) + length(categorical_vars), nrow = length(label_columns)))
  colnames(final_results) <- c(continuous_vars, categorical_vars)
  rownames(final_results) <- label_columns
  
  for (label_column in label_columns) {
    results <- results_list[[label_column]]
    for (var in c(continuous_vars, categorical_vars)) {
      key <- paste(var, label_column, sep = "_")
      if (!is.na(p_values[[key]])) {
        adjusted_p <- adjusted_p_values[idx]
        if (var %in% continuous_vars) {
          df_test <- df[!is.na(df[[var]]) & !is.na(df[[label_column]]), ]
          formula <- as.formula(paste(var, "~", label_column))
          anova_result <- Anova(lm(formula, data = df_test))
          final_results[label_column, var] <- paste(round(anova_result$`F value`[1], 2), "(", ifelse(adjusted_p < 0.001, "<0.001", round(adjusted_p, 4)), ")", sep = "")
        } else {
          df_test_cat <- df[!is.na(df[[var]]) & !is.na(df[[label_column]]), ]
          table_data <- table(df_test_cat[[var]], df_test_cat[[label_column]])
          if (min(dim(table_data)) > 1) {
            chi_result <- chisq.test(table_data)
            final_results[label_column, var] <- paste(round(chi_result$statistic, 2), "(", ifelse(adjusted_p < 0.001, "<0.001", round(adjusted_p, 4)), ")", sep = "")
          }
        }
        idx <- idx + 1
      }
    }
  }
  
  return(final_results)
}
perform_tests_multiple_labels <- function(df, label_columns, target_cluster) {
  # Continuous variables for t-test
  continuous_vars <- c("age", "SCQTOT", "ADHD_I_SUB", "ADHD_HI_SUB", "FULL_IQ", "CBCL_Int", "CBCL_Ext")
  # Categorical variables for chi-square test
  categorical_vars <- c("Dx", "race", "income", "education")
  # Initialize result list
  results_list <- list()
  # Collect p-values
  p_values <- list()
  for (label_column in label_columns) {
    results <- data.frame(matrix(ncol = length(continuous_vars) + length(categorical_vars), nrow = 1))
    colnames(results) <- c(continuous_vars, categorical_vars)
    # Filter the dataframe to include only the target cluster and the reference group
    df_filtered <- df[df[[label_column]] %in% c("Reference", target_cluster), ]
    # Perform t-tests
    for (var in continuous_vars) {
      df_test <- df_filtered[!is.na(df_filtered[[var]]) & !is.na(df_filtered[[label_column]]), ]
      formula <- as.formula(paste(var, "~", label_column))
      anova_result <- Anova(lm(formula, data = df_test))
      p_value <- anova_result$`Pr(>F)`[1]
      p_values[[paste(var, label_column, sep = "_")]] <- p_value
    }
    
    # Perform chi-square tests for categorical variables
    for (var in categorical_vars) {
      df_test_cat <- df_filtered[!is.na(df_filtered[[var]]) & !is.na(df_filtered[[label_column]]), ]
      table_data <- table(df_test_cat[[var]], df_test_cat[[label_column]])
      if (min(dim(table_data)) > 1) {
        chi_result <- chisq.test(table_data)
        p_value <- chi_result$p.value
        p_values[[paste(var, label_column, sep = "_")]] <- p_value
      } else {
        p_values[[paste(var, label_column, sep = "_")]] <- NA
      }
    }
    
    results_list[[label_column]] <- results
  }
  
  # Adjust all p-values together using Bonferroni correction
  all_p_values <- unlist(p_values)
  adjusted_p_values <- p.adjust(all_p_values, method = "fdr")
  
  # Update results with adjusted p-values
  idx <- 1
  final_results <- data.frame(matrix(ncol = length(continuous_vars) + length(categorical_vars), nrow = length(label_columns)))
  colnames(final_results) <- c(continuous_vars, categorical_vars)
  rownames(final_results) <- label_columns
  
  for (label_column in label_columns) {
    results <- results_list[[label_column]]
    for (var in c(continuous_vars, categorical_vars)) {
      key <- paste(var, label_column, sep = "_")
      if (!is.na(p_values[[key]])) {
        adjusted_p <- adjusted_p_values[idx]
        if (var %in% continuous_vars) {
          df_test <- df_filtered[!is.na(df_filtered[[var]]) & !is.na(df_filtered[[label_column]]), ]
          formula <- as.formula(paste(var, "~", label_column))
          anova_result <- Anova(lm(formula, data = df_test))
          final_results[label_column, var] <- paste(round(anova_result$`F value`[1], 2), "(", ifelse(adjusted_p < 0.001, "<0.001", round(adjusted_p, 4)), ")", sep = "")
        } else {
          df_test_cat <- df_filtered[!is.na(df_filtered[[var]]) & !is.na(df_filtered[[label_column]]), ]
          table_data <- table(df_test_cat[[var]], df_test_cat[[label_column]])
          if (min(dim(table_data)) > 1) {
            chi_result <- chisq.test(table_data)
            final_results[label_column, var] <- paste(round(chi_result$statistic, 2), "(", ifelse(adjusted_p < 0.001, "<0.001", round(adjusted_p, 4)), ")", sep = "")
          }
        }
        idx <- idx + 1
      }
    }
  }
  
  return(final_results)
}

# Example usage
label_columns <- c("areal_labels", "thickness_labels", "volume_labels", "subcort_labels")
male_pheno_omnibus_results <- perform_tests_multiple_labels(df1_male, label_columns, "Cluster1")
write.csv(male_pheno_omnibus_results,"Male_pheno_omnibust_cluster1.csv",row.names=TRUE)
male_pheno_omnibus_results <- perform_tests_multiple_labels(df1_male, label_columns, "Cluster2")
write.csv(male_pheno_omnibus_results,"Male_pheno_omnibust_cluster2.csv",row.names=TRUE)
male_pheno_omnibus_results <- perform_tests_multiple_labels(df1_male, c("areal_labels"), "Cluster3")
write.csv(male_pheno_omnibus_results,"Male_pheno_omnibust_cluster3.csv",row.names=TRUE)
female_pheno_omnibus_results <- perform_tests_multiple_labels(df1_female, label_columns, "Cluster1")
write.csv(female_pheno_omnibus_results,"Female_pheno_omnibust_cluster1.csv",row.names=TRUE)
female_pheno_omnibus_results <- perform_tests_multiple_labels(df1_female, label_columns, "Cluster2")
write.csv(female_pheno_omnibus_results,"Female_pheno_omnibust_cluster2.csv",row.names=TRUE)
male_pheno_omnibus_results <- perform_tests_multiple_labels(df2_male, label_columns, "Cluster1")
write.csv(male_pheno_omnibus_results,"Male_pheno_omnibust_cluster1HBN.csv",row.names=TRUE)
male_pheno_omnibus_results <- perform_tests_multiple_labels(df2_male, label_columns, "Cluster2")
write.csv(male_pheno_omnibus_results,"Male_pheno_omnibust_cluster2HBN.csv",row.names=TRUE)
male_pheno_omnibus_results <- perform_tests_multiple_labels(df2_male, c("areal_labels"), "Cluster3")
write.csv(male_pheno_omnibus_results,"Male_pheno_omnibust_cluster3HBN.csv",row.names=TRUE)
female_pheno_omnibus_results <- perform_tests_multiple_labels(df2_female, label_columns, "Cluster1")
write.csv(female_pheno_omnibus_results,"Female_pheno_omnibust_cluster1HBN.csv",row.names=TRUE)
female_pheno_omnibus_results <- perform_tests_multiple_labels(df2_female, label_columns, "Cluster2")
write.csv(female_pheno_omnibus_results,"Female_pheno_omnibust_cluster2HBN.csv",row.names=TRUE)

label_columns <- c("areal_labels", "thickness_labels", "volume_labels", "subcort_labels")
male_pheno_omnibus_results <- perform_tests_multiple_labels(df1_male, label_columns)
write.csv(male_pheno_omnibus_results, "male_pheno_omnibus_results.csv", row.names = TRUE)
male_pheno_omnibus_results_HBN <- perform_tests_multiple_labels(df2_male, label_columns)
write.csv(male_pheno_omnibus_results_HBN, "male_pheno_omnibus_results_HBN.csv", row.names = TRUE)
female_pheno_omnibus_results_HBN <- perform_tests_multiple_labels(df2_female, label_columns)
write.csv(female_pheno_omnibus_results_HBN, "female_pheno_omnibus_results_HBN.csv", row.names = TRUE)
label_columnsX <- c("areal_labels",  "subcort_labels")
female_pheno_omnibus_results <- perform_tests_multiple_labels(df1_female, label_columnsX)
write.csv(female_pheno_omnibus_results, "female_pheno_omnibus_results.csv", row.names = TRUE)
pond <- rbind(df1_male, df1_female)
hbn <- rbind(df2_male, df2_female)
abcd <- rbind(df3_male, df3_female)
normative <- rbind(abcd_df,hcp_df)

main_stats(df1_male$age,df1_male$areal_labels)
main_stats(df1_male$age,df1_male$thickness_labels)
main_stats(df1_male$age,df1_male$volume_labels)
main_stats(df1_male$age,df1_male$subcort_labels)
main_stats(df2_male$age,df2_male$areal_labels)
main_stats(df2_male$age,df2_male$thickness_labels)
main_stats(df2_male$age,df2_male$volume_labels)
main_stats(df2_male$age,df2_male$subcort_labels)

main_stats(df1_male$SCQTOT,df1_male$areal_labels)
main_stats(df1_male$SCQTOT,df1_male$thickness_labels)
main_stats(df1_male$SCQTOT,df1_male$volume_labels)
main_stats(df1_male$SCQTOT,df1_male$subcort_labels)
main_stats(df2_male$SCQTOT,df2_male$areal_labels)
main_stats(df2_male$SCQTOT,df2_male$thickness_labels)
main_stats(df2_male$SCQTOT,df2_male$volume_labels)
main_stats(df2_male$SCQTOT,df2_male$subcort_labels)

main_stats(df1_male$ADHD_I_SUB,df1_male$areal_labels)
main_stats(df1_male$ADHD_I_SUB,df1_male$thickness_labels)
main_stats(df1_male$ADHD_I_SUB,df1_male$volume_labels)
main_stats(df1_male$ADHD_I_SUB,df1_male$subcort_labels)
main_stats(df2_male$ADHD_I_SUB,df2_male$areal_labels)
main_stats(df2_male$ADHD_I_SUB,df2_male$thickness_labels)
main_stats(df2_male$ADHD_I_SUB,df2_male$volume_labels)
main_stats(df2_male$ADHD_I_SUB,df2_male$subcort_labels)

main_stats(df1_male$ADHD_HI_SUB,df1_male$areal_labels)
main_stats(df1_male$ADHD_HI_SUB,df1_male$thickness_labels)
main_stats(df1_male$ADHD_HI_SUB,df1_male$volume_labels)
main_stats(df1_male$ADHD_HI_SUB,df1_male$subcort_labels)
main_stats(df2_male$ADHD_HI_SUB,df2_male$areal_labels)
main_stats(df2_male$ADHD_HI_SUB,df2_male$thickness_labels)
main_stats(df2_male$ADHD_HI_SUB,df2_male$volume_labels)
main_stats(df2_male$ADHD_HI_SUB,df2_male$subcort_labels)

main_stats(df1_male$FULL_IQ,df1_male$areal_labels)
main_stats(df1_male$FULL_IQ,df1_male$thickness_labels)
main_stats(df1_male$FULL_IQ,df1_male$volume_labels)
main_stats(df1_male$FULL_IQ,df1_male$subcort_labels)
main_stats(df2_male$FULL_IQ,df2_male$areal_labels)
main_stats(df2_male$FULL_IQ,df2_male$thickness_labels)
main_stats(df2_male$FULL_IQ,df2_male$volume_labels)
main_stats(df2_male$FULL_IQ,df2_male$subcort_labels)


main_stats(df1_male$CBCL_E,df1_male$areal_labels)
main_stats(df1_male$CBCL_E,df1_male$thickness_labels)
main_stats(df1_male$CBCL_E,df1_male$volume_labels)
main_stats(df1_male$CBCL_E,df1_male$subcort_labels)
main_stats(df2_male$CBCL_E,df2_male$areal_labels)
main_stats(df2_male$CBCL_E,df2_male$thickness_labels)
main_stats(df2_male$CBCL_E,df2_male$volume_labels)
main_stats(df2_male$CBCL_E,df2_male$subcort_labels)


main_stats(df1_male$CBCL_I,df1_male$areal_labels)
main_stats(df1_male$CBCL_I,df1_male$thickness_labels)
main_stats(df1_male$CBCL_I,df1_male$volume_labels)
main_stats(df1_male$CBCL_I,df1_male$subcort_labels)
main_stats(df2_male$CBCL_I,df2_male$areal_labels)
main_stats(df2_male$CBCL_I,df2_male$thickness_labels)
main_stats(df2_male$CBCL_I,df2_male$volume_labels)
main_stats(df2_male$CBCL_I,df2_male$subcort_labels)
main_stats(df2_female$age,df2_female$areal_labels)
main_stats(df2_female$age,df2_female$thickness_labels)
main_stats(df2_female$age,df2_female$volume_labels)
main_stats(df2_female$age,df2_female$subcort_labels)
main_stats(df2_male$age,df2_male$areal_labels)
main_stats(df2_male$age,df2_male$thickness_labels)
main_stats(df2_male$age,df2_male$volume_labels)
main_stats(df2_male$age,df2_male$subcort_labels)


table(df1_male$Dx,df1_male$areal_labels)
table(df1_male$Dx,df1_male$thickness_labels)
table(df1_male$Dx,df1_male$volume_labels)
table(df1_male$Dx,df1_male$subcort_labels)
table(df1_female$Dx,df1_female$areal_labels)
table(df1_female$Dx,df1_female$subcort_labels)
table(df2_male$Dx,df2_male$areal_labels)
table(df2_male$Dx,df2_male$thickness_labels)
table(df2_male$Dx,df2_male$volume_labels)
table(df2_male$Dx,df2_male$subcort_labels)
table(df2_female$Dx,df2_female$areal_labels)
table(df2_female$Dx,df2_female$subcort_labels)
table(df2_female$Dx,df2_female$thickness_labels)
table(df2_female$Dx,df2_female$volume_labels)
table(df2_female$Dx,df2_female$subcort_labels)




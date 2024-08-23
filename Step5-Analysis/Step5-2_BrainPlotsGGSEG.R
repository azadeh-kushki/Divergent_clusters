
library(ggplot2)
library(ggseg)
library(tibble)
library(scales)
library(viridis)
setwd("E:/Step5-Analysis")

plot_r_brains <- function(x, extraname, out_dir) {
  
  # Load in the data
  x$region <- gsub(extraname, "", x$region)
  shap_cur = tibble(label = x$region, effect_size = x$value)
  
  # Define the custom color scale with gray at zero
  viridis_colors <- viridis(10)
  custom_colors <- c(viridis_colors[1:5], "#7F7F7F", viridis_colors[6:10])
  
  color_scale <- scale_fill_gradientn(
    colors = custom_colors,
    values = rescale(c(-1, -0.5, 0, 0.5, 1)),
    limits = c(-1, 1)
  )
  
  # Plot with DK atlas
  plot_dk <- ggplot(shap_cur) + 
    geom_brain(color = 'black', atlas = dk, position = position_brain(hemi ~ side), aes(fill = effect_size)) + 
    color_scale + 
    theme_void()
  
  # Plot with ASEG atlas
  plot_aseg <- ggplot(shap_cur) + 
    geom_brain(color = 'black', atlas = aseg, aes(fill = effect_size)) + 
    color_scale + 
    theme_void()
  
  # Save the plots
  ggsave(filename = file.path('', paste0("brain_plot_dk_", file, ".svg")), plot = plot_dk, device = "svg")
  ggsave(filename = file.path('', paste0("brain_plot_dk_", file, ".png")), plot = plot_dk, device = "png")
  ggsave(filename = file.path('', paste0("brain_plot_aseg_", file, ".svg")), plot = plot_aseg, device = "svg")
  ggsave(filename = file.path('', paste0("brain_plot_aseg_", file, ".png")), plot = plot_aseg, device = "png")
  
  # Print the plots to the screen
  print(plot_dk)
  print(plot_aseg)
}

# Example usage
file <- 'POND_female_subcortical_cluster2_vs_reference_fe.csv'
outf <- 'POND_female_subcortical_cluster2_vs_reference_fe'

extraname <- "_combatted"
x <- read.csv(file)
x$region <- gsub("\\.", "-", x$region)


plot_r_brains(x, extraname, outf)

library(raster)
library(dismo)
library(dplyr)
library(pROC)
library(classInt)
library(rstudioapi)
library(data.table)
library(tidyverse)
library(rjson)
library(yaml)
library(rJava)
set.seed(42)

source('maxent_functions.r')

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

r_start <- as.numeric(args[1])
# r_start <- 1
r_end <- r_start + 29

# specify run_id
run_id <- '2f07dfdd1bef43988e07f22cb2e322c5'

# create folder 
dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_png <- file.path(dir_base_run_id, 'png', 'all')
create_folder(dir_run_id_png)
dir_run_id_tif <- file.path(dir_base_run_id, 'tif', 'all')
create_folder(dir_run_id_tif)
dir_run_id_env_contribution <- file.path(dir_base_run_id, 'env_contribution', 'all')
create_folder(dir_run_id_env_contribution)

# load DeepSDM model configurations
DeepSDM_conf_path <- file.path('predicts', run_id, 'DeepSDM_conf.yaml')
DeepSDM_conf <- yaml.load_file(DeepSDM_conf_path)
env_list <- sort(DeepSDM_conf$training_conf$env_list)

# extent_binary -> 1:prediction area (land); 0:non-prediction area (sea)
# trainval_split -> 1:training split; NA: validation split
extent_binary <- raster(DeepSDM_conf$geo_extent_file)
trainval_split <- raster(file.path('tmp', 'DeepSDM DEMO', '20230911141642-0_partition_extent.tif'))
i_extent <- which(values(extent_binary) == 1) # cell index of prediction area
i_trainsplit <- which(values(trainval_split) == 1) # cell index of training split
i_valsplit <- which(is.na(values(trainval_split))) # cell index of validation split

# load environmental information 
env_info <- fromJSON(file = DeepSDM_conf$meta_json_files$env_inf)

# load filtered csv from 01_prepare_data.ipynb
# sp_occ_filter <- read.csv('workspace/species_data/occurrence_data/species_occurrence_filter.csv')

# load species information
sp_info <- fromJSON(file = DeepSDM_conf$meta_json_files$sp_inf)

# make date_list for prediction
date_list <- DeepSDM_conf$training_conf$date_list_predict
date_list_all <- DeepSDM_conf$training_conf$date_list_train

# make species_list for prediction
species_list <- DeepSDM_conf$training_conf$species_list_train

# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df_all_season <- data.frame(sptime = character(),
                            maxent_all_season_val = numeric(), maxent_all_season_train = numeric(), maxent_all_season_all = numeric(), 
                            p_season = numeric(), p_valpart_season = numeric(), p_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_trainpart_season = numeric())
df_all_all <- data.frame(sptime = character(),
                         maxent_all_all_val = numeric(), maxent_all_all_train = numeric(), maxent_all_all_all = numeric(), 
                         p_all = numeric(), p_valpart_all = numeric(), p_trainpart_all = numeric(), pa_valpart_all = numeric(), pa_trainpart_all = numeric())

#plotting colormap
color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')

env_all <- load_env_allseason(env_list, env_info, date_list_all, DeepSDM_conf)
for(species in species_list[r_start:min(r_end, length(species_list))]){
  # species <- species_list[1]
  generate_points_all()
  
  set_default_variable_all()
  
  # set the prediction result does not exists
  px_all_all_exists <- FALSE
  xm_all <- try(maxent(x = env_all, p = xy_p_all_trainsplit, a = xy_pa_all_sample_trainsplit), silent = T)
  if(!is.character(xm_all)){
    write.csv(xm_all@results,
              file.path(dir_run_id_env_contribution, sprintf('%s_env_contribution_maxentall.csv', species)))
    px_all_all <- predict_maxent(env_all, xm_all)
    px_all_all_exists <- TRUE
    plot_result(species, px_all_all, extent_binary, xy_p_all, 'maxent_all_all', dir_run_id_png, dir_run_id_tif, run_id)
    maxent_all_all_train <- calculate_roc(px_all_all, xy_p_all_trainsplit, xy_pa_all_sample_trainsplit)
    maxent_all_all_val <- calculate_roc(px_all_all, xy_p_all_valsplit, xy_pa_all_sample_valsplit)
    maxent_all_all_all <- calculate_roc(px_all_all, xy_p_all, xy_pa_all_sample)
  }
  
  df_all_all[nrow(df_all_all)+1, ] <- c(species, 
                                        maxent_all_all_val, maxent_all_all_train, maxent_all_all_all, 
                                        p_all, p_valpart_all, p_trainpart_all, pa_valpart_all, pa_trainpart_all)
  for(time in date_list){
    # time <- date_list[1]
    print(paste('start', species, time))
    sp_season <- paste0(species, '_', time)
    
    if(px_all_all_exists){
      
      generate_points()
      
      # set variables as default values
      set_default_variable()
      
      # load env layers of season
      env_season <- load_env_season(env_list, env_info, time, DeepSDM_conf)
      
      # maxent
      # check if maxent predictions have existed
      maxent_path <- file.path(dir_run_id_tif, sprintf('%s_maxent_all_season_%s.tif', sp_season, run_id))
      maxent_exists <- file.exists(maxent_path)
      if(maxent_exists){
        px_all_season <- raster::raster(maxent_path)
        plot_result(sp_season, px_all_season, extent_binary, xy_p_season, 'maxent_all_season', dir_run_id_png, dir_run_id_tif, run_id)
        maxent_all_season_train <- calculate_roc(px_all_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
        maxent_all_season_val <- calculate_roc(px_all_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
        maxent_all_season_all <- calculate_roc(px_all_season, xy_p_season, xy_pa_season_sample)
      }else{
        px_all_season <- predict_maxent(env_season, xm_all)
        plot_result(sp_season, px_all_season, extent_binary, xy_p_season, 'maxent_all_season', dir_run_id_png, dir_run_id_tif, run_id)
        maxent_all_season_train <- calculate_roc(px_all_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
        maxent_all_season_val <- calculate_roc(px_all_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
        maxent_all_season_all <- calculate_roc(px_all_season, xy_p_season, xy_pa_season_sample)
      }
    }
    df_all_season[nrow(df_all_season)+1, ] <- c(sp_season, 
                                                maxent_all_season_val, maxent_all_season_train, maxent_all_season_all, 
                                                p_season, p_valpart_season, p_trainpart_season, pa_valpart_season, pa_trainpart_season)
  }
}

output_csv_all_all_path <- file.path(dir_base_run_id, sprintf('auc_result_all_all_%s.csv', r_start))
output_csv_all_season_path <- file.path(dir_base_run_id, sprintf('auc_result_all_season_%s.csv', r_start))
write.csv(df_all_all, output_csv_all_all_path, row.names = FALSE)
write.csv(df_all_season, output_csv_all_season_path, row.names = FALSE)

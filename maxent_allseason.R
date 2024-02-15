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
r_end <- r_start + 2

# specify run_id
run_id <- '2f07dfdd1bef43988e07f22cb2e322c5'

# create folder 
dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_png <- file.path(dir_base_run_id, 'png', 'allseason')
create_folder(dir_run_id_png)
dir_run_id_tif <- file.path(dir_base_run_id, 'tif', 'allseason')
create_folder(dir_run_id_tif)
dir_run_id_env_contribution <- file.path(dir_base_run_id, 'env_contribution', 'allseason')
create_folder(dir_run_id_env_contribution)
dir_run_id_xm <- file.path(dir_base_run_id, 'xm', 'allseason')
create_folder(dir_run_id_xm)

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
allseason_list <- unique(substr(date_list, 6, 10))

# make species_list for prediction
species_list <- DeepSDM_conf$training_conf$species_list_train

# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df_allseason_season <- data.frame(species = character(), time = character(), 
                                  maxent_allseason_season_val = numeric(), maxent_allseason_season_train = numeric(), maxent_allseason_season_all = numeric(), 
                                  p_season = numeric(), p_valpart_season = numeric(), p_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_trainpart_season = numeric())
df_allseason_allseason <- data.frame(sptime = character(),
                                     maxent_allseason_allseason_val = numeric(), maxent_allseason_allseason_train = numeric(), maxent_allseason_allseason_all = numeric(), 
                                     p_allseason = numeric(), p_valpart_allseason = numeric(), p_trainpart_allseason = numeric(), pa_valpart_allseason = numeric(), pa_trainpart_allseason = numeric())

#plotting colormap
color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')

for(allseason in allseason_list[r_start:r_end]){
  # allseason <- allseason_list[1]
  # load env layers of season
  date_list_all_selectseason <- date_list_all[which(substr(date_list_all, 6, 10) == allseason)]
  date_list_selectseason <- date_list[which(substr(date_list, 6, 10) == allseason)]
  
  env_allseason <- load_env_allseason(env_list, env_info, date_list_all_selectseason, DeepSDM_conf)
  
  for(species in species_list){
    # species <- species_list[1]
    sp_allseason <- paste0(species, '_', allseason)
    
    generate_points_allseason()
    
    set_default_variable_allseason()
    
    # set the prediction result does not exists
    px_allseason_allseason <- NULL
    
    # check if the maxent process has been conducted
    xm_path <- file.path(dir_run_id_xm, sprintf('%s_xm.RData', sp_allseason))
    xm_exists <- file.exists(xm_path)
    
    if(xm_exists){
      load(xm_path)
      px_allseason_allseason <- predict_maxent(env_allseason, xm_allseason)
      plot_result(sp_allseason, px_allseason_allseason, extent_binary, xy_p_allseason, 'maxent_allseason_allseason', dir_run_id_png, dir_run_id_tif, run_id)
      maxent_allseason_allseason_train <- calculate_roc(px_allseason_allseason, xy_p_allseason_trainsplit, xy_pa_allseason_sample_trainsplit)
      maxent_allseason_allseason_val <- calculate_roc(px_allseason_allseason, xy_p_allseason_valsplit, xy_pa_allseason_sample_valsplit)
      maxent_allseason_allseason_all <- calculate_roc(px_allseason_allseason, xy_p_allseason, xy_pa_allseason_sample)
    }else{
      xm_allseason <- try(maxent(x = env_allseason, p = xy_p_allseason_trainsplit, a = xy_pa_allseason_sample_trainsplit), silent = T)
      if(!is.character(xm_allseason)){
        write.csv(xm_allseason@results,
                  file.path(dir_run_id_env_contribution, sprintf('%s_env_contribution_maxentallseason.csv', sp_allseason)))
        save(xm_allseason, file = xm_path)
        px_allseason_allseason <- predict_maxent(env_allseason, xm_allseason)
        plot_result(sp_allseason, px_allseason_allseason, extent_binary, xy_p_allseason, 'maxent_allseason_allseason', dir_run_id_png, dir_run_id_tif, run_id)
        maxent_allseason_allseason_train <- calculate_roc(px_allseason_allseason, xy_p_allseason_trainsplit, xy_pa_allseason_sample_trainsplit)
        maxent_allseason_allseason_val <- calculate_roc(px_allseason_allseason, xy_p_allseason_valsplit, xy_pa_allseason_sample_valsplit)
        maxent_allseason_allseason_all <- calculate_roc(px_allseason_allseason, xy_p_allseason, xy_pa_allseason_sample)
      }
    }


    df_allseason_allseason[nrow(df_allseason_allseason)+1, ] <- c(sp_allseason, 
                                                                  maxent_allseason_allseason_val, maxent_allseason_allseason_train, maxent_allseason_allseason_all, 
                                                                  p_allseason, p_valpart_allseason, p_trainpart_allseason, pa_valpart_allseason, pa_trainpart_allseason)
    for(time in date_list_selectseason){
      # time <- date_list_selectseason[1]
      print(paste('start', species, time))
      sp_season <- paste0(species, '_', time)
      
      if(!is.null(px_allseason_allseason)){
        
        generate_points()
        
        # set variables as default values
        set_default_variable()
        
        # load env layers of season
        env_season <- load_env_season(env_list, env_info, time, DeepSDM_conf)
        
        # maxent
        # check if maxent predictions have existed
        maxent_path <- file.path(dir_run_id_tif, sprintf('%s_maxent_allseason_season_%s.tif', sp_season, run_id))
        maxent_exists <- file.exists(maxent_path)
        if(maxent_exists){
          px_allseason_season <- raster::raster(maxent_path)
          plot_result(sp_season, px_allseason_season, extent_binary, xy_p_season, 'maxent_allseason_season', dir_run_id_png, dir_run_id_tif, run_id)
          maxent_allseason_season_train <- calculate_roc(px_allseason_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
          maxent_allseason_season_val <- calculate_roc(px_allseason_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
          maxent_allseason_season_all <- calculate_roc(px_allseason_season, xy_p_season, xy_pa_season_sample)
        }else{
          px_allseason_season <- predict_maxent(env_season, xm_allseason)
          plot_result(sp_season, px_allseason_season, extent_binary, xy_p_season, 'maxent_allseason_season', dir_run_id_png, dir_run_id_tif, run_id)
          maxent_allseason_season_train <- calculate_roc(px_allseason_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
          maxent_allseason_season_val <- calculate_roc(px_allseason_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
          maxent_allseason_season_all <- calculate_roc(px_allseason_season, xy_p_season, xy_pa_season_sample)
        }
      }
      df_allseason_season[nrow(df_allseason_season)+1, ] <- c(species, time, 
                                                              maxent_allseason_season_val, maxent_allseason_season_train, maxent_allseason_season_all, 
                                                              p_season, p_valpart_season, p_trainpart_season, pa_valpart_season, pa_trainpart_season)
    }
  }
}
output_csv_allseason_allseason_path <- file.path(dir_base_run_id, sprintf('auc_result_allseason_allseason_%s.csv', r_start))
output_csv_allseason_season_path <- file.path(dir_base_run_id, sprintf('auc_result_allseason_season_%s.csv', r_start))
write.csv(df_allseason_allseason, output_csv_allseason_allseason_path, row.names = FALSE)
write.csv(df_allseason_season, output_csv_allseason_season_path, row.names = FALSE)

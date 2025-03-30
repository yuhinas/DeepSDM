# -*- coding: utf-8 -*-
library(hdf5r)
library(raster)
library(dismo)
library(pROC)
library(tidyverse)
library(rjson)
library(yaml)
library(rJava)
set.seed(42)

source('maxent_functions.R')

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

# specify run_id, exp_id
run_id <- 'e52c8ac9a3e24c75ac871f63bbdea060'
exp_id <- '115656750127464383'

# create folder 
dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_png <- file.path(dir_base_run_id, 'png', 'all')
create_folder(dir_run_id_png)
dir_run_id_h5 <- file.path(dir_base_run_id, 'h5', 'all')
create_folder(dir_run_id_h5)
dir_run_id_env_contribution <- file.path(dir_base_run_id, 'env_contribution', 'all')
create_folder(dir_run_id_env_contribution)
dir_maxent_model <- file.path(dir_base_run_id, 'maxent_model')
create_folder(dir_maxent_model)

# load DeepSDM model configurations
DeepSDM_conf_path <- file.path('predicts', run_id, 'DeepSDM_conf.yaml')
DeepSDM_conf <- yaml.load_file(DeepSDM_conf_path)
env_list <- sort(DeepSDM_conf$training_conf$env_list)

# extent_binary -> 1:prediction area (land); 0:non-prediction area (sea)
# trainval_split -> 1:training split; NA: validation split
extent_binary <- raster(DeepSDM_conf$geo_extent_file)
trainval_split <- raster(file.path('mlruns', exp_id, run_id, 'artifacts', 'extent_binary', 'partition_extent.tif'))
i_extent <- which(values(extent_binary) == 1) # cell index of prediction area
i_trainsplit <- which(values(trainval_split) == 1) # cell index of training split
i_valsplit <- which(is.na(values(trainval_split))) # cell index of validation split

# load environmental information 
env_info_path <- file.path('predicts', run_id, 'env_inf.json')
env_info <- fromJSON(file = env_info_path)

# load filtered csv from 01_prepare_data.ipynb
# sp_occ_filter <- read.csv('workspace/species_data/occurrence_data/species_occurrence_filter.csv')

# load species information
sp_info_path <- file.path('predicts', run_id, 'sp_inf.json')
sp_info <- fromJSON(file = sp_info_path)

# make date_list for prediction
date_list_predict <- DeepSDM_conf$training_conf$date_list_predict

# make date_list for training
date_list_train <- DeepSDM_conf$training_conf$date_list_train

# make species_list for prediction
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df_all_season <- data.frame(spdate = character(),
                            maxent_all_season_val = numeric(), maxent_all_season_train = numeric(), maxent_all_season_all = numeric(), 
                            deepsdm_all_season_val = numeric(), deepsdm_all_season_train = numeric(), deepsdm_all_season_all = numeric(), 
                            maxent_TSS = numeric(), deepsdm_TSS = numeric(), maxent_kappa = numeric(), deepsdm_kappa = numeric(), maxent_f1 = numeric(), deepsdm_f1 = numeric(), 
                            p_season = numeric(), p_valpart_season = numeric(), p_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_trainpart_season = numeric())
df_all_all <- data.frame(species = character(),
                         maxent_all_all_val = numeric(), maxent_all_all_train = numeric(), maxent_all_all_all = numeric(), 
                         p_all = numeric(), p_valpart_all = numeric(), p_trainpart_all = numeric(), pa_valpart_all = numeric(), pa_trainpart_all = numeric())

#plotting colormap
color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')

env_all <- load_env_allseason(env_list, env_info, date_list_train, DeepSDM_conf)

r_start <- as.numeric(args[1])
# r_start <- 121
r_end <- r_start + 2
for(species in species_list[r_start:min(r_end, length(species_list))]){
  # species <- species_list[121]
  # species <- 'Alauda_gulgula'
  # set the prediction result does not exists
  
  dir_run_id_png_sp <- file.path(dir_run_id_png, species)
  create_folder(dir_run_id_png_sp)
  dir_run_id_h5_sp <- file.path(dir_run_id_h5, species)
  create_folder(dir_run_id_h5_sp)
    
  result <- tryCatch({
    generate_points_all(date_list_train)  # 呼叫函數
    TRUE  # 如果成功，返回 TRUE
  }, error = function(e) {
    message("Error encountered: ", e$message)  # 顯示錯誤訊息
    FALSE  # 如果出現錯誤，返回 FALSE
  })
  # 如果函數發生錯誤，跳過這次的 loop
  if (!result) {
    message("Skipping this iteration due to an error.")
    next 
  }
    
  # set default values
  set_default_variable_all()
    
  maxent_all_all_exists <- FALSE
  maxent_h5_path <- file.path(dir_run_id_h5_sp, sprintf('%s.h5', species))
  maxent_all_all_exists <- check_dataset_in_h5(maxent_h5_path, 'all')

  if(!maxent_all_all_exists){
      xm_all <- try(maxent(x = env_all, p = xy_p_all_trainsplit, a = xy_pa_all_sample_trainsplit), silent = T)
      save(xm_all, file = file.path(dir_maxent_model, sprintf('%s_all.RData', species)))
      if(!is.character(xm_all)){
        write.csv(xm_all@results,
                  file.path(dir_run_id_env_contribution, sprintf('%s_env_contribution_maxentall.csv', species)))
        maxent_all_all <- predict_maxent(env_all, xm_all)
        maxent_all_all_exists <- TRUE
        plot_result(species, maxent_all_all, extent_binary, xy_p_all, 'maxent_all_all', dir_run_id_png_sp, dir_run_id_h5_sp, run_id)
        maxent_all_all_train <- calculate_roc(maxent_all_all, xy_p_all_trainsplit, xy_pa_all_sample_trainsplit)
        maxent_all_all_val <- calculate_roc(maxent_all_all, xy_p_all_valsplit, xy_pa_all_sample_valsplit)
        maxent_all_all_all <- calculate_roc(maxent_all_all, xy_p_all, xy_pa_all_sample)
      }
  }else{      
    maxent_all_all <- h5dataset_to_raster(maxent_h5_path, 'all')
    maxent_all_all_exists <- TRUE
    load(file.path(dir_maxent_model, sprintf('%s_all.RData', species)))
    maxent_all_all_train <- calculate_roc(maxent_all_all, xy_p_all_trainsplit, xy_pa_all_sample_trainsplit)
    maxent_all_all_val <- calculate_roc(maxent_all_all, xy_p_all_valsplit, xy_pa_all_sample_valsplit)
    maxent_all_all_all <- calculate_roc(maxent_all_all, xy_p_all, xy_pa_all_sample)
  }
  df_all_all[nrow(df_all_all)+1, ] <- c(species, 
                                        maxent_all_all_val, maxent_all_all_train, maxent_all_all_all, 
                                        p_all, p_valpart_all, p_trainpart_all, pa_valpart_all, pa_trainpart_all)
    
  for(date in date_list_predict){
    # date <- date_list_predict[1]
    # date <- '2018-12-01'
    print(paste('start', species, date))
    sp_season <- paste0(species, '_', date)
    set_default_variable()
      
    maxent_all_season_exists <- FALSE
    
    if(maxent_all_all_exists){
      
      generate_points(num_pa = 'num_p')
      
      # set variables as default values
      set_default_variable()
        
      # load env layers of season
      env_season <- load_env_season(env_list, env_info, date, DeepSDM_conf)
        
      # maxent
      # check if maxent predictions have existed
      maxent_all_season_exists <- check_dataset_in_h5(maxent_h5_path, date)
      if(maxent_all_season_exists){
        maxent_all_season <- h5dataset_to_raster(maxent_h5_path, date)
      }else{
          maxent_all_season <- predict_maxent(env_season, xm_all)
          maxent_all_season_exists <- TRUE
          plot_result(species, maxent_all_season, extent_binary, xy_p_season, 'maxent_all_season', dir_run_id_png_sp, dir_run_id_h5_sp, run_id, date)
      }
      maxent_all_season_train <- calculate_roc(maxent_all_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
      maxent_all_season_val <- calculate_roc(maxent_all_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
      maxent_all_season_all <- calculate_roc(maxent_all_season, xy_p_season, xy_pa_season_sample)
      maxent_other_indicator <- calculate_indicator(maxent_all_season)
      maxent_TSS <- maxent_other_indicator[1]
      maxent_kappa <- maxent_other_indicator[2]
      maxent_f1 <- maxent_other_indicator[3]
      maxent_threshold <- maxent_other_indicator[4]
        
      # deepsdm
      deepsdm_h5_path <- file.path('predicts', run_id, 'h5', species, sprintf('%s.h5', species))
      deepsdm_all_season_exists <- check_dataset_in_h5(deepsdm_h5_path, date)
      
      if (deepsdm_all_season_exists){
        deepsdm_all_season <- h5dataset_to_raster(deepsdm_h5_path, date)
        plot_result_deepsdm(sp_season, deepsdm_all_season, extent_binary, xy_p_season, 'deepsdm_all_season', dir_run_id_png_sp, run_id)
        deepsdm_all_season_train <- try(calculate_roc(deepsdm_all_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit))
        deepsdm_all_season_val <- try(calculate_roc(deepsdm_all_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit))
        deepsdm_all_season_all <- try(calculate_roc(deepsdm_all_season, xy_p_season, xy_pa_season_sample))
        deepsdm_other_indicator <- calculate_indicator(deepsdm_all_season)
        deepsdm_TSS <- deepsdm_other_indicator[1]
        deepsdm_kappa <- deepsdm_other_indicator[2]
        deepsdm_f1 <- deepsdm_other_indicator[3]
        deepsdm_threshold <- deepsdm_other_indicator[4]
      }
    }
    df_all_season[nrow(df_all_season)+1, ] <- c(sp_season,
                                                maxent_all_season_val, maxent_all_season_train, maxent_all_season_all, 
                                                deepsdm_all_season_val, deepsdm_all_season_train, deepsdm_all_season_all, 
                                                maxent_TSS, deepsdm_TSS, maxent_kappa, deepsdm_kappa, maxent_f1, deepsdm_f1, 
                                                p_season, p_valpart_season, p_trainpart_season, pa_valpart_season, pa_trainpart_season)
  }
}

output_csv_all_season_path <- file.path(dir_base_run_id, sprintf('all_indicator_result_all_season_num_pa%03d.csv', r_start))
write.csv(df_all_season, output_csv_all_season_path, row.names = FALSE)

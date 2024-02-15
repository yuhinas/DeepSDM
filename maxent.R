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
r_end <- r_start + 3071

# specify run_id
run_id <- '2f07dfdd1bef43988e07f22cb2e322c5'

# create folder 
dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_png <- file.path(dir_base_run_id, 'png', 'season')
create_folder(dir_run_id_png)
dir_run_id_tif <- file.path(dir_base_run_id, 'tif', 'season')
create_folder(dir_run_id_tif)
dir_run_id_env_contribution <- file.path(dir_base_run_id, 'env_contribution', 'season')
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
date_list <- DeepSDM_conf$training_conf$date_list_train

# make species_list for prediction
species_list <- DeepSDM_conf$training_conf$species_list_train

# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df <- data.frame(sptime = character(),
                 maxent_season_season_val = numeric(), deepsdm_all_season_val = numeric(), 
                 maxent_season_season_train = numeric(), deepsdm_all_season_train = numeric(), 
                 maxent_season_season_all = numeric(), deepsdm_all_season_all = numeric(), 
                 p_season = numeric(), p_valpart_season = numeric(), p_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_trainpart_season = numeric())


# create 'files' by DeepSDM_conf
files <- outer(species_list, date_list, FUN=function(sp, date) {
  sprintf('%s_%s_predict.tif', sp, date)
})
files <- as.vector(files)

for(f in files[r_start:min(r_end, length(files))]){
  # f <- 'Glaucidium_brodiei_2018-04-01_predict.tif'
  # f <- 'sp13_2017-12-01_predict.tif'
  # f <- files[1]
  # f <- 'Phoenicurus_fuliginosus_2009-01-01_predict.tif'
  print(paste('start', f))
  f_split <- (f %>% strsplit('_'))[[1]]
  species <- paste(f_split[1:(length(f_split)-2)], collapse = '_')
  sptime <- paste(f_split[1:(length(f_split)-1)], collapse = '_')
  time <- f_split[length(f_split)-1]
  
  # load env layers of season
  env_season <- load_env_season(env_list, env_info, time, DeepSDM_conf)
  
  generate_points()
  
  # set variables as default values
  set_default_variable()
  
  #plotting
  color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')
  
  # maaxent
  # check if maxent predictions have existed
  maxent_path <- file.path(dir_run_id_tif, sprintf('%s_maxent_season_season_%s.tif', sptime, run_id))
  maxent_exists <- file.exists(maxent_path)
  if(maxent_exists){
    px_season_season <- raster::raster(maxent_path)
    maxent_season_season_train <- calculate_roc(px_season_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
    maxent_season_season_val <- calculate_roc(px_season_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
    maxent_season_season_all <- calculate_roc(px_season_season, xy_p_season, xy_pa_season_sample)
  }else{
    xm_season <- try(maxent(x = env_season, p = xy_p_season_trainsplit, a = xy_pa_season_sample_trainsplit), silent = T)
    if(!is.character(xm_season)){
      
      write.csv(xm_season@results,
                file.path(dir_run_id_env_contribution, sprintf('%s_env_contribution_maxentseason.csv', sptime)))
      
      px_season_season <- predict_maxent(env_season, xm_season)
      
      plot_result(sptime, px_season_season, extent_binary, xy_p_season, 'maxent_season_season', dir_run_id_png, dir_run_id_tif, run_id)
      maxent_season_season_train <- calculate_roc(px_season_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
      maxent_season_season_val <- calculate_roc(px_season_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
      maxent_season_season_all <- calculate_roc(px_season_season, xy_p_season, xy_pa_season_sample)
    }
  }

  # deepsdm
  deepsdm_path <- file.path('predicts', run_id, 'tif', f)
  deepsdm <- try(raster::raster(deepsdm_path), silent = TRUE)
  if (!is.character(deepsdm)){
    deepsdm_all_season_train <- try(calculate_roc(deepsdm, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit))
    deepsdm_all_season_val <- try(calculate_roc(deepsdm, xy_p_season_valsplit, xy_pa_season_sample_valsplit))
    deepsdm_all_season_all <- try(calculate_roc(deepsdm, xy_p_season, xy_pa_season_sample))
    plot_result_deepsdm(sptime, deepsdm, extent_binary, 'deepsdm_all_season', dir_run_id_png, run_id)
  }

  
  df[nrow(df)+1, ] <- c(sptime,
                        maxent_season_season_val, deepsdm_all_season_val, 
                        maxent_season_season_train, deepsdm_all_season_train, 
                        maxent_season_season_all, deepsdm_all_season_all, 
                        p_season, p_valpart_season, p_trainpart_season, pa_valpart_season, pa_trainpart_season)
}

output_csv_path <- file.path(dir_base_run_id, sprintf('auc_result_season_season_%s.csv', r_start))
write.csv(df, output_csv_path, row.names = FALSE)
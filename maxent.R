set.seed(42)
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

source('maxent_functions.r')

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

r_start <- as.numeric(args[1])
# r_start <- 1
r_end <- r_start + 199

# specify run_id
run_id <- 'a10a969313594b52b2c405aae580f2a6'

# create folder 
dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_png <- file.path(dir_base_run_id, 'png')
create_folder(dir_run_id_png)
dir_run_id_tif <- file.path(dir_base_run_id, 'tif')
create_folder(dir_run_id_tif)
dir_run_id_env_contribution <- file.path(dir_base_run_id, 'env_contribution')
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
date_list_predict <- DeepSDM_conf$training_conf$date_list_predict

# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df <- data.frame(sptime = character(),
                 maxent_season_season_val = numeric(), maxent_season_seasonavg_val = numeric(), maxent_season_all_val = numeric(),
                 maxent_seasonavg_season_val = numeric(), maxent_seasonavg_seasonavg_val = numeric(), maxent_seasonavg_all_val = numeric(),
                 maxent_all_season_val = numeric(), maxent_all_seasonavg_val = numeric(), maxent_all_all_val = numeric(),
                 deepsdm_all_season_val = numeric(), deepsdm_all_seasonavg_val = numeric(), deepsdm_all_all_val = numeric(),
                 maxent_season_season_train = numeric(), maxent_season_seasonavg_train = numeric(), maxent_season_all_train = numeric(),
                 maxent_seasonavg_season_train = numeric(), maxent_seasonavg_seasonavg_train = numeric(), maxent_seasonavg_all_train = numeric(),
                 maxent_all_season_train = numeric(), maxent_all_seasonavg_train = numeric(), maxent_all_all_train = numeric(),
                 deepsdm_all_season_train = numeric(), deepsdm_all_seasonavg_train = numeric(), deepsdm_all_all_train = numeric(),
                 p_season = numeric(), p_valpart_season = numeric(), p_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_trainpart_season = numeric(),
                 p_seasonavg = numeric(), p_valpart_seasonavg = numeric(), p_trainpart_seasonavg = numeric(), pa_valpart_seasonavg = numeric(), pa_trainpart_seasonavg = numeric(),
                 p_all = numeric(), p_valpart_all = numeric(), p_trainpart_all = numeric(), pa_valpart_all = numeric(), pa_trainpart_all = numeric())

files_path <- file.path('predicts', run_id, 'tif')
files <- list.files(files_path)
files <- sort(files)

for(f in files[r_start:min(r_end, length(files))]){
  # f <- 'Strix_nivicolum_2018-01-01_predict.tif'
  # f <- 'sp18_2018-09-01_predict.tif'
  # f <- files[1]
  print(paste('start', f))
  f_split <- (f %>% strsplit('_'))[[1]]
  species <- paste(f_split[1:(length(f_split)-2)], collapse = '_')
  sptime <- paste(f_split[1:(length(f_split)-1)], collapse = '_')
  time <- f_split[length(f_split)-1]
  
  # load env layers of season
  env_season <- load_env_season(env_list, env_info, time)
  
  occ_rst_path <- file.path(sp_info$dir_base, sp_info$file_name[[species]][[time]])
  occ_rst <- raster::raster(occ_rst_path)
  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_season <- xyFromCell(occ_rst, i_p_occ_rst) # x,y value from cells with presence records
  xy_p_season_trainsplit <- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_season_valsplit <- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))
  
  i_pa_season <- intersect(i_pa_occ_rst, i_extent)
  xy_pa_season <- xyFromCell(occ_rst, i_pa_season)
  i_pa_season_sample <- sample(i_pa_season, 10000)
  xy_pa_season_sample_trainsplit <- xyFromCell(occ_rst, intersect(i_pa_season_sample, i_trainsplit))
  xy_pa_season_sample_valsplit <- xyFromCell(occ_rst, intersect(i_pa_season_sample, i_valsplit))
  
  # set variables as default values
  set_default_variable()
  p_season <- nrow(xy_p_season)
  p_valpart_season <- nrow(xy_p_season_valsplit)
  p_trainpart_season <- nrow(xy_p_season_trainsplit)
  pa_valpart_season <- nrow(xy_pa_season_sample_valsplit)
  pa_trainpart_season <- nrow(xy_pa_season_sample_trainsplit)
  
  
  #plotting
  color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')
  xm_season <- try(maxent(x = env_season, p = xy_p_season_trainsplit, a = xy_pa_season_sample_trainsplit), silent = T)
  if(!is.character(xm_season)){

    write.csv(xm_season@results,
              file.path(dir_run_id_env_contribution, sprintf('%s_env_contribution_maxentseason.csv', sptime)))

    px_season_season <- predict_maxent(env_season, xm_season)

    plot_result(sptime, species, px_season_season, extent_binary, xy_p_season, xy_p_season, xy_p_season, 'maxent_season_season', dir_run_id_png, dir_run_id_tif, run_id)
    maxent_season_season_train <- calculate_roc(px_season_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
    maxent_season_season_val <- calculate_roc(px_season_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
  }
  # if maxent predictions have existed, run the code below
  #------------------------------------------------------------------------------------
  # maxent_path <- file.path(dir_run_id_tif, sprintf('%s_maxent_season_season_%s.tif', sptime, run_id))
  # if(file.exists(maxent_path)){
  #   px_season_season <- raster::raster(maxent_path)
  #   maxent_season_season_train <- calculate_roc(px_season_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
  #   maxent_season_season_val <- calculate_roc(px_season_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
  # }

  deepsdm_path <- file.path('predicts', run_id, 'tif', f)
  deepsdm <- raster::raster(deepsdm_path)
  # deepsdmall_all_train <- try(calculate_roc(deepsdm, p_trainall, bg_trainall))
  # deepsdmall_all_test <- try(calculate_roc(deepsdm, p_testall, bg_testall))
  # deepsdmall_12_train <- try(calculate_roc(deepsdm, p_train12, bg_train12))
  # deepsdmall_12_test <- try(calculate_roc(deepsdm, p_test12, bg_test12))
  deepsdm_all_season_train <- try(calculate_roc(deepsdm, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit))
  deepsdm_all_season_val <- try(calculate_roc(deepsdm, xy_p_season_valsplit, xy_pa_season_sample_valsplit))
  # plot_result_deepsdm(sptime, species, deepsdm, extent_binary, xy_p_season, xy_p_season, xy_p_season, 'deepsdm_all_season', dir_run_id_png, run_id)
  
  df[nrow(df)+1, ] <- c(sptime,
                        maxent_season_season_val, maxent_season_seasonavg_val, maxent_season_all_val,
                        maxent_seasonavg_season_val, maxent_seasonavg_seasonavg_val, maxent_seasonavg_all_val,
                        maxent_all_season_val, maxent_all_seasonavg_val, maxent_all_all_val,
                        deepsdm_all_season_val, deepsdm_all_seasonavg_val, deepsdm_all_all_val,
                        maxent_season_season_train, maxent_season_seasonavg_train, maxent_season_all_train,
                        maxent_seasonavg_season_train, maxent_seasonavg_seasonavg_train, maxent_seasonavg_all_train,
                        maxent_all_season_train, maxent_all_seasonavg_train, maxent_all_all_train,
                        deepsdm_all_season_train, deepsdm_all_seasonavg_train, deepsdm_all_all_train,
                        p_season, p_valpart_season, p_trainpart_season, pa_valpart_season, pa_trainpart_season,
                        p_seasonavg, p_valpart_seasonavg, p_trainpart_seasonavg, pa_valpart_seasonavg, pa_trainpart_seasonavg,
                        p_all, p_valpart_all, p_trainpart_all, pa_valpart_all, pa_trainpart_all)
}

output_csv_path <- file.path(dir_base_run_id, sprintf('auc_result_%s.csv', r_start))
write.csv(df, output_csv_path, row.names = FALSE)
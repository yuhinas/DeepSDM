library(raster)
library(dplyr)
library(pROC)
library(dismo)
library(yaml)
library(rjson)
library(stringr)
set.seed(42)

source('f1_functions.r')

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

r_start <- as.numeric(args[1])
# r_start <- 1
r_end <- r_start + 2

# specify run_id
run_id <- '2f07dfdd1bef43988e07f22cb2e322c5'

# load DeepSDM model configurations
DeepSDM_conf_path <- file.path('predicts', run_id, 'DeepSDM_conf.yaml')
DeepSDM_conf <- yaml.load_file(DeepSDM_conf_path)
env_list <- sort(DeepSDM_conf$training_conf$env_list)

# extent_binary -> 1:prediction area (land); 0:non-prediction area (sea)
# trainval_split -> 1:training split; NA: validation split
extent_binary <- raster(DeepSDM_conf$geo_extent_file)
trainval_split <- raster('tmp/DeepSDM DEMO/20230911141642-0_partition_extent.tif')
i_extent <- which(values(extent_binary) == 1) # cell index of prediction area
i_trainsplit <- which(values(trainval_split) == 1) # cell index of training split
i_valsplit <- which(is.na(values(trainval_split))) # cell index of validation split

# load species information
sp_info <- fromJSON(file = DeepSDM_conf$meta_json_files$sp_inf)

# load environmental information 
env_info <- fromJSON(file = DeepSDM_conf$meta_json_files$env_inf)

# make date_list for prediction
date_list <- DeepSDM_conf$training_conf$date_list_train
date_list_all <- DeepSDM_conf$training_conf$date_list_train
allseason_list <- unique(substr(date_list_all, 6, 10))

# make species_list for prediction
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

# season: specific months in one year
# allseason: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df_allseason_allseason <- data.frame(sptime = character(), 
                                     f1_maxent_allseason_allseason_train = numeric(), f1_maxent_allseason_allseason_val = numeric(), f1_maxent_allseason_allseason_all = numeric(), 
                                     threshold_maxent_allseason_allseason = numeric(), 
                                     p_trainpart_allseason = numeric(), p_valpart_allseason = numeric(), p_allseason = numeric(), 
                                     pa_trainpart_allseason = numeric(), pa_valpart_allseason = numeric(), pa_allseason = numeric())
df_allseason_season <- data.frame(species = character(), date = character(), 
                                  f1_maxent_allseason_season_train = numeric(), f1_maxent_allseason_season_val = numeric(), f1_maxent_allseason_season_all = numeric(),
                                  threshold_maxent_allseason_allseason = numeric(), 
                                  p_trainpart_season = numeric(), p_valpart_season = numeric(), p_season = numeric(), 
                                  pa_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_season = numeric())

dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_binary <- file.path(dir_base_run_id, 'binary', 'allseason')
create_folder(dir_run_id_binary)
dir_run_id_tif <- file.path(dir_base_run_id, 'tif', 'allseason')


for(allseason in allseason_list[r_start:r_end]){
  # allseason <- allseason_list[1]
  # allseason <- '07-01'
  date_list_all_selectseason <- date_list_all[which(substr(date_list_all, 6, 10) == allseason)]
  date_list_selectseason <- date_list[which(substr(date_list, 6, 10) == allseason)]
  
  for(species in species_list){
    # species <- species_list[1]
    # species <- 'Egretta_sacra'
    sp_allseason <- paste0(species, '_', allseason)
    
    generate_points_allseason()
    
    set_default_variable_allseason()
    
    # set the prediction result does not exists
    maxent_allseason_allseason_exists <- FALSE
    
    # check if the maxent process has been conducted
    maxent_allseason_allseason_tif_path <- file.path(dir_run_id_tif, sprintf('%s_%s_%s.tif', sp_allseason, 'maxent_allseason_allseason', run_id))
    maxent_allseason_allseason_exists <- file.exists(maxent_allseason_allseason_tif_path)
    
    if(maxent_allseason_allseason_exists){
      maxent_allseason_allseason <- raster::raster(maxent_allseason_allseason_tif_path)
      
      maxent_allseason_allseason_binary_file <- sprintf('%s_%s_%s_%s_binary.tif', species, allseason, 'maxent_allseason_allseason', run_id)
      maxent_allseason_allseason_results <- prediction_process(maxent_allseason_allseason, 
                                                               xy_p_allseason_trainsplit, xy_pa_allseason_sample_trainsplit, 
                                                               xy_p_allseason_valsplit, xy_pa_allseason_sample_valsplit, 
                                                               xy_p_allseason, xy_pa_allseason_sample, 
                                                               dir_run_id_binary, maxent_allseason_allseason_binary_file)
      f1_maxent_allseason_allseason_train <- maxent_allseason_allseason_results$f1_train
      f1_maxent_allseason_allseason_val <- maxent_allseason_allseason_results$f1_val
      f1_maxent_allseason_allseason_all <- maxent_allseason_allseason_results$f1_all
      threshold_maxent_allseason_allseason <- maxent_allseason_allseason_results$threshold
    }
    df_allseason_allseason[nrow(df_allseason_allseason)+1, ] <- c(sp_allseason, 
                                                                  f1_maxent_allseason_allseason_train, f1_maxent_allseason_allseason_val, f1_maxent_allseason_allseason_all, 
                                                                  threshold_maxent_allseason_allseason, 
                                                                  p_trainpart_allseason, p_valpart_allseason, p_allseason, 
                                                                  pa_trainpart_allseason, pa_valpart_allseason, pa_allseason)
    
    for(time in date_list_selectseason){
      # time <- date_list_selectseason[1]
      # time <- '2018-07-01'
      print(paste('start', species, time))
      sp_season <- paste0(species, '_', time)
      
      generate_points()
      
      # set variables as default values
      set_default_variable()
      
      # set the prediction result does not exists
      maxent_allseason_season_exists <- FALSE
      
      # check if the maxent process has been conducted
      maxent_allseason_season_tif_path <- file.path(dir_run_id_tif, sprintf('%s_%s_%s.tif', sp_season, 'maxent_allseason_season', run_id))
      maxent_allseason_season_exists <- file.exists(maxent_allseason_season_tif_path)
    
      if(maxent_allseason_season_exists){
        
        # read maxent_allseason_season
        maxent_allseason_season <- raster::raster(maxent_allseason_season_tif_path)
        
        # process
        maxent_allseason_season_binary_file <- sprintf('%s_%s_%s_%s_binary.tif', species, time, 'maxent_allseason_season', run_id)
        maxent_allseason_season_results <- prediction_process(maxent_allseason_season, 
                                                              xy_p_season_trainsplit, xy_pa_season_sample_trainsplit, 
                                                              xy_p_season_valsplit, xy_pa_season_sample_valsplit, 
                                                              xy_p_season, xy_pa_season_sample, 
                                                              dir_run_id_binary, maxent_allseason_season_binary_file)

          f1_maxent_allseason_season_val <- maxent_allseason_season_results$f1_val
          f1_maxent_allseason_season_train <- maxent_allseason_season_results$f1_train
          f1_maxent_allseason_season_all <- maxent_allseason_season_results$f1_all
          threshold_maxent_allseason_season <- maxent_allseason_season_results$threshold
      }
      df_allseason_season[nrow(df_allseason_season)+1, ] <- c(species, time,  
                                                              f1_maxent_allseason_season_train, f1_maxent_allseason_season_val, f1_maxent_allseason_season_all, 
                                                              threshold_maxent_allseason_season, 
                                                              p_trainpart_season, p_valpart_season, p_season, pa_trainpart_season, pa_valpart_season, pa_season)
    }
  }
}
output_csv_allseason_allseason_path <- file.path(dir_base_run_id, sprintf('f1_result_allseason_allseason_%s.csv', r_start))
output_csv_allseason_season_path <- file.path(dir_base_run_id, sprintf('f1_result_allseason_season_%s.csv', r_start))
write.csv(df_allseason_allseason, output_csv_allseason_allseason_path, row.names = FALSE)
write.csv(df_allseason_season, output_csv_allseason_season_path, row.names = FALSE)

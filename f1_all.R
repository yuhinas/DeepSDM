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
r_end <- r_start + 39

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
date_list <- DeepSDM_conf$training_conf$date_list_predict
date_list_all <- DeepSDM_conf$training_conf$date_list_train

# make species_list for prediction
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

# season: specific months in one year
# allseason: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df_all_all <- data.frame(sptime = character(), 
                         f1_maxent_all_all_train = numeric(), f1_maxent_all_all_val = numeric(), f1_maxent_all_all_all = numeric(), 
                         threshold_maxent_all_all = numeric(), 
                         p_trainpart_all = numeric(), p_valpart_all = numeric(), p_all = numeric(), 
                         pa_trainpart_all = numeric(), pa_valpart_all = numeric(), pa_all = numeric())
df_all_season <- data.frame(species = character(), date = character(), 
                            f1_maxent_all_season_train = numeric(), f1_maxent_all_season_val = numeric(), f1_maxent_all_season_all = numeric(),
                            threshold_maxent_all_season = numeric(), 
                            p_trainpart_season = numeric(), p_valpart_season = numeric(), p_season = numeric(), 
                            pa_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_season = numeric())

dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_binary <- file.path(dir_base_run_id, 'binary', 'all')
create_folder(dir_run_id_binary)
dir_run_id_tif <- file.path(dir_base_run_id, 'tif', 'all')


for(species in species_list[r_start:min(r_end, length(species_list))]){
  # species <- species_list[1]
  # species <- 'Ketupa_flavipes'
  generate_points_all()
  
  set_default_variable_all()
  
  # set the prediction result does not exists
  maxent_all_all_exists <- FALSE
  
  # check if the maxent process has been conducted
  maxent_all_all_tif_path <- file.path(dir_run_id_tif, sprintf('%s_%s_%s.tif', species, 'maxent_all_all', run_id))
  maxent_all_all_exists <- file.exists(maxent_all_all_tif_path)
  
  if(maxent_all_all_exists){
    maxent_all_all <- raster::raster(maxent_all_all_tif_path)
    
    maxent_all_all_binary_file <- sprintf('%s_%s_%s_binary.tif', species, 'maxent_all_all', run_id)
    maxent_all_all_results <- prediction_process(maxent_all_all, 
                                                 xy_p_all_trainsplit, xy_pa_all_sample_trainsplit, 
                                                 xy_p_all_valsplit, xy_pa_all_sample_valsplit, 
                                                 xy_p_all, xy_pa_all_sample, 
                                                 dir_run_id_binary, maxent_all_all_binary_file)
    f1_maxent_all_all_train <- maxent_all_all_results$f1_train
    f1_maxent_all_all_val <- maxent_all_all_results$f1_val
    f1_maxent_all_all_all <- maxent_all_all_results$f1_all
    threshold_maxent_all_all <- maxent_all_all_results$threshold
  }
  df_all_all[nrow(df_all_all)+1, ] <- c(species, 
                                        f1_maxent_all_all_train, f1_maxent_all_all_val, f1_maxent_all_all_all, 
                                        threshold_maxent_all_all, 
                                        p_trainpart_all, p_valpart_all, p_all, 
                                        pa_trainpart_all, pa_valpart_all, pa_all)
  
  for(time in date_list){
    # time <- date_list[1]
    print(paste('start', species, time))

    generate_points()
    
    # set variables as default values
    set_default_variable()
    
    # set the prediction result does not exists
    maxent_all_season_exists <- FALSE
    
    # check if the maxent process has been conducted
    maxent_all_season_tif_path <- file.path(dir_run_id_tif, sprintf('%s_%s_%s_%s.tif', species, time, 'maxent_all_season', run_id))
    maxent_all_season_exists <- file.exists(maxent_all_season_tif_path)
    
    if(maxent_all_season_exists){
      
      # read maxent_all_season
      maxent_all_season <- raster::raster(maxent_all_season_tif_path)
      
      # process
      maxent_all_season_binary_file <- sprintf('%s_%s_%s_%s_binary.tif', species, time, 'maxent_all_season', run_id)
      maxent_all_season_results <- prediction_process(maxent_all_season, 
                                                      xy_p_season_trainsplit, xy_pa_season_sample_trainsplit, 
                                                      xy_p_season_valsplit, xy_pa_season_sample_valsplit, 
                                                      xy_p_season, xy_pa_season_sample, 
                                                      dir_run_id_binary, maxent_all_season_binary_file)
      
      f1_maxent_all_season_val <- maxent_all_season_results$f1_val
      f1_maxent_all_season_train <- maxent_all_season_results$f1_train
      f1_maxent_all_season_all <- maxent_all_season_results$f1_all
      threshold_maxent_all_season <- maxent_all_season_results$threshold
    }
    df_all_season[nrow(df_all_season)+1, ] <- c(species, time,  
                                                f1_maxent_all_season_train, f1_maxent_all_season_val, f1_maxent_all_season_all, 
                                                threshold_maxent_all_season, 
                                                p_trainpart_season, p_valpart_season, p_season, pa_trainpart_season, pa_valpart_season, pa_season)
    }
}
output_csv_all_all_path <- file.path(dir_base_run_id, sprintf('f1_result_all_all_%s.csv', r_start))
output_csv_all_season_path <- file.path(dir_base_run_id, sprintf('f1_result_all_season_%s.csv', r_start))
write.csv(df_all_all, output_csv_all_all_path, row.names = FALSE)
write.csv(df_all_season, output_csv_all_season_path, row.names = FALSE)

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
r_end <- r_start + 59

# specify run_id
# specify `run_id_maxent` as the run_id to call maxent predictions
run_id_maxent <- '2f07dfdd1bef43988e07f22cb2e322c5'
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
date_list_selectseason <- DeepSDM_conf$training_conf$date_list_predict

# make species_list for prediction
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

# season: specific months in one year
# allseason: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model
df <- data.frame(species = character(), date = character(), 
                 f1_maxent_season_season_train = numeric(), f1_maxent_season_season_val = numeric(), f1_maxent_season_season_all = numeric(), 
                 f1_deepsdm_all_season_train = numeric(), f1_deepsdm_all_season_val = numeric(), f1_deepsdm_all_season_all = numeric(),
                 threshold_maxent_season_season = numeric(), threshold_deepsdm_all_season = numeric(), 
                 p_trainpart_season = numeric(), p_valpart_season = numeric(), p_season = numeric(), 
                 pa_trainpart_season = numeric(), pa_valpart_season = numeric(), pa_season = numeric())


dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_binary <- file.path(dir_base_run_id, 'binary', 'season')
create_folder(dir_run_id_binary)

for(species in species_list[r_start:min(r_end, length(species_list))]){
  for(time in date_list){
    # species <- species_list[1]
    # time <- date_list_selectseason[1]
    print(sprintf('%s --- %s', species, time))
    
    # generate points
    generate_points()
  
    set_default_variable()

    if(p_trainpart_season == 0 | p_valpart_season == 0 | pa_trainpart_season == 0 | pa_valpart_season == 0){
      df[nrow(df)+1, ] <- c(species, time, 
                            f1_maxent_season_season_train, f1_maxent_season_season_val, f1_maxent_season_season_all,
                            f1_deepsdm_all_season_train, f1_deepsdm_all_season_val, f1_deepsdm_all_season_all, 
                            threshold_maxent_season_season, threshold_deepsdm_all_season, 
                            p_trainpart_season, p_valpart_season, p_season, pa_trainpart_season, pa_valpart_season, pa_season)
      next
    }
    maxent_season_season_file <- sprintf('%s_%s_maxent_season_season_%s.tif', species, time, run_id_maxent)
    maxent_season_season_path <- file.path('predict_maxent', run_id_maxent, 'tif', 'season', maxent_season_season_file)
    maxent_season_season_exists <- F
    maxent_season_season <- try(raster::raster(maxent_season_season_path))
    maxent_season_season_exists <- inherits(maxent_season_season, 'RasterLayer')
    
    deepsdm_all_season_file <- sprintf('%s_%s_predict.tif', species, time)
    deepsdm_all_season_path <- file.path('predicts', run_id, 'tif', deepsdm_all_season_file)
    deepsdm_all_season_exists <- FALSE
    deepsdm_all_season <- try(raster::raster(deepsdm_all_season_path))
    deepsdm_all_season_exists <- inherits(deepsdm_all_season, 'RasterLayer')
    
    # maxent_season_season
    if(maxent_season_season_exists){
      maxent_file_name <- sprintf('%s_%s_%s_%s_binary.tif', species, time, 'maxent_season_season', run_id_maxent)
      maxent_season_season_results <- prediction_process(maxent_season_season, 
                                                         xy_p_season_trainsplit, xy_pa_season_sample_trainsplit, 
                                                         xy_p_season_valsplit, xy_pa_season_sample_valsplit, 
                                                         xy_p_season, xy_pa_season_sample, 
                                                         dir_run_id_binary, maxent_file_name)
      f1_maxent_season_season_train <- maxent_season_season_results$f1_train
      f1_maxent_season_season_val <- maxent_season_season_results$f1_val
      f1_maxent_season_season_all <- maxent_season_season_results$f1_all
    }
    
    # deepsdm
    if(deepsdm_all_season_exists){
      deepsdm_file_name <- sprintf('%s_%s_%s_%s_binary.tif', species, time, 'deepsdm_all_season', run_id)
      deepsdm_all_season_results <- prediction_process(deepsdm_all_season, 
                                                       xy_p_season_trainsplit, xy_pa_season_sample_trainsplit, 
                                                       xy_p_season_valsplit, xy_pa_season_sample_valsplit, 
                                                       xy_p_season, xy_pa_season_sample, 
                                                       dir_run_id_binary, deepsdm_file_name)
      f1_deepsdm_all_season_train <- deepsdm_all_season_results$f1_train
      f1_deepsdm_all_season_val <- deepsdm_all_season_results$f1_val
      f1_deepsdm_all_season_all <- deepsdm_all_season_results$f1_all
    }
    df[nrow(df)+1, ] <- c(species, time, 
                          f1_maxent_season_season_train, f1_maxent_season_season_val, f1_maxent_season_season_all, 
                          f1_deepsdm_all_season_train, f1_deepsdm_all_season_val, f1_deepsdm_all_season_all, 
                          threshold_maxent_season_season, threshold_deepsdm_all_season, 
                          p_trainpart_season, p_valpart_season, p_season, pa_trainpart_season, pa_valpart_season, pa_season)
  }
}

out_file <- sprintf('f1_result_season_season_%s.csv', r_start)
out_path <- file.path('predict_maxent', run_id, out_file)
write.csv(df, out_path, row.names = FALSE)

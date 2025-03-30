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
calculate_thresholddepend_indi <- function(pred_1, pred_0, actual_1, actual_0, threshold){
    
    pred_classes <- ifelse(c(pred_1, pred_0) >= threshold, 1, 0)
    pred_factor <- factor(pred_classes, levels = c(0, 1))
    actual_factor <- factor(c(actual_1, actual_0), levels = c(0, 1))
    confusion_matrix <- table(pred_factor, actual_factor)
    
    # 計算 Sensitivity 和 Specificity
    TP <- confusion_matrix[2, 2]  # True Positive
    TN <- confusion_matrix[1, 1]  # True Negative
    FP <- confusion_matrix[2, 1]  # False Positive
    FN <- confusion_matrix[1, 2]  # False Negative

    # 計算靈敏度（Sensitivity）和特異性（Specificity）和 Precision
    sensitivity <- TP / (TP + FN)
    specificity <- TN / (TN + FP)
    precision <- TP / (TP + FP)
    N <- length(pred_0) + length(pred_1)
    p_0 <- (TP + TN) / N
    p_e <- ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N^2

    # 計算 TSS
    TSS <- sensitivity + specificity - 1

    # 計算 F1
    f1 <- 2 * precision * sensitivity / (precision + sensitivity)

    # 計算 kappa
    kappa <- (p_0 - p_e) / (1 - p_e)

    return(c(TSS, kappa, f1))    
}                          



args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

# specify run_id, exp_id
run_id <- 'e52c8ac9a3e24c75ac871f63bbdea060'
exp_id <- '115656750127464383'

# create folder 
dir_base_run_id <- file.path('predict_maxent', run_id)
dir_run_id_h5 <- file.path(dir_base_run_id, 'h5', 'all')
dir_run_id_h5_constantthreshold <- file.path(dir_base_run_id, 'h5', 'binary_constantthreshold')
create_folder(dir_run_id_h5_constantthreshold)
dir_run_id_png_constantthreshold <- file.path(dir_base_run_id, 'png', 'binary_constantthreshold')
create_folder(dir_run_id_png_constantthreshold)

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
# date_list_predict <- c('2018-01-01', '2018-04-01', '2018-07-01', '2018-10-01')

# make date_list for training
date_list_train <- DeepSDM_conf$training_conf$date_list_train

# make species_list for prediction
species_list <- sort(DeepSDM_conf$training_conf$species_list_train)

# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
# AAA_BBB_CCC_DDD -> In DDD split, auc_roc score of using BBB env data for training and CCC env data for prediction with model AAA
# eg. 'maxent_season_season_val' means In validation split, auc_roc score of using 'season' env data for training and predicting with 'season' data with maxent model

#plotting colormap
color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')

df_indi <- data.frame(species = character(), date = character(), 
                      maxent_train_TSS = numeric(), maxent_train_kappa = numeric(), maxent_train_f1 = numeric(), 
                      deepsdm_train_TSS = numeric(), deepsdm_train_kappa = numeric(), deepsdm_train_f1 = numeric(), 
                      maxent_val_TSS = numeric(), maxent_val_kappa = numeric(), maxent_val_f1 = numeric(), 
                      deepsdm_val_TSS = numeric(), deepsdm_val_kappa = numeric(), deepsdm_val_f1 = numeric(), 
                      train_points = numeric(), val_points = numeric(), 
                      threshold_maxent = numeric(), threshold_deepsdm = numeric())

r_start <- as.numeric(args[1])
# r_start <- 121
r_end <- r_start + 2
for(species in species_list[r_start:min(r_end, length(species_list))]){
  # species <- species_list[121]
  # species <- 'Alauda_gulgula'
  # set the prediction result does not exists
    
    dir_run_id_h5_sp <- file.path(dir_run_id_h5, species)
    
    dir_run_id_h5_constantthreshold_sp <- file.path(dir_run_id_h5_constantthreshold, species)
    create_folder(dir_run_id_h5_constantthreshold_sp)
    dir_run_id_png_constantthreshold_sp <- file.path(dir_run_id_png_constantthreshold, species)
    create_folder(dir_run_id_png_constantthreshold_sp)
    
    maxent_h5_exists <- FALSE
    maxent_h5_path <- file.path(dir_run_id_h5_sp, sprintf('%s.h5', species))
    maxent_h5_exists <- file.exists(maxent_h5_path)
    
    deepsdm_h5_exists <- FALSE
    deepsdm_h5_path <- file.path('predicts', run_id, 'h5', species, sprintf('%s.h5', species))
    deepsdm_h5_exists <- file.exists(deepsdm_h5_path)
    
    maxent_10 <- list()
    deepsdm_10 <- list()
    actual_10 <- list()
    for(date in date_list_predict){
        # date <- date_list_predict[1]
        # date <- '2018-12-01'
        print(paste('start', species, date))
        set_default_variable()

        generate_points(num_pa = 'num_p')
        if(nrow(xy_p_season_trainsplit) == 0 | nrow(xy_p_season_valsplit) == 0 | nrow(xy_pa_season_sample_trainsplit) == 0 | nrow(xy_pa_season_sample_valsplit) == 0){
                            next
                        }
        
        actual_10[[date]] <- list()
        actual_10[[date]]$train_actual_1 <- rep(1, nrow(xy_p_season_trainsplit))
        actual_10[[date]]$train_actual_0 <- rep(0, nrow(xy_pa_season_sample_trainsplit))  
        actual_10[[date]]$val_actual_1 <- rep(1, nrow(xy_p_season_valsplit))
        actual_10[[date]]$val_actual_0 <- rep(0, nrow(xy_pa_season_sample_valsplit))
        actual_10[[date]]$xy_p_season <- xy_p_season
        # maxent
        maxent_h5_date_exists <- FALSE
        
        if(maxent_h5_exists){
            
            # set variables as default values
            set_default_variable()
            
            # check if maxent predictions have existed
            maxent_h5_date_exists <- check_dataset_in_h5(maxent_h5_path, date)
            if(maxent_h5_date_exists){
                maxent_all_season <- h5dataset_to_raster(maxent_h5_path, date)
                
                maxent_10[[date]] <- list()
                maxent_10[[date]]$train_pred_1 <- raster::extract(maxent_all_season, xy_p_season_trainsplit) %>% as.numeric()
                maxent_10[[date]]$train_pred_0 <- raster::extract(maxent_all_season, xy_pa_season_sample_trainsplit)
                maxent_10[[date]]$val_pred_1 <- raster::extract(maxent_all_season, xy_p_season_valsplit)
                maxent_10[[date]]$val_pred_0 <- raster::extract(maxent_all_season, xy_pa_season_sample_valsplit)                
            }
        }
        
        # deepsdm
        deepsdm_h5_date_exists <- FALSE
        
        if(deepsdm_h5_exists){
            # set variables as default values
            set_default_variable()
            
            # check if maxent predictions have existed
            deepsdm_h5_date_exists <- check_dataset_in_h5(deepsdm_h5_path, date)
            if(deepsdm_h5_date_exists){
                deepsdm_all_season <- h5dataset_to_raster(deepsdm_h5_path, date)

                deepsdm_10[[date]] <- list()
                deepsdm_10[[date]]$train_pred_1 <- raster::extract(deepsdm_all_season, xy_p_season_trainsplit) %>% as.numeric()
                deepsdm_10[[date]]$train_pred_0 <- raster::extract(deepsdm_all_season, xy_pa_season_sample_trainsplit)
                deepsdm_10[[date]]$val_pred_1 <- raster::extract(deepsdm_all_season, xy_p_season_valsplit)
                deepsdm_10[[date]]$val_pred_0 <- raster::extract(deepsdm_all_season, xy_pa_season_sample_valsplit)
            }            
        }
    }
    if(length(actual_10) == 0){
        next
    }
    
    # calculate ONE threshold of two models based on all the presence and pseudo-absence
    all_train_maxent_1 <- unlist(lapply(maxent_10, function(sublist) sublist$train_pred_1))
    all_train_maxent_0 <- unlist(lapply(maxent_10, function(sublist) sublist$train_pred_0))
    all_train_deepsdm_1 <- unlist(lapply(deepsdm_10, function(sublist) sublist$train_pred_1))
    all_train_deepsdm_0 <- unlist(lapply(deepsdm_10, function(sublist) sublist$train_pred_0))
    all_train_actual_1 <- unlist(lapply(actual_10, function(sublist) sublist$train_actual_1))
    all_train_actual_0 <- unlist(lapply(actual_10, function(sublist) sublist$train_actual_0))
    all_val_maxent_1 <- unlist(lapply(maxent_10, function(sublist) sublist$val_pred_1))
    all_val_maxent_0 <- unlist(lapply(maxent_10, function(sublist) sublist$val_pred_0))
    all_val_deepsdm_1 <- unlist(lapply(deepsdm_10, function(sublist) sublist$val_pred_1))
    all_val_deepsdm_0 <- unlist(lapply(deepsdm_10, function(sublist) sublist$val_pred_0))
    all_val_actual_1 <- unlist(lapply(actual_10, function(sublist) sublist$val_actual_1))
    all_val_actual_0 <- unlist(lapply(actual_10, function(sublist) sublist$val_actual_0))
    
    roc_train_maxent <- roc(c(all_train_actual_1, all_train_actual_0), c(all_train_maxent_1, all_train_maxent_0))
    roc_train_deepsdm <- roc(c(all_train_actual_1, all_train_actual_0), c(all_train_deepsdm_1, all_train_deepsdm_0))
                                      
    best_threshold_train_maxent <- coords(roc_train_maxent, 'best', ret=c('threshold')) %>% pull() %>% min()
    best_threshold_train_deepsdm <- coords(roc_train_deepsdm, 'best', ret=c('threshold')) %>% pull() %>% min()
                                      
                                 
    # Use the threshold to change continuous prediction map to binary map
    # And calculate the TSS and f1 of each time
    for(date in date_list_predict){
        
        if(maxent_h5_exists){
            if(check_dataset_in_h5(maxent_h5_path, date)){        
                maxent_all_season <- h5dataset_to_raster(maxent_h5_path, date)
                log_binary(dir_run_id_h5_constantthreshold, dir_run_id_png_constantthreshold, 
                           species, date, maxent_all_season, best_threshold_train_maxent, extent_binary, 
                           'maxent_all_all_binary_constantthreshold', run_id, actual_10[[date]]$xy_p_season, sprintf('%s_maxent.h5', species))
            }
        }
        if(deepsdm_h5_exists){
            if(check_dataset_in_h5(deepsdm_h5_path, date)){        
                deepsdm_all_season <- h5dataset_to_raster(deepsdm_h5_path, date)
                log_binary(dir_run_id_h5_constantthreshold, dir_run_id_png_constantthreshold, 
                           species, date, deepsdm_all_season, best_threshold_train_deepsdm, extent_binary, 
                           'deepsdm_all_all_binary_constantthreshold', run_id, actual_10[[date]]$xy_p_season, sprintf('%s_deepsdm.h5', species))
            }
        }        

        
        
        
        maxent_train_indi <- calculate_thresholddepend_indi(maxent_10[[date]]$train_pred_1, 
                                                            maxent_10[[date]]$train_pred_0, 
                                                            actual_10[[date]]$train_actual_1, 
                                                            actual_10[[date]]$train_actual_0, 
                                                            best_threshold_train_maxent)
        deepsdm_train_indi <- calculate_thresholddepend_indi(deepsdm_10[[date]]$train_pred_1, 
                                                             deepsdm_10[[date]]$train_pred_0, 
                                                             actual_10[[date]]$train_actual_1, 
                                                             actual_10[[date]]$train_actual_0, 
                                                             best_threshold_train_deepsdm)           
        
        maxent_val_indi <- calculate_thresholddepend_indi(maxent_10[[date]]$val_pred_1, 
                                                            maxent_10[[date]]$val_pred_0, 
                                                            actual_10[[date]]$val_actual_1, 
                                                            actual_10[[date]]$val_actual_0, 
                                                            best_threshold_train_maxent)        
        deepsdm_val_indi <- calculate_thresholddepend_indi(deepsdm_10[[date]]$val_pred_1, 
                                                           deepsdm_10[[date]]$val_pred_0, 
                                                           actual_10[[date]]$val_actual_1, 
                                                           actual_10[[date]]$val_actual_0, 
                                                           best_threshold_train_deepsdm)           
        df_indi[nrow(df_indi)+1, ] = c(species, date, 
                                       unlist(maxent_train_indi), unlist(deepsdm_train_indi), 
                                       unlist(maxent_val_indi), unlist(deepsdm_val_indi), 
                                       length(actual_10[[date]]$train_actual_1), length(actual_10[[date]]$train_actual_0), 
                                       best_threshold_train_maxent, best_threshold_train_deepsdm)
    }
        
}
output_path <- file.path(dir_base_run_id, sprintf('only_threshold_depend_indi_%03d.csv', r_start))
write.csv(df_indi, output_path, row.names = FALSE)

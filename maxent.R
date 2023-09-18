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

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

r_start <- as.numeric(args[1])
# r_start <- 1
r_end <- r_start + 299

DeepSDM_conf <- yaml.load_file('DeepSDM_conf.yaml')

extent_binary <- raster('workspace/extent_binary.tif')
extent_binary[extent_binary == 0] <- NA
trainval_split <- raster('tmp/DeepSDM DEMO/20230911141642-0_partition_extent.tif')
trainval_split[is.na(values(trainval_split))] = 0

env_info <- fromJSON(file = 'workspace/env_information.json')
env_list <- sort(DeepSDM_conf$env_list)

sp_occ_filter <- read.csv('workspace/species_data/occurrence_data/species_occurrence_filter.csv')
# points_season <- read.csv('..//raster_points_season.csv')
# points_months <- read.csv('..//raster_points_months.csv')
# species_index <- read.csv('..//species_index.csv')
date_list_predict <- c()
i <- 1
for (y_ in c(2018)){
  for (m_ in 1:12){
    date_list_predict[i] <- paste0(sprintf('%04d', y_), '-', sprintf('%02d', m_))
    i <- i + 1
  }
}
# season: specific months in one year
# seasonavg: every speciefic months in the whole time span
# all: the whole time span
df <- data.frame(spyrmon = character(),
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


files <- list.files('predicts/tif/')
files <- sort(files)

predict_maxent <- function(env, xm){
  result <- try(predict(env, xm, progress = ''), silent = T)
  return(result)
}

plot_result <- function(spyrmon, species, xm, extent_binary, pall, pseasonavg, pseason, log_info){
  if (!dir.exists(sprintf('predicts_maxent/png/%s', species))){
    dir.create(sprintf('predicts_maxent/png/%s', species), recursive = T)
  }
  if (!dir.exists(sprintf('predicts_maxent/tif/%s', species))){
    dir.create(sprintf('predicts_maxent/tif/%s', species), recursive = T)
  }
  png(file.path('predicts_maxent',
                'png',
                species,
                sprintf('%s_%s.png', spyrmon, log_info)),
      width = 2000,
      height = 2000)
  
  plot(xm * extent_binary,
       main = sprintf('%s_%s', spyrmon, log_info),
       axes = FALSE,
       box = FALSE,
       legend = FALSE,
       cex.main = 5,
       col = color,
       breaks = seq(0, 1, 0.125))
  
  # points(pall, pch = 16, col = 'black', cex = 1.5)
  # points(pseasonavg, pch = 16, col = 'blue', cex = 1.5)
  points(pseason, pch = 16, col = 'red', cex = 1.5)
  dev.off()
  writeRaster(xm * extent_binary, 
              file.path('predicts_maxent', 
                        'tif', 
                        species, 
                        sprintf('%s_%s.tif', spyrmon, log_info)), 
              overwrite = T)
}

plot_result_deepsdm <- function(spyrmon, species, xm, extent_binary, pall, pseasonavg, pseason, log_info){
  png(file.path('predicts_maxent',
                'png', 
                species,
                sprintf('%s_%s.png', spyrmon, log_info)),
      width = 2000,
      height = 2000)
  
  plot(xm * extent_binary,
       main = sprintf('%s_%s', spyrmon, log_info),
       axes = FALSE,
       box = FALSE,
       legend = FALSE,
       cex.main = 5,
       col = color,
       breaks = seq(0, 1, 0.125))
  
  # points(pall, pch = 16, col = 'black', cex = 1.5)
  # points(pseasonavg, pch = 16, col = 'blue', cex = 1.5)
  points(pseason, pch = 16, col = 'red', cex = 1.5)
  dev.off()
}

calculate_roc <- function(px, p, bg){
  if(nrow(p) == 0){
    return(-9)
  }else{
    pred_1 <- raster::extract(px, p)
    pred_0 <- raster::extract(px, bg)
    actual_1 <- rep(1, nrow(p))
    actual_0 <- rep(0, nrow(bg))
    roc <- roc(c(actual_1, actual_0), c(pred_1, pred_0))
    return(roc$auc[1])
  }
}


if (!dir.exists('predicts_maxent')){
  dir.create('predicts_maxent')
}

for(f in files[r_start:r_end]){
  # f <- files[1]
  print(paste('start', f))
  f_split <- ((f %>% strsplit('_'))[[1]][2] %>% strsplit('-'))[[1]] # "Acridotheres", "cristatellus", "2018", "01", "01"
  species <- paste(f_split[1], f_split[2], sep = '_')
  species_ <- paste(f_split[1], f_split[2], sep = '-')

  spyrmon <- paste(f_split[1], f_split[2], f_split[3], f_split[4], f_split[5], sep = '_')
  yrmon <-paste0(f_split[3], '-', f_split[4])
  yrmonday <- paste0(f_split[3], '-', f_split[4], '-', f_split[5])
  
  # load env layers of season
  files_env <- c()
  i <- 1
  for(env in sort(DeepSDM_conf$env_list)){
    files_env[i] <- file.path(env_info$info[[env]][[yrmon]]$tif_span_avg)
    i <- i + 1
  }
  env_season <- raster::stack(files_env)
  names(env_season) <- sort(DeepSDM_conf$env_list)
  
  list.files(sprintf('workspace/raster_data/species_occurrence/%s', species))
  tif <- raster::raster(sprintf('workspace/raster_data/species_occurrence/%s/%s_%s.tif', species, species, yrmonday))
  xy_p_season <- xyFromCell(tif, which(values(tif) == 1)) # x,y value from cells with presence records
  xy_p_season_trainsplit <- xyFromCell(tif, which((values(tif) == 1) & (values(trainval_split) == 1)))
  xy_p_season_valsplit <- xyFromCell(tif, which((values(tif) == 1) & (values(trainval_split) == 0)))

  xy_pa_season <- xyFromCell(tif, which((values(tif) == 0) & (!is.na(values(extent_binary)))))
  xy_pa_season_sample <- xy_pa_season[sample(nrow(xy_pa_season), 10000), ]
  xy_pa_season_sample_trainsplit <- xy_pa_season_sample[which(values(trainval_split)[cellFromXY(trainval_split, xy_pa_season_sample)] == 1), ]
  xy_pa_season_sample_valsplit <- xy_pa_season_sample[which(values(trainval_split)[cellFromXY(trainval_split, xy_pa_season_sample)] == 0), ]
  
  
  maxent_season_season_val = -9
  maxent_season_seasonavg_val = -9
  maxent_season_all_val = -9
  maxent_seasonavg_season_val = -9
  maxent_seasonavg_seasonavg_val = -9
  maxent_seasonavg_all_val = -9
  maxent_all_season_val = -9
  maxent_all_seasonavg_val = -9
  maxent_all_all_val = -9
  deepsdm_all_season_val = -9
  deepsdm_all__seasonavg_val = -9
  deepsdm_all_all_val = -9
  maxent_season_season_train = -9
  maxent_season_seasonavg_train = -9
  maxent_season_all_train = -9
  maxent_seasonavg_season_train = -9
  maxent_seasonavg_seasonavg_train = -9
  maxent_seasonavg_all_train = -9
  maxent_all_season_train = -9
  maxent_all_seasonavg_train = -9
  maxent_all_all_train = -9
  deepsdm_all_season_train = -9
  deepsdm_all_seasonavg_train = -9
  deepsdm_all_all_train = -9
  p_season = nrow(xy_p_season)
  p_valpart_season = nrow(xy_p_season_valsplit)
  p_trainpart_season = nrow(xy_p_season_trainsplit)
  pa_valpart_season = nrow(xy_pa_season_sample_valsplit)
  pa_trainpart_season = nrow(xy_pa_season_sample_trainsplit)
  p_seasonavg = -9
  p_valpart_seasonavg = -9
  p_trainpart_seasonavg = -9
  pa_valpart_seasonavg = -9
  pa_trainpart_seasonavg = -9
  p_all = -9
  p_valpart_all = -9
  p_trainpart_all = -9
  pa_valpart_all = -9
  pa_trainpart_all = -9
  
  
  #plotting
  color <- c('#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704')
  # xmall <- try(maxent(x = envall, p = p_trainall, a = bg_trainall), silent = T)
  # if(!is.character(xmall)){
  #   
  #   write.csv(xmall@results, 
  #             file.path('for_testing', 
  #                       paste('result', time_log, sep = '_'), 
  #                       species, 
  #                       paste(spyrmon, 'env_contribution_maxentall.csv', sep = '_')))
  #   
  #   pxall_all <- predict_maxent(envall, xmall)
  #   
  #   plot_result(time_log, spyrmon, pxall_all, extent10, pall, p12, p456, epoch, 'maxent_all_all')
  #   maxentall_all_train <- calculate_roc(pxall_all, p_trainall, bg_trainall)
  #   maxentall_all_test <- calculate_roc(pxall_all, p_testall, bg_testall)
  #   
  #   
  #   pxall_12 <- predict_maxent(env12, xmall)
  #   
  #   plot_result(time_log, spyrmon, pxall_12, extent10, pall, p12, p456, epoch, 'maxent_all_12')
  #   maxentall_12_train <- calculate_roc(pxall_12, p_train12, bg_train12)
  #   maxentall_12_test <- calculate_roc(pxall_12, p_test12, bg_test12)
  #   
  #   
  #   pxall_456 <- predict_maxent(env456, xmall)
  #   
  #   plot_result(time_log, spyrmon, pxall_456, extent10, pall, p12, p456, epoch, 'maxent_all_456')
  #   maxentall_456_train <- calculate_roc(pxall_456, p_train456, bg_train456)
  #   maxentall_456_test <- calculate_roc(pxall_456, p_test456, bg_test456)
  #   
  # }
  # 
  # xm12 <- try(maxent(x = env12, p = p_train12, a = bg_train12), silent = T)
  # if(!is.character(xm12)){
  #   
  #   write.csv(xm12@results, 
  #             file.path('for_testing', 
  #                       paste('result', time_log, sep = '_'), 
  #                       species, 
  #                       paste(spyrmon, 'env_contribution_maxent12.csv', sep = '_')))
  #   
  #   px12_all <- predict_maxent(envall, xm12)
  #   
  #   plot_result(time_log, spyrmon, px12_all, extent10, pall, p12, p456, epoch, 'maxent_12_all')
  #   maxent12_all_train <- calculate_roc(px12_all, p_trainall, bg_trainall)
  #   maxent12_all_test <- calculate_roc(px12_all, p_testall, bg_testall)
  #   
  #   px12_12 <- predict_maxent(env12, xm12)
  #   
  #   plot_result(time_log, spyrmon, px12_12, extent10, pall, p12, p456, epoch, 'maxent_12_12')
  #   maxent12_12_train <- calculate_roc(px12_12, p_train12, bg_train12)
  #   maxent12_12_test <- calculate_roc(px12_12, p_test12, bg_test12)
  #   
  #   
  #   px12_456 <- predict_maxent(env456, xm12)
  #   
  #   plot_result(time_log, spyrmon, px12_456, extent10, pall, p12, p456, epoch, 'maxent_12_456')
  #   maxent12_456_train <- calculate_roc(px12_456, p_train456, bg_train456)
  #   maxent12_456_test <- calculate_roc(px12_456, p_test456, bg_test456)
  #   
  # }
  # 
  # 
  xm_season <- try(maxent(x = env_season, p = xy_p_season_trainsplit, a = xy_pa_season_sample_trainsplit), silent = T)
  if(!is.character(xm_season)){

    write.csv(xm_season@results,
              file.path('for_testing',
                        paste('result', time_log, sep = '_'),
                        species,
                        paste(spyrmon, 'env_contribution_maxent456.csv', sep = '_')))

    # px456_all <- predict(envall, xm456)
    # 
    # plot_result(time_log, spyrmon, px456_all, extent10, pall, p12, p456, epoch, 'maxent_456_all')
    # maxent456_all_train <- calculate_roc(px456_all, p_trainall, bg_trainall)
    # maxent456_all_test <- calculate_roc(px456_all, p_testall, bg_testall)


    # px456_12 <- predict(env12, xm456)
    # 
    # plot_result(time_log, spyrmon, px456_12, extent10, pall, p12, p456, epoch, 'maxent_456_12')
    # maxent456_12_train <- calculate_roc(px456_12, p_train12, bg_train12)
    # maxent456_12_test <- calculate_roc(px456_12, p_test12, bg_test12)


    px_season_season <- predict_maxent(env_season, xm_season)

    plot_result(spyrmon, species, px_season_season, extent_binary, xy_p_season, xy_p_season, xy_p_season, 'maxent_season_season')
    maxent_season_season_train <- calculate_roc(px_season_season, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit)
    maxent_season_season_val <- calculate_roc(px_season_season, xy_p_season_valsplit, xy_pa_season_sample_valsplit)
  }
  deepsdm <- raster::raster(sprintf('predicts/tif/_%s-%s_predict.tif', species_, yrmonday))
  # deepsdmall_all_train <- try(calculate_roc(deepsdm, p_trainall, bg_trainall))
  # deepsdmall_all_test <- try(calculate_roc(deepsdm, p_testall, bg_testall))
  # deepsdmall_12_train <- try(calculate_roc(deepsdm, p_train12, bg_train12))
  # deepsdmall_12_test <- try(calculate_roc(deepsdm, p_test12, bg_test12))
  deepsdm_all_season_train <- try(calculate_roc(deepsdm, xy_p_season_trainsplit, xy_pa_season_sample_trainsplit))
  deepsdm_all_season_val <- try(calculate_roc(deepsdm, xy_p_season_valsplit, xy_pa_season_sample_valsplit))
  plot_result_deepsdm(spyrmon, species, deepsdm, extent_binary, xy_p_season, xy_p_season, xy_p_season, 'deepsdm_all_season')
  
  df[nrow(df)+1, ] <- c(spyrmon,
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
if(!file.exists(file.path('for_testing', paste('result', time_log, sep = '_'), epoch))){
  file.create(file.path('for_testing', paste('result', time_log, sep = '_'), epoch))
}
write.csv(d, 
          file.path('for_testing', 
                    paste('result', time_log, sep = '_'), 
                    epoch, 
                    paste('auc_result', paste0(r_start, '.csv'), sep = '_')), 
          row.names = FALSE)



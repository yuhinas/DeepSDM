# create folder 
create_folder <- function(dir){
  if (!dir.exists(dir)){
    dir.create(dir, recursive = T)
  }
}

# function of maxent
predict_maxent <- function(env, xm){
  result <- try(predict(env, xm, progress = ''), silent = T)
  return(result)
}

# function of plotting maxent results
plot_result <- function(sp, xm, extent_binary, p, log_info, dir_timelog_png_sp, dir_timelog_h5_sp, timelog, time = NULL){
  sptime <- if (is.null(time)) sp else sprintf('%s_%s', sp, time)
  png(file.path(dir_timelog_png_sp, 
                sprintf('%s_%s_%s.png', sptime, log_info, timelog)
               ),
      width = 2000,
      height = 2000)
  
  plot(xm * extent_binary,
       main = sprintf('%s_%s_%s', sptime, log_info, timelog),
       axes = FALSE,
       box = FALSE,
       legend = FALSE,
       cex.main = 4,
       col = color,
       breaks = seq(0, 1, 0.125))
  
  points(p, pch = 16, col = 'red', cex = 1.5)
  dev.off()
#   writeRaster(xm * extent_binary, 
#               file.path(dir_timelog_tif_sp, 
#                         sprintf('%s_%s_%s.tif', sptime, log_info, timelog)), 
#               overwrite = T)
  # HDF5 文件路徑
  h5_file_path <- file.path(dir_timelog_h5_sp, sprintf('%s.h5', sp))

  # 打開或創建 HDF5 文件
  h5_file <- H5File$new(h5_file_path, mode = "a")  # 'a' 模式

  # 確保 dataset 名稱不重複
  if (sptime %in% h5_file$ls()) {
      h5_file[[sptime]]$delete()  # 刪除現有 dataset
  }

  # 寫入數據
  h5_file[[sptime]] <- xm * extent_binary  # 存儲數據

  # 關閉文件
  h5_file$close()
}

# function of plotting deepsdm result
plot_result_deepsdm <- function(sptime, xm, extent_binary, p, log_info, dir_timelog_png_sp, timelog){
  png(file.path(dir_timelog_png_sp, 
                sprintf('%s_%s_%s.png', sptime, log_info, timelog)),
      width = 2000,
      height = 2000)
  
  plot(xm * extent_binary,
       main = sprintf('%s_%s_%s', sptime, log_info, timelog),
       axes = FALSE,
       box = FALSE,
       legend = FALSE,
       cex.main = 4,
       col = color,
       breaks = seq(0, 1, 0.125))
  
  points(p, pch = 16, col = 'red', cex = 1.5)
  dev.off()
}

# function of auc_roc calculations
calculate_roc <- function(px, p, bg){
  if(nrow(p) == 0 | nrow(bg) == 0){
    return(-9999)
  }else{
    pred_1 <- raster::extract(px, p)
    pred_0 <- raster::extract(px, bg)
    actual_1 <- rep(1, nrow(p))
    actual_0 <- rep(0, nrow(bg))
    roc <- roc(c(actual_1, actual_0), c(pred_1, pred_0))
    return(roc$auc[1])
  }
}

# function of other indicator calculations
calculate_indicator <- function(rst){
  # 如果沒有presence points就直接return 0
  if(nrow(xy_p_season_trainsplit) == 0 | nrow(xy_p_season_valsplit) == 0 | nrow(xy_pa_season_sample_trainsplit) == 0 | nrow(xy_pa_season_sample_valsplit) == 0){
    return(c(-9999, -9999, -9999))
  }
  
  train_pred_1 <- raster::extract(rst, xy_p_season_trainsplit) %>% as.numeric()
  train_pred_0 <- raster::extract(rst, xy_pa_season_sample_trainsplit)
  train_actual_1 <- rep(1, nrow(xy_p_season_trainsplit))
  train_actual_0 <- rep(0, nrow(xy_pa_season_sample_trainsplit))
  roc_obj_train <- roc(c(train_actual_1, train_actual_0), c(train_pred_1, train_pred_0))
  
  # 提取最佳靈敏度和特異性，計算 TSS
  best_threshold_train <- coords(roc_obj_train, 'best', ret=c('threshold')) %>% pull() %>% median()

  # 根據最佳閾值進行分類
  val_pred_1 <- raster::extract(rst, xy_p_season_valsplit)
  val_pred_0 <- raster::extract(rst, xy_pa_season_sample_valsplit)
  val_actual_1 <- rep(1, nrow(xy_p_season_valsplit))
  val_actual_0 <- rep(0, nrow(xy_pa_season_sample_valsplit))
  val_predicted_classes <- ifelse(c(val_pred_1, val_pred_0) >= best_threshold_train, 1, 0)
  
  # 生成混淆矩陣
  predicted_factor <- factor(val_predicted_classes, levels = c(0, 1))
  actual_factor <- factor(c(val_actual_1, val_actual_0), levels = c(0, 1))
  confusion_matrix <- table(predicted_factor, actual_factor)
  
  # 計算TSS
  # 計算 Sensitivity 和 Specificity
  TP <- confusion_matrix[2, 2]  # True Positive
  TN <- confusion_matrix[1, 1]  # True Negative
  FP <- confusion_matrix[2, 1]  # False Positive
  FN <- confusion_matrix[1, 2]  # False Negative
  
  # 計算靈敏度（Sensitivity）和特異性（Specificity）和 Precision
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  N <- length(val_pred_0) + length(val_pred_1)
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

# load env season layers
load_env_season <- function(env_list, env_info, time, DeepSDM_conf){
  files_env <- c()
  i <- 1
  for(env in env_list){
    files_env[i] <- file.path(env_info$info[[env]][[time]]$tif_span_avg)
    i <- i + 1
  }
  env_season <- raster::stack(files_env)
  names(env_season) <- env_list
  for(env in env_list){
    if(!(env %in% DeepSDM_conf$training_conf$non_normalize_env_list)){
      values(env_season[[env]]) <- (values(env_season[[env]]) - env_info$info[[env]]$mean) / env_info$info[[env]]$sd
    }
  }
  return(env_season)
}

# load env allseason layers
load_env_allseason <- function(env_list, env_info, date_list_all_selectseason, DeepSDM_conf){
  
  date_list_all_selectseason <- as.vector(date_list_all_selectseason)
  
  env_allseason_list <- list()
  for(time in date_list_all_selectseason){
    print(time)
    files_env <- lapply(env_list, function(env) {
      file.path(env_info$info[[env]][[time]]$tif_span_avg)
    }) %>% unlist()
    
    env_allseason <- raster::stack(files_env)
    names(env_allseason) <- env_list
    lapply(env_list, function(env) {
      if(!(env %in% DeepSDM_conf$training_conf$non_normalize_env_list)) {
        values(env_allseason[[env]]) <<- (values(env_allseason[[env]]) - env_info$info[[env]]$mean) / env_info$info[[env]]$sd
      }
    })
    env_allseason_list[[time]] <- env_allseason
  }
  layer_means <- lapply(1:length(env_list), function(layer_index) {
    print(paste0('env_', layer_index))
    layer_stack <- stack(lapply(names(env_allseason_list), function(t) raster::raster(env_allseason_list[[t]], layer_index)))
    calc(layer_stack, fun = mean)
  })
  out <- raster::stack(layer_means)
  names(out) <- env_list
  return(out)
}

# set default value of logged variable
set_default_variable <- function(default_value = -9999){
  deepsdm_all_season_val <<- default_value
  deepsdm_all_season_train <<- default_value
  deepsdm_all_season_all <<- default_value
  
  maxent_season_season_all <<- default_value
  maxent_season_season_train <<- default_value
  maxent_season_season_val <<- default_value
  
  maxent_allseason_season_val <<- default_value
  maxent_allseason_season_train <<- default_value
  maxent_allseason_season_all <<- default_value

  maxent_all_season_val <<- default_value
  maxent_all_season_train <<- default_value
  maxent_all_season_all <<- default_value
  
  maxent_TSS <<- default_value
  deepsdm_TSS <<- default_value
  maxent_kappa <<- default_value
  deepsdm_kappa <<- default_value
  maxent_f1 <<- default_value
  deepsdm_f1 <<- default_value
  
  p_season <<- default_value
  p_valpart_season <<- default_value
  p_trainpart_season <<- default_value
  pa_valpart_season <<- default_value
  pa_trainpart_season <<- default_value
  p_season <<- ifelse(exists('xy_p_season'), nrow(xy_p_season), default_value)
  p_valpart_season <<- ifelse(exists('xy_p_season_valsplit'), nrow(xy_p_season_valsplit), default_value)
  p_trainpart_season <<- ifelse(exists('xy_p_season_trainsplit'), nrow(xy_p_season_trainsplit), default_value)
  pa_valpart_season <<- ifelse(exists('xy_pa_season_sample_valsplit'), nrow(xy_p_season_trainsplit), default_value)
  pa_trainpart_season <<- ifelse(exists('xy_pa_season_sample_trainsplit'), nrow(xy_p_season_trainsplit), default_value)
}
set_default_variable_allseason <- function(default_value = -9999){
  maxent_allseason_allseason_val <<- default_value
  maxent_allseason_allseason_train <<- default_value
  maxent_allseason_allseason_all <<- default_value
  
  p_allseason <<- default_value
  p_valpart_allseason <<- default_value
  p_trainpart_allseason <<- default_value
  pa_valpart_allseason <<- default_value
  pa_trainpart_allseason <<- default_value
  
  p_allseason <<- ifelse(exists('xy_p_allseason'), nrow(xy_p_allseason), default_value)
  p_valpart_allseason <<- ifelse(exists('xy_p_allseason_valsplit'), nrow(xy_p_allseason_valsplit), default_value)
  p_trainpart_allseason <<- ifelse(exists('xy_p_allseason_trainsplit'), nrow(xy_p_allseason_trainsplit), default_value)
  pa_valpart_allseason <<- ifelse(exists('xy_pa_allseason_sample_valsplit'), nrow(xy_pa_allseason_sample_valsplit), default_value)
  pa_trainpart_allseason <<- ifelse(exists('xy_pa_allseason_sample_trainsplit'), nrow(xy_pa_allseason_sample_trainsplit), default_value) 
}
set_default_variable_all <- function(default_value = -9999){
  maxent_all_all_val <<- default_value
  maxent_all_all_train <<- default_value
  maxent_all_all_all <<- default_value
  deepsdm_all_all_val <<- default_value
  deepsdm_all_all_train <<- default_value
  deepsdm_all_all_all <<- default_value
  
  p_all <<- default_value
  p_valpart_all <<- default_value
  p_trainpart_all <<- default_value
  pa_valpart_all <<- default_value
  pa_trainpart_all <<- default_value
  
  p_all <<- ifelse(exists('xy_p_all'), nrow(xy_p_all), default_value) 
  p_valpart_all <<- ifelse(exists('xy_p_all_valsplit'), nrow(xy_p_all_valsplit), default_value)
  p_trainpart_all <<- ifelse(exists('xy_p_all_trainsplit'), nrow(xy_p_all_trainsplit), default_value)
  pa_valpart_all <<- ifelse(exists('xy_pa_all_sample_valsplit'), nrow(xy_pa_all_sample_valsplit), default_value)
  pa_trainpart_all <<- ifelse(exists('xy_pa_all_sample_trainsplit'), nrow(xy_pa_all_sample_trainsplit), default_value)
}

# generate presence and pseudo-absence / absence point data of one season
generate_points <- function(num_pa = 10000){
  occ_rst_path <- file.path('.', sp_info$dir_base, sp_info$file_name[[species]][[time]])
  occ_rst <- raster::raster(occ_rst_path)
  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_season <<- xyFromCell(occ_rst, i_p_occ_rst) # x,y value from cells with presence records
  xy_p_season_trainsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_season_valsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))
  
  i_pa_season <- intersect(i_pa_occ_rst, i_extent)
  xy_pa_season <- xyFromCell(occ_rst, i_pa_season)
  if(num_pa == 'num_p'){
    i_pa_season_sample <- sample(i_pa_season, nrow(xy_p_season))
  }else{
    i_pa_season_sample <- sample(i_pa_season, num_pa)
  }
  xy_pa_season_sample <<- xyFromCell(occ_rst, i_pa_season_sample)
  xy_pa_season_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_season_sample, i_trainsplit))
  xy_pa_season_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_season_sample, i_valsplit))
}

# generate presence and pseudo-absence / absence point data of allseason
generate_points_allseason <- function(){
  
  occ_rst_path <- lapply(date_list_all_selectseason, function(time){
    file.path('.', sp_info$dir_base, sp_info$file_name[[species]][[time]])
  })
  occ_rst_path <- unlist(occ_rst_path)
  occ_rst <- raster::stack(occ_rst_path)
  occ_rst <- calc(occ_rst, function(x) {
    sign(sum(x, na.rm = TRUE))
  })
  
  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_allseason <<- xyFromCell(occ_rst, i_p_occ_rst) # x,y value from cells with presence records
  xy_p_allseason_trainsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_allseason_valsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))
  
  i_pa_allseason <- intersect(i_pa_occ_rst, i_extent)
  xy_pa_allseason <- xyFromCell(occ_rst, i_pa_allseason)
  i_pa_allseason_sample <- sample(i_pa_allseason, 10000)
  xy_pa_allseason_sample <<- xyFromCell(occ_rst, i_pa_allseason_sample)
  xy_pa_allseason_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_allseason_sample, i_trainsplit))
  xy_pa_allseason_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_allseason_sample, i_valsplit))
}

# generate presence and pseudo-absence / absence point data of all
generate_points_all <- function(date_list_all){
  
  occ_rst_path <- lapply(date_list_all, function(time){
    file.path('.', sp_info$dir_base, sp_info$file_name[[species]][[time]])
  })
  occ_rst_path <- unlist(occ_rst_path)
  occ_rst <- raster::stack(occ_rst_path)
  occ_rst <- calc(occ_rst, function(x) {
    sign(sum(x, na.rm = TRUE))
  })
  
  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_all <<- xyFromCell(occ_rst, i_p_occ_rst) # x,y value from cells with presence records
  xy_p_all_trainsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_all_valsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))
  
  i_pa_all <- intersect(i_pa_occ_rst, i_extent)
  xy_pa_all <- xyFromCell(occ_rst, i_pa_all)
  i_pa_all_sample <- sample(i_pa_all, 10000)
  xy_pa_all_sample <<- xyFromCell(occ_rst, i_pa_all_sample)
  xy_pa_all_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_all_sample, i_trainsplit))
  xy_pa_all_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_all_sample, i_valsplit))
}
# create folder 
create_folder <- function(dir){
  if (!dir.exists(dir)){
    dir.create(dir, recursive = T)
  }
}

# set default value of logged variable
set_default_variable <- function(default_value = -9999){
  f1_maxent_season_season_train <<- default_value
  f1_maxent_season_season_val <<- default_value
  f1_maxent_season_season_all <<- default_value
  
  f1_maxent_allseason_season_train <<- default_value
  f1_maxent_allseason_season_val <<- default_value
  f1_maxent_allseason_season_all <<- default_value
  
  f1_maxent_all_season_train <<- default_value
  f1_maxent_all_season_val <<- default_value
  f1_maxent_all_season_all <<- default_value
  
  f1_deepsdm_all_season_train <<- default_value
  f1_deepsdm_all_season_val <<- default_value
  f1_deepsdm_all_season_all <<- default_value
  
  threshold_maxent_season_season <<- default_value
  threshold_deepsdm_all_season <<- default_value
  threshold_maxent_allseason_season <<- default_value
  threshold_maxent_all_season <<- default_value
  
  p_trainpart_season <<- nrow(xy_p_season_trainsplit)
  p_valpart_season <<- nrow(xy_p_season_valsplit)
  p_season <<- nrow(xy_p_season)
  pa_trainpart_season <<- nrow(xy_pa_season_sample_trainsplit)
  pa_valpart_season <<- nrow(xy_pa_season_sample_valsplit)
  pa_season <<- nrow(xy_pa_season_sample)
}
set_default_variable_allseason <- function(default_value = -9999){
  f1_maxent_allseason_allseason_val <<- default_value
  f1_maxent_allseason_allseason_train <<- default_value
  f1_maxent_allseason_allseason_all <<- default_value
  
  threshold_maxent_allseason_allseason <<- default_value
  
  p_allseason <<- nrow(xy_p_allseason)
  p_valpart_allseason <<- nrow(xy_p_allseason_valsplit)
  p_trainpart_allseason <<- nrow(xy_p_allseason_trainsplit)
  pa_allseason <<- nrow(xy_pa_allseason_sample)
  pa_valpart_allseason <<- nrow(xy_pa_allseason_sample_valsplit)
  pa_trainpart_allseason <<- nrow(xy_pa_allseason_sample_trainsplit)
}
set_default_variable_all <- function(default_value = -9999){
  f1_maxent_all_all_val <<- default_value
  f1_maxent_all_all_train <<- default_value
  f1_maxent_all_all_all <<- default_value
  
  threshold_maxent_all_all <<- default_value
  
  p_all <<- nrow(xy_p_all)
  p_valpart_all <<- nrow(xy_p_all_valsplit)
  p_trainpart_all <<- nrow(xy_p_all_trainsplit)
  pa_all <<- nrow(xy_pa_all_sample)
  pa_valpart_all <<- nrow(xy_pa_all_sample_valsplit)
  pa_trainpart_all <<- nrow(xy_pa_all_sample_trainsplit)
}

# generate presence and pseudo-absence / absence point data of one season
generate_points <- function(){
  occ_rst_path <- file.path('.', sp_info$dir_base, sp_info$file_name[[species]][[time]])
  occ_rst <- raster::raster(occ_rst_path)
  i_p_occ_rst <- which(values(occ_rst) == 1)
  i_pa_occ_rst <- which(values(occ_rst) == 0)
  xy_p_season <<- xyFromCell(occ_rst, i_p_occ_rst) # x,y value from cells with presence records
  xy_p_season_trainsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_trainsplit))
  xy_p_season_valsplit <<- xyFromCell(occ_rst, intersect(i_p_occ_rst, i_valsplit))
  
  i_pa_season <- intersect(i_pa_occ_rst, i_extent)
  xy_pa_season <- xyFromCell(occ_rst, i_pa_season)
  i_pa_season_sample <- sample(i_pa_season, length(i_p_occ_rst))
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
  i_pa_allseason_sample <- sample(i_pa_allseason, length(i_p_occ_rst))
  xy_pa_allseason_sample <<- xyFromCell(occ_rst, i_pa_allseason_sample)
  xy_pa_allseason_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_allseason_sample, i_trainsplit))
  xy_pa_allseason_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_allseason_sample, i_valsplit))
}

# generate presence and pseudo-absence / absence point data of all
generate_points_all <- function(){
  
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
  i_pa_all_sample <- sample(i_pa_all, length(i_p_occ_rst))
  xy_pa_all_sample <<- xyFromCell(occ_rst, i_pa_all_sample)
  xy_pa_all_sample_trainsplit <<- xyFromCell(occ_rst, intersect(i_pa_all_sample, i_trainsplit))
  xy_pa_all_sample_valsplit <<- xyFromCell(occ_rst, intersect(i_pa_all_sample, i_valsplit))
}

# function to calculate f1 score
calculatef1 <- function(tp, fn, fp) {
  2 * tp / (2 * tp + fn + fp)
}

# function to operate raster and evaluate model predictions
prediction_process <- function(raster_prediction, xy_p_train, xy_pa_train, xy_p_val, xy_pa_val, xy_p, xy_pa, dir_output, file_name, precalculate_threshold = NULL) {
  
  if(is.null(precalculate_threshold)){
    
    if(nrow(xy_p_train) == 0 | nrow(xy_pa_train) == 0){
      return(list(f1_train = -9999, f1_val = -9999, f1_all = -9999, threshold = -9999))
    }
    
    p_trainsplit <- raster::extract(raster_prediction, xy_p_train)
    pa_trainsplit <- raster::extract(raster_prediction, xy_pa_train)
    
    eva_model <- evaluate(p = p_trainsplit, a = pa_trainsplit)
    f1_model <- calculatef1(eva_model@confusion[, 'tp'], eva_model@confusion[, 'fn'], eva_model@confusion[, 'fp'])
    threshold_model <- eva_model@t[which.max(f1_model)]
  }else{
    threshold_model <- precalculate_threshold
  }
  model_binary <- raster_prediction > threshold_model
  writeRaster(model_binary, file.path(dir_output, file_name), overwrite = TRUE)
  
  f1_train <- calculatef1(sum(raster::extract(model_binary, xy_p_train) == 1), 
                          sum(raster::extract(model_binary, xy_p_train) == 0), 
                          sum(raster::extract(model_binary, xy_pa_train) == 1))
  f1_val <- calculatef1(sum(raster::extract(model_binary, xy_p_val) == 1), 
                        sum(raster::extract(model_binary, xy_p_val) == 0), 
                        sum(raster::extract(model_binary, xy_pa_val) == 1))
  f1_all <- calculatef1(sum(raster::extract(model_binary, xy_p) == 1), 
                        sum(raster::extract(model_binary, xy_p) == 0), 
                        sum(raster::extract(model_binary, xy_pa) == 1))
  out <- list(f1_train = f1_train, f1_val = f1_val, f1_all = f1_all, threshold = threshold_model)
  out <- lapply(out, FUN = function(i){
    if(is.nan(i)){
      -9999
    }else{
      i
    }
  })
  return(out)
}

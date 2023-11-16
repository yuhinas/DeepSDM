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
plot_result <- function(sptime, species, xm, extent_binary, pall, pseasonavg, pseason, log_info, dir_timelog_png, dir_timelog_tif, timelog){
  png(file.path(dir_timelog_png, 
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
  
  # points(pall, pch = 16, col = 'black', cex = 1.5)
  # points(pseasonavg, pch = 16, col = 'blue', cex = 1.5)
  points(pseason, pch = 16, col = 'red', cex = 1.5)
  dev.off()
  writeRaster(xm * extent_binary, 
              file.path(dir_timelog_tif, 
                        sprintf('%s_%s_%s.tif', sptime, log_info, timelog)), 
              overwrite = T)
}

# function of plotting deepsdm result
plot_result_deepsdm <- function(sptime, species, xm, extent_binary, pall, pseasonavg, pseason, log_info, dir_timelog_png, timelog){
  png(file.path(dir_timelog_png, 
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
  
  # points(pall, pch = 16, col = 'black', cex = 1.5)
  # points(pseasonavg, pch = 16, col = 'blue', cex = 1.5)
  points(pseason, pch = 16, col = 'red', cex = 1.5)
  dev.off()
}

# function of auc_roc calculations
calculate_roc <- function(px, p, bg){
  if(nrow(p) == 0){
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

# load env season layers
load_env_season <- function(env_list, env_info, time){
  files_env <- c()
  i <- 1
  for(env in env_list){
    files_env[i] <- file.path(env_info$info[[env]][[time]]$tif_span_avg)
    i <- i + 1
  }
  env_season <- raster::stack(files_env)
  names(env_season) <- env_list
  return(env_season)
}

# set default value of logged variable
set_default_variable <- function(default_value = -9999){
  maxent_season_season_val <<- default_value
  maxent_season_seasonavg_val <<- default_value
  maxent_season_all_val <<- default_value
  maxent_seasonavg_season_val <<- default_value
  maxent_seasonavg_seasonavg_val <<- default_value
  maxent_seasonavg_all_val <<- default_value
  maxent_all_season_val <<- default_value
  maxent_all_seasonavg_val <<- default_value
  maxent_all_all_val <<- default_value
  deepsdm_all_season_val <<- default_value
  deepsdm_all_seasonavg_val <<- default_value
  deepsdm_all_all_val <<- default_value
  maxent_season_season_train <<- default_value
  maxent_season_seasonavg_train <<- default_value
  maxent_season_all_train <<- default_value
  maxent_seasonavg_season_train <<- default_value
  maxent_seasonavg_seasonavg_train <<- default_value
  maxent_seasonavg_all_train <<- default_value
  maxent_all_season_train <<- default_value
  maxent_all_seasonavg_train <<- default_value
  maxent_all_all_train <<- default_value
  deepsdm_all_season_train <<- default_value
  deepsdm_all_seasonavg_train <<- default_value
  deepsdm_all_all_train <<- default_value
  p_season <<- default_value
  p_valpart_season <<- default_value
  p_trainpart_season <<- default_value
  pa_valpart_season <<- default_value
  pa_trainpart_season <<- default_value
  p_seasonavg <<- default_value
  p_valpart_seasonavg <<- default_value
  p_trainpart_seasonavg <<- default_value
  pa_valpart_seasonavg <<- default_value
  pa_trainpart_seasonavg <<- default_value
  p_all <<- default_value
  p_valpart_all <<- default_value
  p_trainpart_all <<- default_value
  pa_valpart_all <<- default_value
  pa_trainpart_all <<- default_value
}

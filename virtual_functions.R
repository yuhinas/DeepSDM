# generate all-time environment layers
generate_envall <- function(env_list, extent_binary, medium_inf, version_path){
  df <- data.frame('env' = character(0), 'mean' = numeric(0), 'sd' = numeric(0))
  envavg_path <- file.path(version_path, 'env_avg')
  if(!dir.exists(envavg_path)){
    dir.create(envavg_path, recursive = TRUE)
  }
  for(env in env_list){
    result_env <- raster(extent_binary)
    values(result_env) <- 0
    value <- c()
    for(t in time){
      rst <- raster(medium_inf[[env]][[t]])
      value <- c(value, values(rst)[values(extent_binary) == 1])
      result_env <- result_env + rst
    }
    df[nrow(df)+1, ] <- c(env, mean(value, na.rm = TRUE), sd(value, na.rm = TRUE))
    result_env <- result_env / length(time)
    writeRaster(result_env, file.path(envavg_path, sprintf('%s_avg.tif', env)), overwrite = T)
  }
  write.csv(df, file.path(envavg_path, 'env_inf.csv'), row.names = FALSE)
  
  # load last 1 yr average data
  df <- read.csv(file.path(envavg_path, 'env_inf.csv'), row.names = 1)
  for(env in env_list){
    env_rst <- (raster(file.path(envavg_path, sprintf('%s_avg.tif', env))) - df[env, 'mean']) / df[env, 'sd']
    values(env_rst)[values(extent_binary) == 0] <- NA
    assign(paste0(env, '_all'), env_rst)
  }
  rsts <- lapply(env_list, function(env_list) get(paste0(env_list, "_all")))
  env_all <- stack(rsts)
  names(env_all) <- env_list
  return(list(df = df, env_all = env_all))
}

# generate environment layers for pca
# 'all': Use all-time average environment layers to create 
# 'random': Use a random time environment layers to create
generate_env_pca <- function(virtual_conf, env_all, time, env_list, df, medium_inf){
  if(virtual_conf$env_pca == 'all'){
    env_pca <- list(env_pca = env_all)
  }
  if(virtual_conf$env_pca == 'random'){
    time_random <- sample(time, 1)
    rsts_random <- lapply(env_list, function(env_list) {
      rst_normalize <- (raster(medium_inf[[env_list]][[time_random]]) - df[env_list, 'mean']) / df[env_list, 'sd']
      values(rst_normalize)[is.na(values(rst_normalize))] <- 0
      values(rst_normalize)[values(extent_binary) == 0] <- NA
      return(rst_normalize)
    })
    env_random <- stack(rsts_random)
    names(env_random) <- env_list
    env_pca <- list(env_pca = env_random, time_random = time_random)
  }
  return(env_pca)
}

# generate presence-absence binary map
generate_convertToPA <- function(virtual_conf, random_sp_time, random_method, random_beta, random_alpha, random_cutoff){
  if(virtual_conf$virtualspecies_conf$PA.method == 'probability'){
    random_sp_time_pa <- convertToPA(random_sp_time,
                                     PA.method = random_method, 
                                     beta = as.numeric(random_beta),
                                     alpha = as.numeric(random_alpha), 
                                     plot = T)
  }
  if(virtual_conf$virtualspecies_conf$PA.method == 'threshold'){
    random_sp_time_pa <- convertToPA(random_sp_time,
                                     PA.method = random_method, 
                                     beta = as.numeric(random_cutoff), 
                                     plot = T)
  }
  return(random_sp_time_pa)
}

# create folder 
create_folder <- function(dir){
  if (!dir.exists(dir)){
    dir.create(dir, recursive = T)
  }
}
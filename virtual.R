library(raster)
library(virtualspecies)
library(dplyr)
library(yaml)
library(rjson)
library(yaml)

set.seed(42)
# functions
generate_envall <- function(env_list, extent_binary, env_inf){
  df <- data.frame('env' = character(0), 'mean' = numeric(0), 'sd' = numeric(0))
  for(env in env_list){
    result_env <- raster(extent_binary)
    values(result_env) <- 0
    value <- c()
    for(t in time){
      rst <- raster(env_inf$info[[env]][[t]]$tif_span_avg)
      value <- c(value, values(rst)[values(extent_binary) == 1])
      result_env <- result_env + rst
    }
    df[nrow(df)+1, ] <- c(env, mean(value), sd(value))
    result_env <- result_env / length(time)
    writeRaster(result_env, file.path('virtual', version, sprintf('%s_1yr_avg.tif', env)), overwrite = T)
  }
  write.csv(df, file.path('virtual', version, 'env_inf.csv'), row.names = FALSE)
  
  # load last 1 yr average data
  df <- read.csv(file.path('virtual', version, 'env_inf.csv'), row.names = 1)
  for(env in env_list){
    env_rst <- (raster(file.path('virtual', version, sprintf('%s_1yr_avg.tif', env))) - df[env, 'mean']) / df[env, 'sd']
    assign(paste0(env, '_all'), env_rst)
  }
  rsts <- lapply(env_list, function(env_list) get(paste0(env_list, "_all")))
  env_all <- stack(rsts)
  names(env_all) <- env_list
  return(list(df = df, env_all = env_all))
}
generate_env_pca <- function(virtual_conf, env_all, time, env_list, env_inf){
  if(virtual_conf$env_pca == 'all'){
    env_pca <- list(env_pca = env_all)
  }
  if(virtual_conf$env_pca == 'random'){
    time_random <- sample(time, 1)
    rsts_random <- lapply(env_list, function(env_list) (raster(env_inf$info[[env_list]][[time_random]]$tif_span_avg) - df[env_list, 'mean']) / df[env_list, 'sd'])
    env_random <- stack(rsts_random)
    names(env_random) <- env_list
    env_pca <- list(env_pca = env_random, time_random = time_random)
  }
  return(env_pca)
}
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

virtual_conf <- yaml.load_file('./virtual_conf.yaml')
extent_binary <- raster(virtual_conf$extent_binary)
version <- virtual_conf$version
if(!dir.exists(sprintf('virtual/%s', version))){
  dir.create(sprintf('virtual/%s', version), recursive = TRUE)
}
write_yaml(virtual_conf, sprintf('virtual/%s/virtual_conf_%s.yaml', version, version))

# load environmental information 
env_inf <- fromJSON(file = virtual_conf$meta_json_files$env_inf)
k_inf <- fromJSON(file = virtual_conf$meta_json_files$k_inf)

# load DeepSDM model configurations
env_list <- sort(virtual_conf$env_list)

# last year average
time <- virtual_conf$time
prevalence <- virtual_conf$prevalence

# generate env_all
output <- generate_envall(env_list, extent_binary, env_inf)
df <- output$df
env_all <- output$env_all

for(i_sp in 1:virtual_conf$num_species){
  # i_sp <- 1
  sp <- paste0('sp', sprintf('%02d', i_sp))
  
  # differenct virtual_conf$env_pca
  env_pca <- generate_env_pca(virtual_conf, env_all, time, env_list, env_inf)
  
  # use 21 year average environment to conduct PCA
  random_sp <- generateRandomSp(env_pca$env_pca, 
                                approach = virtual_conf$virtualspecies_conf$approach, 
                                realistic.sp = virtual_conf$virtualspecies_conf$realistic.sp, 
                                convert.to.PA = virtual_conf$virtualspecies_conf$convert.to.PA, 
                                PA.method = virtual_conf$virtualspecies_conf$PA.method,
                                species.prevalence = prevalence[((i_sp-1)%%length(prevalence) + 1)],
                                plot = T)
  # create folders
  dir_sp <- sprintf('virtual/%s/%s', version, sp)
  if(!dir.exists(dir_sp)){
    dir.create(dir_sp)
  }  
  
  # if random time env is set, log the time_random
  if(virtual_conf$env_pca == 'random'){
    random_sp[['time_used']] <- env_pca$time_random
  }
  save(random_sp, file = file.path(dir_sp, sprintf('%s.RData', sp)))
  
  random_means <- random_sp$details$means
  random_sds <- random_sp$details$sds
  random_pca <- random_sp$details$pca
  random_method <- random_sp$PA.conversion['conversion.method']
  random_cutoff <- random_sp$PA.conversion['cutoff']
  random_beta <- random_sp$PA.conversion['beta']
  random_alpha <- random_sp$PA.conversion['alpha']
  
  #specific time data
  for(t in time){
    # t <- time[1]
    rsts_time <- lapply(env_list, function(env_list) (raster(env_inf$info[[env_list]][[t]]$tif_span_avg) - df[env_list, 'mean']) / df[env_list, 'sd'])
    env_time <- stack(rsts_time)
    names(env_time) <- env_list
    
    # create folders
    dir_sp_time <- sprintf('virtual/%s/%s/%s', version, sp, t)
    if(!dir.exists(dir_sp_time)){
      dir.create(dir_sp_time)
    }  
    # generate data based on the PCA result and the environment of specific time
    random_sp_time <- generateSpFromPCA(env_time, 
                                        means = random_means, 
                                        sds = random_sds, 
                                        pca = random_pca, 
                                        plot = F)
    save(random_sp_time, file = file.path(dir_sp_time, sprintf('%s_%s.RData', sp, t)))
    
    # convert continuous map to presence-absence map
    random_sp_time_pa <- generate_convertToPA(virtual_conf, random_sp_time, random_method, random_beta, random_alpha, random_cutoff)
    save(random_sp_time_pa, file = file.path(dir_sp_time, sprintf('%s_%s_pa.RData', sp, t)))
    pa_raster <- random_sp_time_pa$pa.raster
    values(pa_raster)[values(extent_binary) == 0] <- NA
    writeRaster(pa_raster, file.path(dir_sp_time, sprintf('%s_%s_pa.tif', sp, t)), overwrite = T)
    
    # CONDITION I: a random sampling based on the specific percentage of all grids of Taiwan
    for(p in virtual_conf$pa_percentage){  
      # p <- virtual_conf$pa_percentage[1]
      max_value <- sum(values(extent_binary) == 1)
      sampleocc <- sampleOccurrences(random_sp_time_pa, 
                                     n = round(p * max_value), 
                                     type = 'presence-absence', 
                                     plot = F)
      save(sampleocc, file = file.path(dir_sp_time, sprintf('sampleocc_%s_%s_%s.RData', sp, t, p)))
      points_ideal <- sampleocc$sample.points
      
      # save the long and lat of all points
      write.csv(points_ideal, file = file.path(dir_sp_time, sprintf('ideal_points_%s_%s_%s.csv', sp, t, p)), row.names = F)
      
      # save the raster of the occurrence
      occurrence_ideal <- raster(random_sp_time_pa$pa.raster)
      values(occurrence_ideal)[!is.na(values(occurrence_ideal))] <- 0
      occurrence_ideal[cellFromXY(occurrence_ideal, points_ideal)] <- points_ideal$Observed
      writeRaster(occurrence_ideal, file.path(dir_sp_time, sprintf('ideal_map_%s_%s_%s.tif', sp, t, p)), overwrite = T)
    }
    
    # CONDITION II: biased sampling based on real situation. Only the grids with observation in real world count. 
    # only calculate the whole real situation
    sampleocc <- sampleOccurrences(random_sp_time_pa, 
                                   n = max_value, 
                                   type = 'presence-absence', 
                                   plot = F)
    save(sampleocc, file = file.path(dir_sp_time, sprintf('sampleocc_%s_%s_real.RData', sp, t)))
    points_real = sampleocc$sample.points
    k2 <- raster(file.path(k_inf$dir_base, k_inf$file_name[[sprintf('%s-01', t)]]))
    values(k2)[values(k2) == 0] <- NA
    points_real <- points_real[!is.na(raster::extract(k2, points_real[c('x', 'y')])), ]
    
    # save the long and lat of all points
    write.csv(points_real, file = file.path(dir_sp_time, sprintf('real_points_%s_%s.csv', sp, t)), row.names = F)
    
    # save the raster of the occurrence
    occurrence_real <- raster(random_sp$pa.raster)
    values(occurrence_real)[!is.na(values(occurrence_real))] <- 0
    occurrence_real[cellFromXY(occurrence_real, points_real)] <- points_real$Observed
    writeRaster(occurrence_real, file.path(dir_sp_time, sprintf('real_map_%s_%s.tif', sp, t)), overwrite = T)
  }
}
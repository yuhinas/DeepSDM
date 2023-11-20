library(raster)
library(virtualspecies)
library(dplyr)
library(yaml)
library(rjson)

set.seed(42)

source('virtual_functions.r')

virtual_conf <- yaml.load_file('virtual_conf.yaml')
extent_binary <- raster(virtual_conf$extent_binary)
version <- virtual_conf$version

version_path <- file.path('virtual', version)
create_folder(version_path)
write_yaml(virtual_conf, file.path(version_path, sprintf('virtual_conf_%s.yaml', version)))

# load environmental information 
env_inf <- fromJSON(file = virtual_conf$meta_json_files$env_inf)
k_inf <- fromJSON(file = virtual_conf$meta_json_files$k_inf)
medium_inf <- fromJSON(file = virtual_conf$meta_json_files$medium)

# load DeepSDM model configurations
env_list <- sort(virtual_conf$env_list)

# last year average
time <- virtual_conf$time
prevalence <- virtual_conf$prevalence

# generate env_all
output <- generate_envall(env_list, extent_binary, medium_inf, version_path)
df <- output$df
env_all <- output$env_all

# species information (virtual species)
# sp_inf_bias_virtual: maps with geological bias
# sp_inf_true_virtual: maps without any bias
sp_inf_bias_virtual <- list(dir_base = file.path('virtual', version, 'medium'), file_name = list())
sp_inf_true_virtual <- list(dir_base = file.path('virtual', version, 'medium'), file_name = list())

# load species filter csv (real world)
sp_filter_path <- file.path('workspace', 'species_data', 'occurrence_data', 'species_occurrence_filter.csv')
sp_filter <- read.csv(sp_filter_path)

for(i_sp in 1:virtual_conf$num_species){
  # i_sp <- 1
  sp <- paste0('sp', sprintf('%02d', i_sp))
  
  # differenct virtual_conf$env_pca
  env_pca <- generate_env_pca(virtual_conf, env_all, time, env_list, df, medium_inf)
  
  # use 21 year average environment to conduct PCA
  random_sp <- generateRandomSp(env_pca$env_pca, 
                                approach = virtual_conf$virtualspecies_conf$approach, 
                                realistic.sp = virtual_conf$virtualspecies_conf$realistic.sp, 
                                convert.to.PA = virtual_conf$virtualspecies_conf$convert.to.PA, 
                                PA.method = virtual_conf$virtualspecies_conf$PA.method,
                                species.prevalence = prevalence[((i_sp-1)%%length(prevalence) + 1)],
                                plot = T)
  # create folders
  sp_path <- file.path(version_path, 'medium', sp)
  create_folder(sp_path)
  
  sp_inf_bias_virtual[['file_name']][[sp]] <- list()
  sp_inf_true_virtual[['file_name']][[sp]] <- list()
  
  # if random time env is set, log the time_random
  if(virtual_conf$env_pca == 'random'){
    random_sp[['time_used']] <- env_pca$time_random
  }
  save(random_sp, file = file.path(sp_path, sprintf('%s.RData', sp)))
  
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
    rsts_time <- lapply(env_list, function(env_list){
      rst_normalize <- (raster(medium_inf[[env_list]][[t]]) - df[env_list, 'mean']) / df[env_list, 'sd']
      values(rst_normalize)[is.na(values(rst_normalize))] <- 0
      values(rst_normalize)[values(extent_binary) == 0] <- NA
      return(rst_normalize)
      })
    env_time <- stack(rsts_time)
    names(env_time) <- env_list
    
    # create folders
    sp_time_path <- file.path(sp_path, t)
    create_folder(sp_time_path)
    
    # generate data based on the PCA result and the environment of specific time
    random_sp_time <- generateSpFromPCA(env_time, 
                                        means = random_means, 
                                        sds = random_sds, 
                                        pca = random_pca, 
                                        plot = F)
    save(random_sp_time, file = file.path(sp_time_path, sprintf('%s_%s.RData', sp, t)))
    
    # convert continuous map to presence-absence map
    random_sp_time_pa <- generate_convertToPA(virtual_conf, random_sp_time, random_method, random_beta, random_alpha, random_cutoff)
    save(random_sp_time_pa, file = file.path(sp_time_path, sprintf('%s_%s_pa.RData', sp, t)))
    pa_raster <- raster(random_sp_time_pa$pa.raster)
    values(pa_raster)[values(extent_binary) == 0] <- -9999
    NAvalue(pa_raster) <- -9999
    writeRaster(pa_raster, file.path(sp_time_path, sprintf('%s_%s_pa.tif', sp, t)), overwrite = T)
    
    # CONDITION I: a random sampling based on the specific percentage of all grids of Taiwan
    for(p in virtual_conf$pa_percentage){  
      # p <- virtual_conf$pa_percentage[1]
      max_value <- sum(values(extent_binary) == 1)
      sampleocc <- sampleOccurrences(random_sp_time_pa, 
                                     n = round(p * max_value), 
                                     type = 'presence-absence', 
                                     plot = F)
      save(sampleocc, file = file.path(sp_time_path, sprintf('sampleocc_true_%s_%s_p%s.RData', sp, t, p)))
      points_true_p <- sampleocc$sample.points
      
      # save the long and lat of all points
      write.csv(points_true_p, file = file.path(sp_time_path, sprintf('occpoints_true_%s_%s_p%s.csv', sp, t, p)), row.names = F)
      
      # save the raster of the occurrence
      occurrence_true <- raster(random_sp_time_pa$pa.raster)
      values(occurrence_true)[!is.na(values(occurrence_true))] <- 0
      occurrence_true[cellFromXY(occurrence_true, points_true_p)] <- points_true_p$Observed
      values(occurrence_true)[values(extent_binary) == 0] <- -9999
      NAvalue(occurrence_true) <- -9999
      writeRaster(occurrence_true, file.path(sp_time_path, sprintf('true_map_%s_%s_p%s.tif', sp, t, p)), overwrite = T)
    }
    
    # CONDITION II: biased sampling based on real situation. Only the grids with observation in real world count. 
    # only calculate the whole real situation
    sampleocc <- sampleOccurrences(random_sp_time_pa, 
                                   n = max_value, 
                                   type = 'presence-absence', 
                                   plot = F)
    save(sampleocc, file = file.path(sp_time_path, sprintf('sampleocc_bias_%s_%s.RData', sp, t)))
    points_bias = sampleocc$sample.points
    
    # generate k_rst
    t_split <- strsplit(t, '-')[[1]]
    sp_filter_time <- sp_filter[((sp_filter['year'] == t_split[1]) & (sp_filter['month'] == as.numeric(t_split[2]))), ]
    k_rst <- raster(extent_binary)
    idx <- cellFromXY(k_rst, sp_filter_time[, c('decimalLongitude', 'decimalLatitude')])
    k_rst[idx] <- 1 # k_rst: 1 means with occurrence records; NA means no occurrence records
    
    points_bias <- points_bias[!is.na(raster::extract(k_rst, points_bias[c('x', 'y')])), ]
    
    # save the long and lat of all points
    write.csv(points_bias, file = file.path(sp_time_path, sprintf('occpoints_bias_%s_%s.csv', sp, t)), row.names = F)
    
    # save the raster of the occurrence
    occurrence_bias <- raster(random_sp$pa.raster)
    values(occurrence_bias)[!is.na(values(occurrence_bias))] <- 0
    occurrence_bias[cellFromXY(occurrence_bias, points_bias)] <- points_bias$Observed
    values(occurrence_bias)[values(extent_binary) == 0] <- -9999
    NAvalue(occurrence_bias) <- -9999
    writeRaster(occurrence_bias, file.path(sp_time_path, sprintf('bias_map_%s_%s.tif', sp, t)), overwrite = T)
    sp_inf_bias_virtual[['file_name']][[sp]][[t]] <- file.path(sp, t, sprintf('bias_map_%s_%s.tif', sp, t))
    sp_inf_true_virtual[['file_name']][[sp]][[t]] <- file.path(sp, t, sprintf('%s_%s_pa.tif', sp, t))
  }
}
write_yaml(sp_inf_bias_virtual, file = file.path(version_path, 'species_information_medium_bias_virtual.yaml'))
write_yaml(sp_inf_true_virtual, file = file.path(version_path, 'species_information_medium_true_virtual.yaml'))
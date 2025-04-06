import os
import yaml
import rasterio
import json
import pandas as pd
from types import SimpleNamespace
from sklearn.covariance import EllipticEnvelope
import numpy as np
from matplotlib.patches import Ellipse
import cv2
import matplotlib as mpl
from joblib import Parallel, delayed
import h5py
import feather
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr, ttest_rel
import glob
from rasterio.transform import xy
import statsmodels.api as sm

class PlotUtlis():
    def __init__(self, run_id, exp_id, niche_rst_size):
        self.niche_rst_size = niche_rst_size
        
        
        
        # 路徑
        self.conf_path = os.path.join('mlruns', exp_id, run_id, 'artifacts', 'conf')
        self.predicts_path = os.path.join('predicts', run_id)
        self.predict_maxent_path = os.path.join('predict_maxent', run_id)
        
        self.deepsdm_h5_path = os.path.join('predicts', run_id, 'h5', '[SPECIES]', '[SPECIES].h5')
        self.maxent_h5_path = os.path.join('predict_maxent', run_id, 'h5', 'all', '[SPECIES]', '[SPECIES].h5')  
        
        self.attention_h5_path = os.path.join('predicts', run_id, 'attention', '[SPECIES]', '[SPECIES]_[DATE]_attention.h5')
        
        
        self.traitdataset_taxon_path = os.path.join('dwca-trait_454-v1.68', 'taxon.txt')
        self.traitdataset_mesurement_path = os.path.join('dwca-trait_454-v1.68', 'measurementorfacts.txt')
        
        self.performance_indicator_multithreshold = os.path.join('predict_maxent', run_id, 'all_indicator_result_all_season_num_pa*.csv')
        self.performance_indicator_singlethreshold = os.path.join('predict_maxent', run_id, 'only_threshold_depend_indi_*.csv')
        
        
        
        
        # 變數
        # DeepSDM configurations
        self.DeepSDM_conf_path = os.path.join(self.conf_path, 'DeepSDM_conf.yaml')
        with open(self.DeepSDM_conf_path, 'r') as f:
            self.DeepSDM_conf = SimpleNamespace(**yaml.load(f, Loader = yaml.FullLoader))
        
        # load extent binary map
        with rasterio.open(os.path.join(self.conf_path, 'extent_binary.tif'), 'r') as f:
            self.transform = f.transform
            self.height, self.width = f.shape
            self.lon_min, self.lat_max = rasterio.transform.xy(self.transform, 0, 0)  # 左上角
            self.lon_max, self.lat_min = rasterio.transform.xy(self.transform, self.height - 1, self.width - 1)  # 右下角
            self.extent_binary = f.read(1)
        self.extent_binary_extent = [self.lon_min, self.lon_max, self.lat_min, self.lat_max]
        
        # species occurrence points
        with open(os.path.join(self.predicts_path, 'sp_inf.json'), 'r') as f:
            self.sp_info = json.load(f)
        with open(os.path.join(self.conf_path, 'cooccurrence_vector.json')) as f:
            self.coocc_vector = json.load(f)
        with open(os.path.join(self.conf_path, 'env_information.json')) as f:
            self.env_info = json.load(f)        
        
        self.species_list_train = sorted(self.DeepSDM_conf.training_conf['species_list_train'])
        self.species_list_predict = sorted(self.DeepSDM_conf.training_conf['species_list_predict'])
        self.species_list_all = sorted(list(self.coocc_vector.keys()))

        self.date_list_predict = self.DeepSDM_conf.training_conf['date_list_predict']
        self.date_list_train = self.DeepSDM_conf.training_conf['date_list_train']      
        
        self.species_occ = pd.read_csv(self.DeepSDM_conf.cooccurrence_conf['sp_filter_from'])
        
        elev_path = os.path.join(self.env_info['dir_base'], self.env_info['info']['elev'][self.date_list_train[0]]['tif_span_avg'])
        with rasterio.open(elev_path, 'r') as f:
            self.elev_rst = f.read(1)
        
        self.coocc_counts = pd.read_csv('./workspace/species_data/cooccurrence_data/cooccurrence.csv', sep = '\t')
            
        self.env_list = self.DeepSDM_conf.training_conf['env_list']
        env_list_change = {
            'clt': 'Cloud area fraction',
            'hurs': 'Relative humidity',
            'pr': 'Precipitation',
            'rsds': 'Shortwave radiation',
            'sfcWind': 'Wind speed',
            'tas': 'Temperature',
            'EVI': 'EVI',
            'landcover_PC00': 'Landcover (LPC1)',
            'landcover_PC01': 'Landcover (LPC2)',
            'landcover_PC02': 'Landcover (LPC3)',
            'landcover_PC03': 'Landcover (LPC4)',
            'landcover_PC04': 'Landcover (LPC5)', 
        }
        self.env_list_detail = [env_list_change[i] for i in self.env_list]
        
        self.x_pca = 1
        self.y_pca = 2
        self.color_list = ['#4daf4a', '#984ea3', '#ff7f00']
        
        
        
        # 子資料夾路徑
        self.plot_path_embedding_dimension_reduction = os.path.join('plots', run_id, 'Fig2_embedding_dimension_reduction')
        self.plot_path_embedding_correlation = os.path.join('plots', run_id, 'Fig2_embedding_correlation')        
        self.plot_path_attention = os.path.join('plots', run_id, 'Fig3_attention')
        self.plot_path_attentionstats = os.path.join('plots', run_id, 'FigS2_attentionstats')
        self.plot_path_nichespace = os.path.join('plots', run_id, 'Fig4_nichespace')
        self.plot_path_envcorrelation = os.path.join('plots', run_id, 'FigS3_envcorrelation')
        self.plot_path_nichespace_clustering = os.path.join('plots', run_id, 'Fig5_nichespace_clustering')
        self.plot_path_nichespace_clustering_test = os.path.join('plots', run_id, 'FigS4_nichespace_clustering_test')
        self.plot_path_cph = os.path.join('plots', run_id, 'Fig6_cph')
        
        
        # 輸出路徑
        # output path
        self.avg_elev_path = os.path.join(self.plot_path_embedding_dimension_reduction, 'avg_elevation.csv')
        self.df_attention_path = os.path.join(self.plot_path_attention, 'df_attention.csv')
        self.env_pca_loadings_path = os.path.join(self.plot_path_nichespace, 'env_pca_loadings.csv')
        self.df_env_corr_path = os.path.join(self.plot_path_nichespace, 'env_correlation.csv')
        self.df_env_pca_path = os.path.join(self.plot_path_nichespace, 'df_env_pca.feather')
        self.pc_info_path = os.path.join(self.plot_path_nichespace, 'pc_info.yaml')
        self.bin_info_path = os.path.join(self.plot_path_nichespace, 'bin_info.yaml')
        self.extent_info_path = os.path.join(self.plot_path_nichespace, 'extent_info.yaml')
        self.plot_path_df_species = os.path.join(self.plot_path_nichespace, 'df_species', '[SPECIES].feather')
        self.plot_path_nichespace_h5 = os.path.join(self.plot_path_nichespace, 'h5', '[SPECIES].h5')
        self.plot_path_nichespace_png_sp = os.path.join(self.plot_path_nichespace, 'png', '[SPECIES]', '[SPECIES]_nichespace_[SUFFIX].png')        
        self.df_grid_path = os.path.join(self.plot_path_nichespace, 'df_grid.feather')
        self.df_spearman_path = os.path.join(self.plot_path_nichespace, 'df_spearman.csv')
        self.cluster_labels_path = os.path.join(self.plot_path_nichespace_clustering, 'cluster_labels.yaml')
        self.cluster_avg_nichespace_path = os.path.join(self.plot_path_nichespace_clustering, 'cluster_avg_nichespace.yaml')
        self.df_nichespace_center_coordinate_path = os.path.join(self.plot_path_nichespace_clustering, 'nichespecies_center_coordinate.csv')
        self.df_spearman_ecogeo_path = os.path.join(self.plot_path_cph, 'df_spearman_ecogeo.csv')
        self.niche_beta_params_path = os.path.join(self.plot_path_cph, 'niche_beta_params.json')
        self.geographical_beta_params_path = os.path.join(self.plot_path_cph, 'geographical_beta_params.json')
        self.beta_cluster_result_path = os.path.join(self.plot_path_cph, '[CENTER_TYPE]_cluster_[CLUSTER]_beta_result.txt')
        self.niche_beta_cluster_params_path = os.path.join(self.plot_path_cph, 'niche_beta_cluster_params.json')
        self.geographical_beta_cluster_params_path = os.path.join(self.plot_path_cph, 'geographical_beta_cluster_params.json')        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.load_existing_files()

        
        
    # For Fig2
    def calculate_avg_elev(self):
        # Initialize dictionary to store average elevation per species
        species_avg_elevation = {}

        # Calculate average elevation for each species
        for i, sp in enumerate(self.species_list_all):
            print(f'Processing {i}: {sp}\r', end='')

            # Filter occurrence records for current species
            occ_sp = self.species_occ[self.species_occ.species == sp]

            # Convert coordinates (longitude, latitude) to raster indices
            rows, cols = rasterio.transform.rowcol(
                self.transform,
                occ_sp['decimalLongitude'].values,
                occ_sp['decimalLatitude'].values
            )

            # Extract elevation values at corresponding raster indices
            elevations = self.elev_rst[rows, cols]

            # Replace invalid elevation data (-9999) with NaN
            valid_elevations = np.where(elevations == -9999, np.nan, elevations)

            # Calculate mean elevation, handle case with no valid data
            if np.any(~np.isnan(valid_elevations)):
                species_avg_elevation[sp] = np.nanmean(valid_elevations)
            else:
                species_avg_elevation[sp] = None

        # Convert results into DataFrame
        avg_elevation_df = pd.DataFrame.from_dict(
            species_avg_elevation, orient='index', columns=['AverageElevation']
        )
        avg_elevation_df.reset_index(inplace=True)
        avg_elevation_df.rename(columns={'index': 'Species'}, inplace=True)

        return avg_elevation_df
    
    # For Fig2
    def set_cooccurrence_counts(self, species_list):
        coocc_counts = self.coocc_counts.copy()
        
        # remove sp1 == sp2
        coocc_counts = coocc_counts[(coocc_counts.sp1 != coocc_counts.sp2)].reset_index(drop = True)

        # delete records with "counts" equals 0
        coocc_counts = coocc_counts[coocc_counts.counts != 0].reset_index(drop = True)

        # delete records not in species_list
        coocc_counts = coocc_counts[(coocc_counts.sp1.isin(species_list)) & (coocc_counts.sp2.isin(species_list))].reset_index(drop = True)
        
        return coocc_counts
    
    # For Fig3
    def create_attention_df(self):
        
        def process_species(species, env_list, date_list_predict):
            sp_attention = np.zeros([len(env_list), 560, 336], dtype=np.float32)
            for date in date_list_predict:
                with h5py.File(self.attention_h5_path.replace('[SPECIES]', species).replace('[DATE]', date), 'r') as hf:
                    for i_env, env in enumerate(env_list):
                        sp_attention[i_env] += hf[env][:]  # 直接累加

            # 處理數據，將小於0的值設為 NaN 並計算平均值
            sp_attention[sp_attention < 0] = np.nan
            species_attention = np.nanmean(sp_attention, axis=(1, 2)) / len(date_list_predict)
            return [species] + species_attention.tolist()

        # 並行處理所有物種
        results = Parallel(n_jobs=-1)(
            delayed(process_species)(species, self.env_list, self.date_list_predict) 
            for species in self.species_list_predict
        )

        # 將結果轉換為 DataFrame
        df_attention = pd.DataFrame(results, columns=['Species'] + self.env_list).set_index('Species')
        
        return df_attention
        
    # For Fig3
    def get_species_habitat(self, df_attention_order):
        
        taxon_df = pd.read_csv(self.traitdataset_taxon_path, delimiter = '\t')
        measurement_df = pd.read_csv(self.traitdataset_mesurement_path, delimiter = '\t')

        habitat_related_df = measurement_df[
            measurement_df['measurementType'].str.contains('Habitat', na=False, case=False)
        ]

        merged_habitat_df = habitat_related_df.merge(taxon_df, left_on = 'id', right_on = 'taxonID', how = 'left')
        merged_habitat_df['measurementValue'] = merged_habitat_df['measurementValue'].astype(float)
        best_habitat = merged_habitat_df.loc[merged_habitat_df.groupby('taxonID')['measurementValue'].idxmax()]
        species_habitat_df = best_habitat[['scientificName', 'measurementType']].copy()
        name_to_change = {'Porzana fusca': 'Zapornia fusca', 
                          'Gallirallus striatus': 'Lewinia striata', 
                          'Stachyridopsis ruficeps': 'Cyanoderma ruficeps', 
                          'Pomatorhinus erythrocnemis': 'Erythrogenys erythrocnemis', 
                          'Alcippe brunnea': 'Schoeniparus brunneus', 
                          'Ictinaetus malayensis': 'Ictinaetus malaiensis', 
                          'Poecile varius': 'Sittiparus castaneoventris', 
                          'Parus holsti': 'Machlolophus holsti', 
                          'Turdus poliocephalus': 'Turdus niveiceps', 
                          'Garrulax poecilorhynchus': 'Pterorhinus poecilorhynchus', 
                          'Garrulax ruficeps': 'Pterorhinus ruficeps', 
                          'Glaucidium brodiei': 'Taenioptynx brodiei', 
                          'Tarsiger indicus': 'Tarsiger formosanus'}

        df_attention_sp_name = list(df_attention_order.index)
        for i, name in enumerate(df_attention_sp_name):
            if name in name_to_change:
                df_attention_sp_name[i] = name_to_change[name]

        habitat = [species_habitat_df.measurementType[species_habitat_df.scientificName == sp].values.tolist()[0] for sp in df_attention_sp_name]
        
        return habitat
           
    # For Fig4
    def set_df_env(self):
        df_env = pd.DataFrame()
        for date in self.date_list_train:
            df_date = pd.DataFrame()
            for env in self.env_list:
                env_path = os.path.join(self.env_info['dir_base'], self.env_info['info'][env][date]['tif_span_avg'])
                with rasterio.open(env_path, 'r') as f:
                    if env in self.DeepSDM_conf.training_conf['non_normalize_env_list']:
                        env_value = (f.read(1))[self.extent_binary == 1].flatten()
                    env_value = ((f.read(1))[self.extent_binary == 1].flatten() - self.env_info['info'][env]['mean']) / self.env_info['info'][env]['sd']
                df_date[env] = env_value
            df_env = pd.concat([df_env, df_date], ignore_index = True)
        
        return df_env
    
    # For Fig4
    def set_df_env_pca(self, pca):
        # Initialize final DataFrame
        df_env_pca = []

        # Load all raster data once and process
        for date in self.date_list_train:
            print(f'\rProcessing date: {date}', end='')
            env_data = []
            original_col_names = []  # Temporary column names
            final_col_names = []     # Column names for final save

            for env in self.env_list:
                # Construct file path for raster
                env_path = os.path.join(self.env_info['dir_base'], self.env_info['info'][env][date]['tif_span_avg'])

                # Load raster data
                with rasterio.open(env_path, 'r') as f:
                    raster_data = f.read(1)[self.extent_binary == 1].flatten()

                # Normalize environmental data
                if env not in self.DeepSDM_conf.training_conf['non_normalize_env_list']:
                    raster_data = (raster_data - self.env_info['info'][env]['mean']) / self.env_info['info'][env]['sd']

                env_data.append(raster_data)
                original_col_names.append(f'{env}')  # Temporary name
                final_col_names.append(f'{env}_{date}')  # Final name

            # Combine all environmental variables into a single DataFrame with temporary names
            df_season = pd.DataFrame(np.array(env_data).T, columns = original_col_names)

            # Apply PCA to the entire batch
            df_pca = pd.DataFrame(
                data = pca.transform(df_season), 
                columns=[f'PC{i+1:02d}_{date}' for i in range(len(pca.components_))]
            )

            # Rename columns to final names for saving
            df_season.columns = final_col_names

            # Append processed data to the list
            df_env_pca.append(pd.concat([df_season, df_pca], axis = 1))

        # Combine all results into a single DataFrame
        df_env_pca = pd.concat(df_env_pca, axis = 1, ignore_index = False)

        # Save DataFrame to feather
        feather.write_dataframe(df_env_pca, self.df_env_pca_path)
        
        return df_env_pca
        print("\nProcessing complete. File saved.")        
        
    # For Fig4
    def set_df_species(self):
        
        create_folder(os.path.dirname(self.plot_path_df_species))
        
        # Function to process a single species
        def process_species(species):
            print(f'Processing: {species}')
            series_dict = {}

            # Maxent all_all prediction
            maxent_h5_species_path = self.maxent_h5_path.replace('[SPECIES]', species)
            if os.path.exists(maxent_h5_species_path):
                with h5py.File(maxent_h5_species_path, 'r') as hf:
                    maxent_all_all_value = (hf['all'][:])[self.extent_binary == 1].flatten()
                    series_dict[f'maxent_all_all_{species}'] = maxent_all_all_value

            # Process predictions for all dates
            for date in self.date_list_predict:
                deepsdm_h5_species_path = self.deepsdm_h5_path.replace('[SPECIES]', species)
                if os.path.exists(deepsdm_h5_species_path):
                    with h5py.File(deepsdm_h5_species_path, 'r') as hf:
                        deepsdm_all_month_value = (hf[date][:])[self.extent_binary == 1].flatten()
                        series_dict[f'deepsdm_all_month_{species}_{date}'] = deepsdm_all_month_value

                if os.path.exists(maxent_h5_species_path):
                    with h5py.File(maxent_h5_species_path, 'r') as hf:
                        maxent_all_month_value = (hf[date][:])[self.extent_binary == 1].flatten()
                        series_dict[f'maxent_all_month_{species}_{date}'] = maxent_all_month_value

                # Occurrence points
                occ_path = os.path.join(self.sp_info['dir_base'], self.sp_info['file_name'][species][date])
                if os.path.exists(occ_path):
                    with rasterio.open(occ_path, 'r') as f:
                        occ_value = (f.read(1))[self.extent_binary == 1].flatten()
                        series_dict[f'occ_{species}_{date}'] = occ_value.astype(int)

            # Combine all data into a single DataFrame
            df_species = pd.DataFrame(series_dict)

            # Save to .feather file
            output_path = self.plot_path_df_species.replace('[SPECIES]', species)
            feather.write_dataframe(df_species, output_path)
            print(f'Finished: {species}')

        # Parallelize processing for all species
        with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust max_workers based on CPU cores
            executor.map(process_species, self.species_list_predict)

        print("All species processed successfully!")
        
    # For Fig4
    def set_pc_bin_extent_info(self, df_env_pca):
        # 各個pc軸的最大值與最小值
        pc_info = dict(
            **{
                f'PC{(n_pc+1):02d}_max': df_env_pca.loc[:, df_env_pca.columns.str.startswith(f'PC{(n_pc+1):02d}')].max().max().tolist() for n_pc in range(len(self.env_list))
            }, 
            **{
                f'PC{(n_pc+1):02d}_min': df_env_pca.loc[:, df_env_pca.columns.str.startswith(f'PC{(n_pc+1):02d}')].min().min().tolist() for n_pc in range(len(self.env_list))
            }
        )

        # 各個pc軸的切割raster資訊
        bin_info = {
            f'PC{(n_pc+1):02d}_bins': np.linspace(pc_info[f'PC{(n_pc+1):02d}_min'], pc_info[f'PC{(n_pc+1):02d}_max'], num = self.niche_rst_size + 1) for n_pc in range(len(self.env_list))
        }
        bin_info_list = {key: bin_info[key].tolist() for key in bin_info}

        # 切割完raster後的extent資訊
        extent_info = dict(
            **{
                f'PC{(n_pc+1):02d}_extent_max': ((bin_info[f'PC{(n_pc+1):02d}_bins'][1:] + bin_info[f'PC{(n_pc+1):02d}_bins'][:-1])/2)[-1].tolist() for n_pc in range(len(self.env_list))
            }, 
            **{
                f'PC{(n_pc+1):02d}_extent_min': ((bin_info[f'PC{(n_pc+1):02d}_bins'][1:] + bin_info[f'PC{(n_pc+1):02d}_bins'][:-1])/2)[0].tolist() for n_pc in range(len(self.env_list))
            }
        )

        with open(self.pc_info_path, 'w') as yaml_file:
            yaml.dump(pc_info, yaml_file)
        with open(self.bin_info_path, 'w') as yaml_file:
            yaml.dump(bin_info_list, yaml_file)
        with open(self.extent_info_path, 'w') as yaml_file:
            yaml.dump(extent_info, yaml_file)
            
        return pc_info, bin_info_list, extent_info
    
    # For Fig4
    def set_df_grid(self, df_env_pca, bin_info):
        
        df_grid = pd.DataFrame()
        for n_pc in range(len(self.env_list)):
            df_env_pca_pc = df_env_pca.filter(regex=f'^PC{(n_pc+1):02d}')

            def apply_cut(column):
                return pd.cut(column, 
                              bins = bin_info[f'PC{(n_pc+1):02d}_bins'], 
                              labels = False, 
                              include_lowest = True)

            grid = df_env_pca_pc.apply(apply_cut)
            grid.columns = [f'{col}_grid' for col in grid.columns]

            df_grid = pd.concat([df_grid, grid], axis = 1)
        
        return df_grid
        
    # For Fig4
    def create_species_nichespace(self):
        
        create_folder(os.path.dirname(self.plot_path_nichespace_h5))

        df_grid = feather.read_dataframe(self.df_grid_path)
        
        for species in self.species_list_predict:

            create_folder(os.path.dirname(self.plot_path_nichespace_png_sp.replace('[SPECIES]', species)))

            df_species = feather.read_dataframe(self.plot_path_df_species.replace('[SPECIES]', species))
            df_species = pd.concat([df_species, df_grid], axis = 1)

            # season
            season_sp_date = [(species, d) for d in self.date_list_predict]
            grid_deepsdm_all_month_sum = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_deepsdm_all_month_count = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_deepsdm_all_month_max = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_maxent_all_month_sum = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_maxent_all_month_count = np.zeros((self.niche_rst_size, self.niche_rst_size))
            grid_maxent_all_month_max = np.zeros((self.niche_rst_size, self.niche_rst_size))
            
            for (sp, d) in season_sp_date:
                # calculate grid values
                grouped = df_species.groupby([f'PC{self.x_pca:02d}_{d}_grid', f'PC{self.y_pca:02d}_{d}_grid'])

                # deepsdm all_month
                grid_deepsdm_all_month_sum_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_deepsdm_all_month_count_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_deepsdm_all_month_max_d = np.zeros((self.niche_rst_size, self.niche_rst_size))

                try:
                    max_values_deepsdm_all_month = grouped[f'deepsdm_all_month_{sp}_{d}'].max()
                    sum_values_deepsdm_all_month = grouped[f'deepsdm_all_month_{sp}_{d}'].sum()
                    mean_values_deepsdm_all_month = grouped[f'deepsdm_all_month_{sp}_{d}'].mean()
                    count_values_deepsdm_all_month = grouped[f'deepsdm_all_month_{sp}_{d}'].count()
                    for (x, y), value in max_values_deepsdm_all_month.items():
                        grid_deepsdm_all_month_max_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in count_values_deepsdm_all_month.items():
                        grid_deepsdm_all_month_count_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in sum_values_deepsdm_all_month.items():
                        grid_deepsdm_all_month_sum_d[self.niche_rst_size-1-y, x] = value                                                 
                except KeyError:
                    pass
                grid_deepsdm_all_month_max = np.nanmax([grid_deepsdm_all_month_max, grid_deepsdm_all_month_max_d], axis = 0)
                grid_deepsdm_all_month_count = grid_deepsdm_all_month_count + grid_deepsdm_all_month_count_d
                grid_deepsdm_all_month_sum = grid_deepsdm_all_month_sum + grid_deepsdm_all_month_sum_d

                # maxent all_month
                grid_maxent_all_month_sum_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_maxent_all_month_count_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                grid_maxent_all_month_max_d = np.zeros((self.niche_rst_size, self.niche_rst_size))
                try:
                    max_values_maxent_all_month = grouped[f'maxent_all_month_{sp}_{d}'].max()
                    sum_values_maxent_all_month = grouped[f'maxent_all_month_{sp}_{d}'].sum()
                    mean_values_maxent_all_month = grouped[f'maxent_all_month_{sp}_{d}'].mean()
                    count_values_maxent_all_month = grouped[f'maxent_all_month_{sp}_{d}'].count()
                    for (x, y), value in max_values_maxent_all_month.items():
                        grid_maxent_all_month_max_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in count_values_maxent_all_month.items():
                        grid_maxent_all_month_count_d[self.niche_rst_size-1-y, x] = value
                    for (x, y), value in sum_values_maxent_all_month.items():
                        grid_maxent_all_month_sum_d[self.niche_rst_size-1-y, x] = value                                                 
                except KeyError:
                    pass
                grid_maxent_all_month_max = np.nanmax([grid_maxent_all_month_max, grid_maxent_all_month_max_d], axis = 0)
                grid_maxent_all_month_count = grid_maxent_all_month_count + grid_maxent_all_month_count_d
                grid_maxent_all_month_sum = grid_maxent_all_month_sum + grid_maxent_all_month_sum_d


            grid_maxent_all_month_count = np.where(grid_maxent_all_month_count == 0, 1, grid_maxent_all_month_count)
            grid_deepsdm_all_month_count = np.where(grid_deepsdm_all_month_count == 0, 1, grid_deepsdm_all_month_count)

            grid_deepsdm_all_month_mean = grid_deepsdm_all_month_sum / grid_deepsdm_all_month_count
            grid_maxent_all_month_mean = grid_maxent_all_month_sum / grid_maxent_all_month_count

            
            logpng_info = {'deepsdm_all_month_max': grid_deepsdm_all_month_max, 
                           'deepsdm_all_month_mean': grid_deepsdm_all_month_mean, 
                           'deepsdm_all_month_sum': grid_deepsdm_all_month_sum, 
                           'maxent_all_month_max': grid_maxent_all_month_max, 
                           'maxent_all_month_mean': grid_maxent_all_month_mean,   
                           'maxent_all_month_sum': grid_maxent_all_month_sum}            
            for key, data in logpng_info.items():
                plot_output = self.plot_path_nichespace_png_sp.replace('[SPECIES]', species).replace('[SUFFIX]', key)
                cv2.imwrite(plot_output, png_operation(data))
            
            

            logh5_info = {'deepsdm_all_month_max': grid_deepsdm_all_month_max, 
                          'deepsdm_all_month_mean': grid_deepsdm_all_month_mean, 
                          'deepsdm_all_month_sum': grid_deepsdm_all_month_sum, 
                          'maxent_all_month_max': grid_maxent_all_month_max, 
                          'maxent_all_month_mean': grid_maxent_all_month_mean,   
                          'maxent_all_month_sum': grid_maxent_all_month_sum}
            for key, data in logh5_info.items():
                with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'a') as hf:
                    if key in hf:
                        del hf[key]
                    hf.create_dataset(key, data = data)
        
    # For Fig4
    def calculate_spearman(self, extent_info):
        extent = [extent_info[f'PC{self.x_pca:02d}_extent_min'], 
                  extent_info[f'PC{self.x_pca:02d}_extent_max'], 
                  extent_info[f'PC{self.y_pca:02d}_extent_min'], 
                  extent_info[f'PC{self.y_pca:02d}_extent_max']]

        # 計算每個像素的中心點坐標
        cell_width = (extent[1] - extent[0]) / self.niche_rst_size
        cell_height = (extent[3] - extent[2]) / self.niche_rst_size
        coordinates_values = {'center_x': [], 'center_y': []}
        for i in range(self.niche_rst_size):
            for j in range(self.niche_rst_size):
                coordinates_values['center_x'].append(extent[0] + j * cell_width + cell_width / 2)  # center_x
                coordinates_values['center_y'].append(extent[3] - i * cell_height - cell_height / 2) # center_y

        df_coords = pd.DataFrame(coordinates_values)
        df_spearman = pd.DataFrame({'model': [], 'species': [], 'rho': [], 'p': []})

        for species in self.species_list_predict:
            print(f'\r{species}', end = '')

            with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                grid_deepsdm_all_month_max = hf['deepsdm_all_month_max'][:]
                grid_deepsdm_all_month_mean = hf['deepsdm_all_month_mean'][:]
                grid_deepsdm_all_month_sum = hf['deepsdm_all_month_sum'][:]
                grid_maxent_all_month_max = hf['maxent_all_month_max'][:]
                grid_maxent_all_month_mean = hf['maxent_all_month_mean'][:]
                grid_maxent_all_month_sum = hf['maxent_all_month_sum'][:]

            # 初始化結果表
            centroids = []
            centroid_dict = {}

            # 處理每個 raster
            for raster_name, grid in {
                'deepsdm_max': grid_deepsdm_all_month_max,
                'deepsdm_mean': grid_deepsdm_all_month_mean,
                'deepsdm_sum': grid_deepsdm_all_month_sum,
                'maxent_max': grid_maxent_all_month_max,
                'maxent_mean': grid_maxent_all_month_mean,
                'maxent_sum': grid_maxent_all_month_sum,
            }.items():
                centroid_x, centroid_y = calculate_weighted_centroid(grid, extent)
                centroids.append({'raster': raster_name, 'centroid_x': centroid_x, 'centroid_y': centroid_y})
                centroid_dict[raster_name] = (centroid_x, centroid_y)


            # 遍歷每個 raster 計算 Mahalanobis 距離
            for raster_name, grid in {
                'deepsdm_max': grid_deepsdm_all_month_max,
                'deepsdm_mean': grid_deepsdm_all_month_mean,
                'deepsdm_sum': grid_deepsdm_all_month_sum,
                'maxent_max': grid_maxent_all_month_max,
                'maxent_mean': grid_maxent_all_month_mean,
                'maxent_sum': grid_maxent_all_month_sum,
            }.items():
                # 獲取對應 raster 的 centroid
                centroid_x, centroid_y = centroid_dict[raster_name]

                # 計算到該 centroid 的 Mahalanobis 距離
                df_cor = df_coords.copy()
                df_cor['value'] = grid.flatten()

                valid_df = df_cor[df_cor['value'] > 0].reset_index(drop = True)
                valid_coords = valid_df[['center_x', 'center_y']].values

                # 計算協方差矩陣和逆矩陣
                cov_matrix = np.cov(valid_coords, rowvar = False)
                inv_cov_matrix = np.linalg.inv(cov_matrix)

                # Mahalanobis 距離計算
                valid_df['distance_mah'] = valid_df[['center_x', 'center_y']].apply(
                    lambda row: mahalanobis(row, (centroid_x, centroid_y), inv_cov_matrix), axis = 1
                )

                # 計算 Spearman 相關係數
                rho, p = spearmanr(valid_df['distance_mah'], valid_df['value'])

                # 記錄到 DataFrame
                df_spearman.loc[len(df_spearman)] = [raster_name, species, rho, p]

            # 保存 Spearman 結果
            df_spearman.to_csv(self.df_spearman_path, index=False)
        
        return df_spearman
    
    # Fig4
    def calculate_rho_niche_coocccounts(self):
        
        # 预加载文件到内存
        def preload_nichespace_files(coocc_counts):
            nichespace_cache = {}
            unique_species = set(coocc_counts['sp1']).union(set(coocc_counts['sp2']))

            for species in unique_species:
                with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                    for model, suffix in [('deepsdm', 'max'), ('maxent', 'max')]:
                        rst = hf[f'{model}_all_month_{suffix}'][:]
                        nichespace_cache[(species, model)] = rst[rst > 0]
            return nichespace_cache
        
        def process_row(data, nichespace_cache):
            try:
                # 获取缓存中的栖息空间数据
                niche_space_deepsdm = [nichespace_cache[(sp, 'deepsdm')] for sp in [data.sp1, data.sp2]]
                niche_space_maxent = [nichespace_cache[(sp, 'maxent')] for sp in [data.sp1, data.sp2]]
                # 检查是否有数据缺失

                if any(ns is None for ns in niche_space_deepsdm) or any(ns is None for ns in niche_space_maxent):
                    return None

                # 计算 Cosine Similarity
                cosine_deepsdm = cosine_similarity_manual(niche_space_deepsdm[0], niche_space_deepsdm[1])
                cosine_maxent = cosine_similarity_manual(niche_space_maxent[0], niche_space_maxent[1])

                return data.counts, cosine_deepsdm, cosine_maxent
            except Exception as e:
                return None
        
        
        coocc_counts_filter = self.coocc_counts[(self.coocc_counts.sp1.isin(self.species_list_predict)) & 
                                                (self.coocc_counts.sp2.isin(self.species_list_predict)) & 
                                                (self.coocc_counts.sp1 != self.coocc_counts.sp2) & 
                                                (self.coocc_counts.counts != 0)].reset_index(drop = True)
        
        
        # 预加载栖息空间文件
        print("Preloading nichespace files...")
        nichespace_cache = preload_nichespace_files(coocc_counts_filter)
        #     print(nichespace_cache)
        # 并行处理每一行数据
        print("Processing co-occurrence data...")

        results = Parallel(n_jobs=-1)(
            delayed(process_row)(data, nichespace_cache) for _, data in coocc_counts_filter.iterrows()
        )
        # 收集结果
        valid_results = [r for r in results if r is not None]

        # 检查 valid_results 是否为空
        if not valid_results:
            print("No valid results were returned. Please check the input data or processing logic.")
        else:
            # 解包结果
            counts, deepsdm_cosine, maxent_cosine = zip(*valid_results)

            # 计算 Spearman 相关系数
            rho_cosine_deepsdm, p_cosine_deepsdm = spearmanr(np.array(deepsdm_cosine), np.array(counts))
            rho_cosine_maxent, p_cosine_maxent = spearmanr(np.array(maxent_cosine), np.array(counts))

            # 输出结果
            print(f'rho of deepsdm: {rho_cosine_deepsdm}, p-value: {p_cosine_deepsdm}')
            print(f'rho of maxent: {rho_cosine_maxent}, p-value: {p_cosine_maxent}')
            
            return rho_cosine_deepsdm, p_cosine_deepsdm, rho_cosine_maxent, p_cosine_maxent
        
    # For Fig4
    def merge_performance_indicator(self):
        
        # 設定檔案路徑並讀取數據
        files = glob.glob(self.performance_indicator_singlethreshold)
        indicator_singlethreshold = pd.concat([pd.read_csv(f) for f in files], ignore_index = True)

        files = glob.glob(self.performance_indicator_multithreshold)
        indicator_multithreshold = pd.concat([pd.read_csv(f) for f in files], ignore_index = True)

        indicator_multithreshold[['species', 'date']] = indicator_multithreshold['spdate'].str.rsplit('_', n = 1, expand = True)

        # 確保日期格式一致
        indicator_multithreshold['date'] = pd.to_datetime(indicator_multithreshold['date'])
        indicator_singlethreshold['date'] = pd.to_datetime(indicator_singlethreshold['date'])

        # 選擇要合併的欄位
        columns_to_merge = [
            'species', 'date', 
            'maxent_all_season_val', 'maxent_all_season_train', 'maxent_all_season_all',
            'deepsdm_all_season_val', 'deepsdm_all_season_train', 'deepsdm_all_season_all'
        ]
        indicator_multithreshold_subset = indicator_multithreshold[columns_to_merge]

        # 合併 DataFrame
        indicator_merged = indicator_singlethreshold.merge(indicator_multithreshold_subset, on=['species', 'date'], how='left')
        
        # 將指標重新命名，將 AUC 指標明確顯示為 AUC
        indicator_merged.rename(columns={
            'deepsdm_all_season_val': 'DeepSDM_AUC', 
            'maxent_all_season_val': 'MaxEnt_AUC', 
            'deepsdm_val_TSS': 'DeepSDM_TSS',
            'maxent_val_TSS': 'MaxEnt_TSS',
            'deepsdm_val_kappa': 'DeepSDM_Kappa',
            'maxent_val_kappa': 'MaxEnt_Kappa',
            'deepsdm_val_f1': 'DeepSDM_F1',
            'maxent_val_f1': 'MaxEnt_F1'
        }, inplace=True)
        
        return indicator_merged
        
    # For Fig5
    def get_deepsdm_nichespace(self, suffix = 'max'):
        nichespace_deepsdm_all = []
        nichespace_deepsdm_all_nonflatten = []
        for species in self.species_list_predict:
            h5_path = self.plot_path_nichespace_h5.replace('[SPECIES]', species)
            with h5py.File(h5_path, 'r') as hf:
                deepsdm_dataset_name = f'deepsdm_all_month_{suffix}'
                if deepsdm_dataset_name in hf.keys():
                    nichespace_deepsdm = hf[deepsdm_dataset_name][:].copy()
                    nichespace_deepsdm = nichespace_deepsdm/nichespace_deepsdm.sum().sum()
                    i_nonzeros = np.where(nichespace_deepsdm.flatten() > 0)
                    nichespace_deepsdm_all.append(nichespace_deepsdm.flatten()[i_nonzeros])
                    nichespace_deepsdm_all_nonflatten.append(nichespace_deepsdm)
                else:
                    print(f'No {sp} in DeepSDM.')

        nichespace_deepsdm_all = np.vstack(nichespace_deepsdm_all)
        nichespace_deepsdm_all_nonflatten = np.stack(nichespace_deepsdm_all_nonflatten, axis = 0)
        
        return nichespace_deepsdm_all, nichespace_deepsdm_all_nonflatten
        
    # For Fig5
    def get_all_cluster_average_nichespace(self, cluster_labels, nichespace_deepsdm_all_nonflatten):
        unique_clusters = sorted(set(cluster_labels))

        all_cluster_avg_nichespace = []
        for target_cluster in unique_clusters:
            species_in_cluster = [sp for sp, label in zip(self.species_list_predict, cluster_labels) if label == target_cluster]

            niche_sum = None
            valid_species_count = 0
            for sp in species_in_cluster:
                i_sp = self.species_list_predict.index(sp)
                niche_image = nichespace_deepsdm_all_nonflatten[i_sp]
                if niche_sum is None:
                    niche_sum = niche_image
                else:
                    niche_sum += niche_image
                valid_species_count += 1

            if valid_species_count > 0 and niche_sum is not None:
                niche_avg = niche_sum / valid_species_count
                all_cluster_avg_nichespace.append((target_cluster, niche_avg))
                print(f'Cluster {target_cluster} - Average niche calculated from {valid_species_count} species.')

            else:
                print(f'No valid niche images found for Cluster {target_cluster}')
                continue

        return all_cluster_avg_nichespace
        
        
    # For Fig5
    def calculate_deepsdm_nichespace_center(self, nichespace_deepsdm_all_nonflatten, cluster_labels):
        center_allspecies = []
        for i_sp in range(nichespace_deepsdm_all_nonflatten.shape[0]):

            nichespace_species = nichespace_deepsdm_all_nonflatten[i_sp]

            coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
            for i in range(self.niche_rst_size):
                for j in range(self.niche_rst_size):
                    coordinates_values['center_x'].append(self.nichespace_extent[0] + j * self.nichespace_cell_width + self.nichespace_cell_width / 2)  #center_x
                    coordinates_values['center_y'].append(self.nichespace_extent[3] - i * self.nichespace_cell_height - self.nichespace_cell_height / 2) # center_y
                    coordinates_values['value_deepsdm'].append(nichespace_species[i, j])  # value_deepsdm

            df_cor = pd.DataFrame(coordinates_values).query('value_deepsdm > 0').reset_index(drop = True)    
            df_cor_only = df_cor.loc[df_cor['value_deepsdm'] > 0, ['center_x', 'center_y']].reset_index(drop = True)

            center_x = np.multiply(np.array(df_cor['center_x']), np.array(df_cor['value_deepsdm'])).sum() / np.array(df_cor['value_deepsdm']).sum()
            center_y = np.multiply(np.array(df_cor['center_y']), np.array(df_cor['value_deepsdm'])).sum() / np.array(df_cor['value_deepsdm']).sum()
            center = np.array([center_x, center_y])
            center_allspecies.append(center)
        
        df_center = pd.DataFrame(np.vstack(center_allspecies), index = self.species_list_predict, columns = ['PC01', 'PC02'])
        df_center['cluster'] = cluster_labels
        
        return df_center

    # For Fig5
    def calculate_correlation_center_maxvariance(self, df_center):
        # 計算物種在 PC01 和 PC02 上的主成分變異量（協方差矩陣的特徵分解）
        cov_matrix = np.cov(df_center[['PC01', 'PC02']].T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 變異量最大的方向（對應於最大特徵值的特徵向量）
        max_variance_index = np.argmax(eigenvalues)
        max_variance_direction = eigenvectors[:, max_variance_index]

        # 計算環境因子 loading 在 PC01 和 PC02 的投影
        env_factors = self.env_pca_loadings[['PC01', 'PC02']].values
        env_correlation = env_factors @ max_variance_direction
        env_correlation_abs = np.abs(env_correlation)

        # 變異量最大的方向（對應於最大特徵值的特徵向量）
        max_variance_index = np.argmax(eigenvalues)
        max_variance_direction = eigenvectors[:, max_variance_index]

        # 計算最大變異方向的直線方程式 y = (v2/v1) * x
        center_maxvarinace_slope = max_variance_direction[1] / max_variance_direction[0]

        # 建立環境因子與最大變異方向相關性的 DataFrame
        env_correlation_df = pd.DataFrame({
            "Environmental Factor": self.env_pca_loadings.index,
            "Correlation with Max Variance Direction": env_correlation
        })

        # 依相關性絕對值排序（降序）
        env_correlation_df = env_correlation_df.reindex(env_correlation_abs.argsort()[::-1])
        return env_correlation_df, center_maxvarinace_slope
        
    # For Fig5
    def get_cluster_env_values(self, cluster_labels, env_plot):
        env_value_cluster_all = []
        for cluster in np.unique(cluster_labels):
            species_list_cluster = np.array(self.species_list_predict)[np.array(cluster_labels) == cluster].tolist()
            
            env_value_cluster = []
            for species in species_list_cluster:
                df_species = feather.read_dataframe(self.plot_path_df_species.replace('[SPECIES]', species))

                occ_value = df_species[sorted([key for key in df_species.keys() if key.startswith('occ')])].values.flatten()
                env_value = self.df_env_pca[sorted([key for key in self.df_env_pca.keys() if key.startswith(env_plot)])].values.flatten()

                mask = ~np.isnan(occ_value) & ~np.isnan(env_value)
                occ_value = occ_value[mask]
                env_value = env_value[mask]

                i_threshold = np.where(occ_value == 1)[0]

                env_value = env_value[i_threshold]
                env_value_cluster.append(env_value)

            env_value_cluster = np.concatenate(env_value_cluster)
            env_value_cluster_all.append((cluster, env_value_cluster))
        return env_value_cluster_all

    # For Fig6
    def calculate_cph_spearman_beta(self):
        # 物種參數字典初始化
        ecological_fit_info = {}   # 格式：{ species: [slope, intercept, min_distance, max_distance] }
        geographical_fit_info = {} # 格式：{ species: [slope, intercept, min_distance, max_distance] }
        df_spearman_ecogeo = pd.DataFrame({'center_type': [], 'species': [], 'rho': [], 'p': []})
        epsilon = 1e-8

        # 依序處理每一個物種
        for species in self.species_list_predict:
            print(f"Processing species: {species}\r", end = '')

            # -------------------------
            # 2. 讀取 h5 檔並累計所有時間點的結果（地理資料）
            img_sum = None
            with h5py.File(self.deepsdm_h5_path.replace('[SPECIES]', species), 'r') as hf:
                for date in self.date_list_predict:
                    if img_sum is None:
                        img_sum = hf[date][:].copy()
                    else:
                        img_sum = np.maximum(img_sum, hf[date][:].copy())

            # -------------------------
            # 3. 使用 rasterio 批量處理將所有像素索引轉換為經緯度
            x_indices = np.where(self.extent_binary == 1)[1]
            y_indices = np.where(self.extent_binary == 1)[0]
            lon_lat_pairs = [xy(self.transform, y, x) for y, x in zip(y_indices, x_indices)]
            lons, lats = zip(*lon_lat_pairs)
            lons = np.array(lons)
            lats = np.array(lats)

            # 計算加權後的地理中心（依據 img_sum 的值加權）
            valid_mask = ~np.isnan(img_sum)
            total = img_sum[valid_mask].sum()
            y_weighted = (np.where(valid_mask)[0] * img_sum[valid_mask]).sum() / total
            x_weighted = (np.where(valid_mask)[1] * img_sum[valid_mask]).sum() / total
            lon_center, lat_center = rasterio.transform.xy(self.transform, y_weighted, x_weighted, offset='center')

            # 計算每個像素與地理中心之間的距離
            distances_all = haversine(lats, lons, lat_center, lon_center)
            # 將 img_sum 平展後取非 NaN 值
            cell_values_all = img_sum[valid_mask].flatten()
            
            rho_geo, p_geo = spearmanr(distances_all, cell_values_all)
            df_spearman_ecogeo.loc[len(df_spearman_ecogeo)] = ['Geographical_center', species, rho_geo, p_geo]
            # -------------------------
            # 4. 計算生態中心（niche space）的資料
            # 透過預先計算的 nichespace_deepsdm 產生網格坐標
            with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                nichespace_deepsdm = hf['deepsdm_all_month_max'][:]

            coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
            for i in range(self.niche_rst_size):
                for j in range(self.niche_rst_size):
                    coordinates_values['center_x'].append(self.nichespace_extent[0] + j * self.nichespace_cell_width + self.nichespace_cell_width / 2)
                    coordinates_values['center_y'].append(self.nichespace_extent[3] - i * self.nichespace_cell_height - self.nichespace_cell_height / 2)
                    coordinates_values['value_deepsdm'].append(nichespace_deepsdm[i, j])
            df_cor = pd.DataFrame(coordinates_values)
            # 僅保留 value_deepsdm > 0 的資料
            df_cor = df_cor.query('value_deepsdm > 0').reset_index(drop=True)

            # 為計算 Mahalanobis 距離，我們先計算生態空間中非零值的中心與協方差矩陣
            df_cor_only = df_cor[['center_x', 'center_y']]
            cov_matrix = np.cov(df_cor_only, rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            # 加權計算生態空間中心（依據 value_deepsdm 的值）
            center_x = np.average(df_cor['center_x'], weights=df_cor['value_deepsdm'])
            center_y = np.average(df_cor['center_y'], weights=df_cor['value_deepsdm'])
            center = np.array([center_x, center_y])
            # 計算 Mahalanobis 距離，並存入新欄位
            df_cor['distance_mah'] = df_cor_only.apply(lambda row: mahalanobis(row, center, inv_cov_matrix), axis=1)

            # 定義條件（實際上 df_cor 的 value_deepsdm 已皆大於 0）
            condition_deepsdm = df_cor['value_deepsdm'] > 0
            
            rho_eco, p_eco = spearmanr(df_cor['distance_mah'][condition_deepsdm], df_cor['value_deepsdm'][condition_deepsdm])
            df_spearman_ecogeo.loc[len(df_spearman_ecogeo)] = ['Niche_center', species, rho_eco, p_eco]
            # -------------------------
            # **生態中心的 Beta 回歸**
            # 第 5 步：Beta regression (生態中心)
            x_eco = df_cor['distance_mah'].values
            y_eco = df_cor['value_deepsdm'].values

            # 正規化到 (0,1) 避免 0、1
            y_eco_normalize = (y_eco - np.min(y_eco)) / (np.max(y_eco) - np.min(y_eco)) \
                              * (1 - 2 * epsilon) + epsilon

            X_eco = sm.add_constant(x_eco)  # shape=(n,2)，第一欄截距
            model_eco = Beta(endog=y_eco_normalize,
                             exog=X_eco,
                             Z=None,                 # phi 沒有額外解釋變數
                             link=Logit(),           # \mu 的連結 (logit)
                             link_phi=sm.families.links.Log())  # phi 的連結 (log)
            result_eco = model_eco.fit(disp=False)

            # 假設 Z=None → 參數通常有 3 個: [intercept_mu, slope_mu, intercept_phi]
            # （如果 Beta 類別實作相同於先前提供的範例）
            params_eco = result_eco.params
            intercept_mu_eco = params_eco[0]   # \(\beta_0\)
            slope_mu_eco     = params_eco[1]   # \(\beta_1\)
            intercept_phi_eco = params_eco[2]  # phi 參數 (常數)

            # 另外紀錄 x, y 的 min/max，之後重畫時才知道如何逆規模化
            x_min_eco = float(np.min(x_eco))
            x_max_eco = float(np.max(x_eco))
            y_min_eco = float(np.min(y_eco))   # 原始 y 的 min
            y_max_eco = float(np.max(y_eco))   # 原始 y 的 max

            # 將本物種的參數存入 dictionary
            ecological_fit_info[species] = {
                "mu_link":    "logit",
                "phi_link":   "log",
                "params_mu":  [float(intercept_mu_eco), float(slope_mu_eco)],
                "params_phi": [float(intercept_phi_eco)],
                "x_min":      x_min_eco,
                "x_max":      x_max_eco,
                "y_min":      y_min_eco,
                "y_max":      y_max_eco,
                "epsilon":    epsilon  # 將用於日後反向還原
            }

            # -------------------------
            # 第 6 步：Beta regression (地理中心)
            x_geo = distances_all
            y_geo = cell_values_all

            y_geo_normalize = (y_geo - np.min(y_geo)) / (np.max(y_geo) - np.min(y_geo)) \
                              * (1 - 2 * epsilon) + epsilon

            X_geo = sm.add_constant(x_geo)
            model_geo = Beta(endog=y_geo_normalize,
                             exog=X_geo,
                             Z=None,
                             link=Logit(),
                             link_phi=sm.families.links.Log())
            result_geo = model_geo.fit(disp=False)

            params_geo = result_geo.params
            intercept_mu_geo = params_geo[0]
            slope_mu_geo     = params_geo[1]
            intercept_phi_geo = params_geo[2]

            x_min_geo = float(np.min(x_geo))
            x_max_geo = float(np.max(x_geo))
            y_min_geo = float(np.min(y_geo))
            y_max_geo = float(np.max(y_geo))

            geographical_fit_info[species] = {
                "mu_link":    "logit",
                "phi_link":   "log",
                "params_mu":  [float(intercept_mu_geo), float(slope_mu_geo)],
                "params_phi": [float(intercept_phi_geo)],
                "x_min":      x_min_geo,
                "x_max":      x_max_geo,
                "y_min":      y_min_geo,
                "y_max":      y_max_geo,
                "epsilon":    epsilon
            }
        return df_spearman_ecogeo, ecological_fit_info, geographical_fit_info
        
    # For Fig6
    def calculate_cph_spearman_beta_cluster(self, n_cluster=3):
        ecological_fit_info = {}   # { species: [slope, intercept, min_distance, max_distance] }
        geographical_fit_info = {} # { species: [slope, intercept, min_distance, max_distance] }
        epsilon = 1e-8

        for cluster in range(1, n_cluster+1):
            species_list_cluster = np.array(self.species_list_predict)[np.array(self.cluster_labels) == cluster]
            # 依序處理每一個物種
            df_cluster = []
            distances_all_cluster = []
            cell_values_all_cluster = []
            for species in species_list_cluster:
                print(f"Processing species: {species}\r", end = '')

                # -------------------------
                # 2. 讀取 h5 檔並累計所有時間點的結果（地理資料）
                img_sum = None
                with h5py.File(self.deepsdm_h5_path.replace('[SPECIES]', species), 'r') as hf:
                    for date in self.date_list_predict:
                        if img_sum is None:
                            img_sum = hf[date][:].copy()
                        else:
                            img_sum = np.maximum(img_sum, hf[date][:].copy())

                # -------------------------
                # 3. 使用 rasterio 批量處理將所有像素索引轉換為經緯度
                x_indices = np.where(self.extent_binary == 1)[1]
                y_indices = np.where(self.extent_binary == 1)[0]
                lon_lat_pairs = [xy(self.transform, y, x) for y, x in zip(y_indices, x_indices)]
                lons, lats = zip(*lon_lat_pairs)
                lons = np.array(lons)
                lats = np.array(lats)

                # 計算加權後的地理中心（依據 img_sum 的值加權）
                valid_mask = ~np.isnan(img_sum)
                total = img_sum[valid_mask].sum()
                y_weighted = (np.where(valid_mask)[0] * img_sum[valid_mask]).sum() / total
                x_weighted = (np.where(valid_mask)[1] * img_sum[valid_mask]).sum() / total
                lon_center, lat_center = rasterio.transform.xy(self.transform, y_weighted, x_weighted, offset='center')

                # 計算每個像素與地理中心之間的距離
                distances_all = haversine(lats, lons, lat_center, lon_center)
                # 將 img_sum 平展後取非 NaN 值
                cell_values_all = img_sum[valid_mask].flatten()

                distances_all_cluster.append(distances_all)
                cell_values_all_cluster.append(cell_values_all)


                # -------------------------
                # 4. 計算生態中心（niche space）的資料
                # 透過預先計算的 nichespace_deepsdm 產生網格坐標
                with h5py.File(self.plot_path_nichespace_h5.replace('[SPECIES]', species), 'r') as hf:
                    nichespace_deepsdm = hf['deepsdm_all_month_max'][:]

                coordinates_values = {'center_x': [], 'center_y': [], 'value_deepsdm': []}
                for i in range(self.niche_rst_size):
                    for j in range(self.niche_rst_size):
                        coordinates_values['center_x'].append(self.nichespace_extent[0] + j * self.nichespace_cell_width + self.nichespace_cell_width / 2)
                        coordinates_values['center_y'].append(self.nichespace_extent[3] - i * self.nichespace_cell_height - self.nichespace_cell_height / 2)
                        coordinates_values['value_deepsdm'].append(nichespace_deepsdm[i, j])
                df_cor = pd.DataFrame(coordinates_values)
                # 僅保留 value_deepsdm > 0 的資料
                df_cor = df_cor.query('value_deepsdm > 0').reset_index(drop=True)

                # 為計算 Mahalanobis 距離，我們先計算生態空間中非零值的中心與協方差矩陣
                df_cor_only = df_cor[['center_x', 'center_y']]
                cov_matrix = np.cov(df_cor_only, rowvar=False)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                # 加權計算生態空間中心（依據 value_deepsdm 的值）
                center_x = np.average(df_cor['center_x'], weights=df_cor['value_deepsdm'])
                center_y = np.average(df_cor['center_y'], weights=df_cor['value_deepsdm'])
                center = np.array([center_x, center_y])
                # 計算 Mahalanobis 距離，並存入新欄位
                df_cor['distance_mah'] = df_cor_only.apply(lambda row: mahalanobis(row, center, inv_cov_matrix), axis=1)

                df_cluster.append(df_cor)

            df_cluster_cor = pd.concat(df_cluster)
            distances_all_cluster = np.concatenate(distances_all_cluster)
            cell_values_all_cluster = np.concatenate(cell_values_all_cluster)

            # -------------------------
            # **生態中心的 Beta 回歸**
            # 第 5 步：Beta regression (生態中心)
            x_eco = df_cluster_cor['distance_mah'].values
            y_eco = df_cluster_cor['value_deepsdm'].values

            # 正規化到 (0,1) 避免 0、1
            y_eco_normalize = (y_eco - np.min(y_eco)) / (np.max(y_eco) - np.min(y_eco)) \
                              * (1 - 2 * epsilon) + epsilon

            X_eco = sm.add_constant(x_eco)  # shape=(n,2)，第一欄截距
            model_eco = Beta(endog=y_eco_normalize,
                             exog=X_eco,
                             Z=None,                 # phi 沒有額外解釋變數
                             link=Logit(),           # \mu 的連結 (logit)
                             link_phi=sm.families.links.Log())  # phi 的連結 (log)
            result_eco = model_eco.fit(disp=False)

            # 假設 Z=None → 參數通常有 3 個: [intercept_mu, slope_mu, intercept_phi]
            # （如果 Beta 類別實作相同於先前提供的範例）
            params_eco = result_eco.params
            intercept_mu_eco = params_eco[0]   # \(\beta_0\)
            slope_mu_eco     = params_eco[1]   # \(\beta_1\)
            intercept_phi_eco = params_eco[2]  # phi 參數 (常數)

            # 另外紀錄 x, y 的 min/max，之後重畫時才知道如何逆規模化
            x_min_eco = float(np.min(x_eco))
            x_max_eco = float(np.max(x_eco))
            y_min_eco = float(np.min(y_eco))   # 原始 y 的 min
            y_max_eco = float(np.max(y_eco))   # 原始 y 的 max

            # 將本物種的參數存入 dictionary
            ecological_fit_info[cluster] = {
                "mu_link":    "logit",
                "phi_link":   "log",
                "params_mu":  [float(intercept_mu_eco), float(slope_mu_eco)],
                "params_phi": [float(intercept_phi_eco)],
                "x_min":      x_min_eco,
                "x_max":      x_max_eco,
                "y_min":      y_min_eco,
                "y_max":      y_max_eco,
                "epsilon":    epsilon  # 將用於日後反向還原
            }

            with open(self.beta_cluster_result_path.replace('[CENTER_TYPE]', 'Niche_center').replace('[CLUSTER]', str(cluster)), 'w') as file:
                file.write(result_eco.summary().as_text())

            # -------------------------
            # 第 6 步：Beta regression (地理中心)
            x_geo = distances_all_cluster
            y_geo = cell_values_all_cluster

            y_geo_normalize = (y_geo - np.min(y_geo)) / (np.max(y_geo) - np.min(y_geo)) \
                              * (1 - 2 * epsilon) + epsilon

            X_geo = sm.add_constant(x_geo)
            model_geo = Beta(endog=y_geo_normalize,
                             exog=X_geo,
                             Z=None,
                             link=Logit(),
                             link_phi=sm.families.links.Log())
            result_geo = model_geo.fit(disp=False)

            params_geo = result_geo.params
            intercept_mu_geo = params_geo[0]
            slope_mu_geo     = params_geo[1]
            intercept_phi_geo = params_geo[2]

            x_min_geo = float(np.min(x_geo))
            x_max_geo = float(np.max(x_geo))
            y_min_geo = float(np.min(y_geo))
            y_max_geo = float(np.max(y_geo))

            geographical_fit_info[cluster] = {
                "mu_link":    "logit",
                "phi_link":   "log",
                "params_mu":  [float(intercept_mu_geo), float(slope_mu_geo)],
                "params_phi": [float(intercept_phi_geo)],
                "x_min":      x_min_geo,
                "x_max":      x_max_geo,
                "y_min":      y_min_geo,
                "y_max":      y_max_geo,
                "epsilon":    epsilon
            }

            with open(self.beta_cluster_result_path.replace('[CENTER_TYPE]', 'Geographical_center').replace('[CLUSTER]', str(cluster)), 'w') as file:
                file.write(result_geo.summary().as_text())
                
        return ecological_fit_info, geographical_fit_info
        
        
        
        
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def load_existing_files(self):
        self.avg_elev = None
        self.df_attention = None
        self.df_env_corr = None
        self.df_env_pca = None
        self.pc_info = None
        self.bin_info = None
        self.extent_info = None
        self.df_grid = None
        self.df_spearman = None
        self.cluster_avg_nichespace = None
        self.cluster_labels = None
        self.df_spearman_ecogeo = None
        
        # 逐一檢查檔案是否存在並讀取
        if os.path.exists(self.avg_elev_path):
            self.avg_elev = pd.read_csv(self.avg_elev_path)

        if os.path.exists(self.df_attention_path):
            self.df_attention = pd.read_csv(self.df_attention_path, index_col=0)

        if os.path.exists(self.df_env_pca_path):
            self.df_env_pca = feather.read_dataframe(self.df_env_pca_path)

        if os.path.exists(self.pc_info_path):
            with open(self.pc_info_path, 'r') as f:
                self.pc_info = yaml.safe_load(f)

        if os.path.exists(self.bin_info_path):
            with open(self.bin_info_path, 'r') as f:
                self.bin_info = yaml.safe_load(f)

        if os.path.exists(self.extent_info_path):
            with open(self.extent_info_path, 'r') as f:
                self.extent_info = yaml.safe_load(f)
            self.nichespace_extent = [self.extent_info[f'PC{self.x_pca:02d}_extent_min'], 
                                      self.extent_info[f'PC{self.x_pca:02d}_extent_max'], 
                                      self.extent_info[f'PC{self.y_pca:02d}_extent_min'], 
                                      self.extent_info[f'PC{self.y_pca:02d}_extent_max']]
            self.nichespace_cell_width = (self.nichespace_extent[1] - self.nichespace_extent[0]) / self.niche_rst_size
            self.nichespace_cell_height = (self.nichespace_extent[3] - self.nichespace_extent[2]) / self.niche_rst_size

        if os.path.exists(self.df_spearman_path):
            self.df_spearman = pd.read_csv(self.df_spearman_path)

        if os.path.exists(self.cluster_avg_nichespace_path):
            with open(self.cluster_avg_nichespace_path, 'r') as f:
                self.cluster_avg_nichespace = yaml.safe_load(f)        
        
        if os.path.exists(self.env_pca_loadings_path):
            self.env_pca_loadings = pd.read_csv(self.env_pca_loadings_path, index_col = 0)
        
        if os.path.exists(self.cluster_labels_path):
            with open(self.cluster_labels_path, 'r') as f:
                self.cluster_labels = yaml.load(f, Loader = yaml.FullLoader)
        
        if os.path.exists(self.df_spearman_ecogeo_path):
            self.df_spearman_ecogeo = pd.read_csv(self.df_spearman_ecogeo_path)
                    
        if os.path.exists(self.niche_beta_params_path):
            with open(self.niche_beta_params_path, 'r') as f:
                self.niche_beta_params = json.load(f)
                
        if os.path.exists(self.geographical_beta_params_path):
            with open(self.geographical_beta_params_path, 'r') as f:
                self.geographical_beta_params = json.load(f)
        
        if os.path.exists(self.niche_beta_cluster_params_path):
            with open(self.niche_beta_cluster_params_path, 'r') as f:
                self.niche_beta_cluster_params = json.load(f)
                
        if os.path.exists(self.geographical_beta_cluster_params_path):
            with open(self.geographical_beta_cluster_params_path, 'r') as f:
                self.geographical_beta_cluster_params = json.load(f)

        
def create_folder(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
def cov_center(data, level=0.95):
    env = EllipticEnvelope(support_fraction=level).fit(data)
    center = env.location_
    covariance = env.covariance_
    return center, covariance

def plot_ellipse(center, covariance, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    在给定的轴上绘制一个椭圆。

    :param center: 椭圆的中心点。
    :param covariance: 椭圆的协方差矩阵。
    :param ax: matplotlib 轴对象。
    :param n_std: 确定椭圆大小的标准差倍数。
    :param facecolor: 椭圆的填充颜色。
    :param kwargs: 传递给 Ellipse 对象的其他参数。
    """
    # 计算协方差矩阵的特征值和特征向量
    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    # 计算椭圆的宽度和高度
    width, height = 2 * n_std * np.sqrt(eigenvals)
    angle = np.degrees(np.arctan2(*eigenvecs[:,0][::-1]))

    # 创建并添加椭圆形状
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)
    
def png_operation(rst):
    grid_normalized = cv2.normalize(rst, None, 0, 255, cv2.NORM_MINMAX)
    grid_uint8 = grid_normalized.astype(np.uint8)
    colored_image = cv2.applyColorMap(grid_uint8, cv2.COLORMAP_JET)
    # 設定要調整的尺寸，例如 300x300
    new_size = (300, 300)
    # 調整 PNG 的大小
    resized_png = cv2.resize(colored_image, new_size, interpolation=cv2.INTER_CUBIC)
    return resized_png

# 計算兩個經緯度點之間的距離，支持向量化操作
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # 地球半徑，單位：公里
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def mm2inch(*values):
    return [v / 25.4 for v in values]

def set_mpl_defaults():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 5
    mpl.rcParams['axes.labelsize'] = 7
    mpl.rcParams['xtick.labelsize'] = 5
    mpl.rcParams['ytick.labelsize'] = 5
    mpl.rcParams['legend.fontsize'] = 5
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['xtick.major.size'] = 2.5
    mpl.rcParams['ytick.major.size'] = 2.5
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['boxplot.boxprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.medianprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.capprops.linewidth'] = 0.5
    mpl.rcParams['boxplot.flierprops.markersize'] = 0.5
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['boxplot.medianprops.color'] = 'black'
    mpl.rcParams['hatch.linewidth'] = 0.5 

# 顯著性標示函式
def get_significance_stars(p_value):
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'n.s.'
    
def convert_to_env_list_detail(env_list_original):
    env_list_change = {
        'clt': 'Cloud area fraction',
        'hurs': 'Relative humidity',
        'pr': 'Precipitation',
        'rsds': 'Shortwave radiation',
        'sfcWind': 'Wind speed',
        'tas': 'Temperature',
        'EVI': 'EVI',
        'landcover_PC00': 'LandcoverPC1',
        'landcover_PC01': 'LandcoverPC2',
        'landcover_PC02': 'LandcoverPC3',
        'landcover_PC03': 'LandcoverPC4',
        'landcover_PC04': 'LandcoverPC5', 
    }
    return [env_list_change[i] for i in env_list_original]

# Fig3
def reorder_df_attention(df_attention, env_order = 'LandcoverPC1'):
    df_attention_order = df_attention.copy()
    df_attention_order.columns = convert_to_env_list_detail(df_attention_order.columns)
    
    df_attention_order = df_attention_order.sort_values(by = env_order, ascending=True)

    factor_order = df_attention_order.mean().sort_values(ascending=True).index
    df_attention_order = df_attention_order[factor_order]

    df_attention_order.index = [f"{sp.split('_')[0]} {sp.split('_')[1]}" for sp in list(df_attention_order.index)]
    
    return df_attention_order

# Fig4
def calculate_weighted_centroid(grid, extent):
    """
    計算 raster 的加權中心
    :param grid: numpy array, raster 數值 (2D)
    :param extent: [xmin, xmax, ymin, ymax], 地理範圍
    :return: (centroid_x, centroid_y)
    """
    # 創建格點的地理座標
    rows, cols = grid.shape
    x_coords = np.linspace(extent[0], extent[1], cols)
    y_coords = np.linspace(extent[3], extent[2], rows)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # 展平數據，方便計算
    values = grid.flatten()
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # 過濾有效值（例如忽略值為零的像素）
    valid_mask = values > 0  # 僅考慮數值大於 0 的像素
    values = values[valid_mask]
    x_flat = x_flat[valid_mask]
    y_flat = y_flat[valid_mask]

    # 計算加權中心
    weighted_x = np.sum(values * x_flat) / np.sum(values)
    weighted_y = np.sum(values * y_flat) / np.sum(values)

    return weighted_x, weighted_y


def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product != 0 else 0

# Fig4
def get_performance_stats(indicator_merged):
    
    # 定義指標名稱和相應的欄位
    indicators = {
        'AUC': ('DeepSDM_AUC', 'MaxEnt_AUC'), 
        'F1': ('DeepSDM_F1', 'MaxEnt_F1'),
        'TSS': ('DeepSDM_TSS', 'MaxEnt_TSS'),
        "Cohen's Kappa": ('DeepSDM_Kappa', 'MaxEnt_Kappa')
    }
    
    # 準備數據
    data_deepsdm = []
    data_maxent = []
    ticks = []
    significance_stars = []
    n = []
    t_stats = []
    p_values = []

    for indicator_name, (col_deepsdm, col_maxent) in indicators.items():
        # 篩選有效數據
        indicator_filtered = indicator_merged[(indicator_merged[col_deepsdm] != -9999) & (indicator_merged[col_deepsdm].notna()) &
                                              (indicator_merged[col_maxent] != -9999) & (indicator_merged[col_maxent].notna())]
        n.append(len(indicator_filtered))

        # 添加數據至列表
        data_deepsdm.append(indicator_filtered[col_deepsdm].values)
        data_maxent.append(indicator_filtered[col_maxent].values)
        ticks.append(indicator_name)

        # 計算 t 檢定並獲取顯著性標示
        t_stat, p_value = ttest_rel(indicator_filtered[col_deepsdm], indicator_filtered[col_maxent])
        significance_stars.append(get_significance_stars(p_value))
        t_stats.append(t_stat)
        p_values.append(p_value)
        
    return data_deepsdm, data_maxent, ticks, significance_stars, n, t_stats, p_values


























# -*- coding: utf-8 -*-

u"""
Beta regression for modeling rates and proportions.
References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.
Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln as lgamma
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod.families import Binomial

# this is only need while #2024 is open.
class Logit(sm.families.links.Logit):

    """Logit tranform that won't overflow with large numbers."""

    def inverse(self, z):
        return 1 / (1. + np.exp(-z))

_init_example = """
    Beta regression with default of logit-link for exog and log-link
    for precision.
    >>> mod = Beta(endog, exog)
    >>> rslt = mod.fit()
    >>> print rslt.summary()
    We can also specify a formula and a specific structure and use the
    identity-link for phi.
    >>> from sm.families.links import identity
    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')
    >>> mod = Beta.from_formula('iyield ~ C(batch, Treatment(10)) + temp',
    ...                         dat, Z=Z, link_phi=identity())
    In the case of proportion-data, we may think that the precision depends on
    the number of measurements. E.g for sequence data, on the number of
    sequence reads covering a site:
    >>> Z = patsy.dmatrix('~ coverage', df)
    >>> mod = Beta.from_formula('methylation ~ disease + age + gender + coverage', df, Z)
    >>> rslt = mod.fit()
"""

class Beta(GenericLikelihoodModel):

    """Beta Regression.
    This implementation uses `phi` as a precision parameter equal to
    `a + b` from the Beta parameters.
    """

    def __init__(self, endog, exog, Z=None, link=Logit(),
            link_phi=sm.families.links.Log(), **kwds):
        """
        Parameters
        ----------
        endog : array-like
            1d array of endogenous values (i.e. responses, outcomes,
            dependent variables, or 'Y' values).
        exog : array-like
            2d array of exogeneous values (i.e. covariates, predictors,
            independent variables, regressors, or 'X' values). A nobs x k
            array where `nobs` is the number of observations and `k` is
            the number of regressors. An intercept is not included by
            default and should be added by the user. See
            `statsmodels.tools.add_constant`.
        Z : array-like
            2d array of variables for the precision phi.
        link : link
            Any link in sm.families.links for `exog`
        link_phi : link
            Any link in sm.families.links for `Z`
        Examples
        --------
        {example}
        See Also
        --------
        :ref:`links`
        """.format(example=_init_example)
        assert np.all((0 < endog) & (endog < 1))
        if Z is None:
            extra_names = ['phi']
            Z = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in \
                        (Z.columns if hasattr(Z, 'columns') else range(1, Z.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names

        super(Beta, self).__init__(endog, exog, **kwds)
        self.link = link
        self.link_phi = link_phi
        
        self.Z = Z
        assert len(self.Z) == len(self.endog)

    def nloglikeobs(self, params):
        """
        Negative log-likelihood.
        Parameters
        ----------
        params : np.ndarray
            Parameter estimates
        """
        return -self._ll_br(self.endog, self.exog, self.Z, params)

    def fit(self, start_params=None, maxiter=100000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model.
        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        """

        if start_params is None:
            start_params = sm.GLM(self.endog, self.exog, family=Binomial()
                                 ).fit(disp=False).params
            start_params = np.append(start_params, [0.5] * self.Z.shape[1])

        return super(Beta, self).fit(start_params=start_params,
                                        maxiter=maxiter,
                                        method=method, disp=disp, **kwds)

    def _ll_br(self, y, X, Z, params):
        nz = self.Z.shape[1]

        Xparams = params[:-nz]
        Zparams = params[-nz:]

        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_phi.inverse(np.dot(Z, Zparams))
        # TODO: derive a and b and constrain to > 0?

        if np.any(phi <= np.finfo(float).eps): return np.array(-np.inf)

        ll = lgamma(phi) - lgamma(mu * phi) - lgamma((1 - mu) * phi) \
                + (mu * phi - 1) * np.log(y) + (((1 - mu) * phi) - 1) \
                * np.log(1 - y)

        return ll

def logistic(z):
    return 1 / (1 + np.exp(-z))
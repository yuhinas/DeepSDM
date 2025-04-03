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
        
        
        
        
        # 子資料夾路徑
        self.plot_path_embedding_dimension_reduction = os.path.join('plots', run_id, 'Fig2_embedding_dimension_reduction')
        self.plot_path_embedding_correlation = os.path.join('plots', run_id, 'Fig2_embedding_correlation')        
        self.plot_path_attention = os.path.join('plots', run_id, 'Fig3_attention')
        self.plot_path_attentionstats = os.path.join('plots', run_id, 'FigS2_attentionstats')
        self.plot_path_nichespace = os.path.join('plots', run_id, 'Fig4_nichespace')
        self.plot_path_envcorrelation = os.path.join('plots', run_id, 'FigS3_envcorrelation')
        self.plot_path_nichespace_clustering = os.path.join('plots', run_id, 'Fig5_nichespace_clustering')
        self.plot_path_nichespace_clustering_test = os.path.join('plots', run_id, 'FigS4_nichespace_clustering_test')
        
        
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
        self.cluster_avg_nichespace_path = os.path.join(self.plot_path_nichespace_clustering, 'cluster_avg_nichespace.yaml')
        self.df_nichespace_center_coordinate_path = os.path.join(self.plot_path_nichespace_clustering, 'nichespecies_center_coordinate.csv')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
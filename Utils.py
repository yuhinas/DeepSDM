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

class PlotUtlis():
    def __init__(self, run_id, exp_id):
        
        # 路徑
        self.conf_path = os.path.join('mlruns', exp_id, run_id, 'artifacts', 'conf')
        self.predicts_path = os.path.join('predicts', run_id)
        
        self.deepsdm_h5_path = os.path.join('predicts', run_id, 'h5', '[SPECIES]', '[SPECIES].h5')
        self.maxent_h5_path = os.path.join('predict_maxent', run_id, 'h5', 'all', '[SPECIES]', '[SPECIES].h5')  
        
        self.attention_h5_path = os.path.join('predicts', run_id, 'attention', '[SPECIES]', '[SPECIES]_[DATE]_attention.h5')
        
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
            self.sp_inf = json.load(f)
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
        
        
        
        # 子資料夾路徑
        self.plot_path_embedding_dimension_reduction = os.path.join('plots', run_id, 'Fig2_embedding_dimension_reduction')
        self.plot_path_embedding_correlation = os.path.join('plots', run_id, 'Fig2_embedding_correlation')        
        self.plot_path_attention = os.path.join('plots', run_id, 'Fig3_attention')
        self.plot_path_attentionstats = os.path.join('plots', run_id, 'FigS2_attentionstats')
        
        
        
        # 輸出路徑
        # output path
        self.avg_elev_path = os.path.join(self.plot_path_embedding_dimension_reduction, 'avg_elevation.csv')
        self.df_attention_path = os.path.join(self.plot_path_attention, 'df_attention.csv')
        
        
        
        
        
        
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
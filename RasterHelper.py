import numpy as np
import os
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import rasterio
from osgeo import gdal
from osgeo import gdalconst
import cv2
from matplotlib import pyplot as plt

class RasterHelper:
    def __init__(self):
        super().__init__()
        
        self.species_filter = None
        self.species_list = None

        if not os.path.exists('./workspace'):
            os.makedirs('./workspace')
        
        
    def res_rounder(self, a, second_round=None):
        if second_round is None:
            return round(a, self.num_digits_after_decimal)
        else:
            return round(round(a, self.num_digits_after_decimal), second_round)

    def time_span(self, sourcedate):
        delta = relativedelta(years = self.year_span, months = self.month_span, days = self.day_span)
        return sourcedate + delta

    def time_step(self, sourcedate):
        delta = relativedelta(years = self.year_step, months = self.month_step, days = self.day_step)
        return sourcedate + delta
        
    def set_temporal_conf(self, temporal_conf):
        self.date_start = datetime.strptime(temporal_conf.date_start, '%Y-%m-%d')
        self.date_end = datetime.strptime(temporal_conf.date_end, '%Y-%m-%d')
        self.year_span = 0
        self.month_span = temporal_conf.month_span
        self.day_span = 0
        self.year_step = 0
        self.month_step = temporal_conf.month_step
        self.day_step = 0
        self.day_first = 0
        self.day_last = (self.date_end - self.date_start).days

    def set_env_conf(self, env_conf):
        self.env_conf = env_conf
        
    def create_extent_binary_from_env_layer(self, input_env, spatial_conf):
        
        self.spatial_conf = spatial_conf
        
        in_ds = gdal.Open(input_env, gdalconst.GA_ReadOnly)
        _, xres, _, _, _, yres = in_ds.GetGeoTransform()
        in_proj = in_ds.GetProjection()

        # TODO
        # Need to clarify what value it would be with tif of southern hemisphere
        
        x_start = self.spatial_conf.x_start
        y_start = self.spatial_conf.y_start
#         x_end = self.spatial_conf.x_end
#         y_end = self.spatial_conf.y_end
        
        if self.spatial_conf.out_res is not None:
            self.spatial_conf.out_res = abs(self.spatial_conf.out_res)
        else:
            assert(abs(xres) == abs(yres))            
            self.spatial_conf.out_res = abs(xres)
            
        digit_parts = str(self.spatial_conf.out_res).split('.')
        assert(len(digit_parts) == 2)
        self.num_digits_after_decimal = min(len(digit_parts[1]), 6)
        
#         self.spatial_conf.x_start = self.res_rounder(self.res_rounder(x_start / self.spatial_conf.out_res, 0) * self.spatial_conf.out_res)
#         self.spatial_conf.y_start = self.res_rounder(self.res_rounder(y_start / self.spatial_conf.out_res, 0) * self.spatial_conf.out_res)

#         self.spatial_conf.x_num_cells = int(np.ceil(self.res_rounder(abs(x_end - self.spatial_conf.x_start) / self.spatial_conf.out_res)))
#         self.spatial_conf.y_num_cells = int(np.ceil(self.res_rounder(abs(y_end - self.spatial_conf.y_start) / self.spatial_conf.out_res)))

        self.spatial_conf.x_num_cells = int(self.spatial_conf.num_of_grid_x * self.spatial_conf.grid_size)
        self.spatial_conf.y_num_cells = int(self.spatial_conf.num_of_grid_y * self.spatial_conf.grid_size)

        self.spatial_conf.x_end = self.res_rounder(self.spatial_conf.x_start + self.spatial_conf.x_num_cells * self.spatial_conf.out_res)
        self.spatial_conf.y_end = self.res_rounder(self.spatial_conf.y_start + self.spatial_conf.y_num_cells * self.spatial_conf.out_res)
        out_trans = (self.spatial_conf.x_start, self.spatial_conf.out_res, 0, self.spatial_conf.y_end, 0, -self.spatial_conf.out_res)
        
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create('workspace/extent_env_example.tif', self.spatial_conf.x_num_cells, self.spatial_conf.y_num_cells, 1, gdalconst.GDT_Float32)

        # 
        output.SetGeoTransform(out_trans)
        output.SetProjection(in_proj)

        #
        out_proj = in_proj
        gdal.ReprojectImage(in_ds, output, in_proj, out_proj, gdalconst.GRA_Bilinear)

        in_ds = None
        driver  = None
        output = None        
        
        #############################################################
        
        in_ds  = gdal.Open('workspace/extent_env_example.tif', gdalconst.GA_ReadOnly)
        in_trans = in_ds.GetGeoTransform()
        in_proj = in_ds.GetProjection()

        extent_array = in_ds.ReadAsArray()
        extent_array = np.where(extent_array == 0, 0, 1)

        driver= gdal.GetDriverByName('GTiff')
        output = driver.Create('workspace/extent_binary.tif', self.spatial_conf.x_num_cells, self.spatial_conf.y_num_cells, 1, gdalconst.GDT_Int16)
        output.SetGeoTransform(in_trans)
        output.SetProjection(in_proj)
        export = output.GetRasterBand(1).WriteArray(extent_array)

        in_ds = None
        driver = None
        output = None
        export = None
        
        return self.spatial_conf

    def random_split_train_val(self, train_ratio=0.7):
        spatial_conf = self.spatial_conf
        total_grids = spatial_conf.num_of_grid_y * spatial_conf.num_of_grid_x
        num_train_grids = int(np.round(total_grids * train_ratio))
        
        train_val_partitions = np.where(np.random.uniform(size=(spatial_conf.num_of_grid_y, spatial_conf.num_of_grid_x)) >= train_ratio, 0, 1)
        
        while train_val_partitions.sum() != num_train_grids:
            train_val_partitions = np.where(np.random.uniform(size=(4,3))>0.7, 0, 1).astype(np.uint8)
            
        np.savetxt('./workspace/partition.txt', train_val_partitions, fmt='%i', delimiter=',')
        print('Partition saved at ./workspace/partition.txt.')
        
    def view_train_val_splits(self, partition_file='./workspace/partition.txt'):
        spatial_conf = self.spatial_conf
        ext_bin  = gdal.Open('./workspace/extent_binary.tif', gdalconst.GA_ReadOnly)
        ext_bin_array = ext_bin.ReadAsArray()
        train_val_partitions = np.loadtxt(partition_file, delimiter=',')
        train_val_mask = cv2.resize(train_val_partitions, (spatial_conf.num_of_grid_x * spatial_conf.grid_size, spatial_conf.num_of_grid_y * spatial_conf.grid_size), interpolation=cv2.INTER_NEAREST)

        plt.imshow(ext_bin_array * (train_val_mask.astype(float) + .5) / 2)
        plt.show()

        
    def create_species_raster(self):

        if self.species_filter is None:
            self.species_filter = pd.read_csv('./workspace/species_data/occurrence_data/species_occurrence_filter.csv')
            self.species_list = np.unique(self.species_filter.species)
            
        species_filter = self.species_filter
        species_list = self.species_list
        
        self.sp_raster_out = './workspace/raster_data/species_occurrence'
        if not os.path.exists(self.sp_raster_out):
            os.makedirs(self.sp_raster_out)
        
        with rasterio.open('./workspace/extent_binary.tif') as raster_:
            extent_crs = raster_.crs
            extent_binary = raster_.read(1)
            extent_transform = raster_.transform
            
        xres = abs(extent_transform[0])
        yres = abs(extent_transform[4])

        sp_inf = dict()
        sp_inf['dir_base'] = self.sp_raster_out
        file_name = dict()
        
        for sp in species_list:

            # create folder
            if not os.path.exists(os.path.join(self.sp_raster_out, f'{sp}')):
                os.makedirs(os.path.join(self.sp_raster_out, f'{sp}'))

            # species information json
            file_name[sp] = dict()

            # filter data by species
            data_s = species_filter[species_filter['species'].values == sp]

            # date operation
            date_s = self.date_start
            t_s = self.day_first
            date_e = self.time_span(date_s)
            t_e = (date_e - date_s).days
            
            while t_s <= self.day_last:

                data_t = data_s[(data_s['daysincebegin'].values < t_e) & (data_s['daysincebegin'].values >= t_s)]
                rst = np.zeros([extent_binary.shape[0], extent_binary.shape[1]])
                for i, row in data_t.iterrows():
                    nlong = self.res_rounder(abs(row['decimalLongitude'] - self.spatial_conf.x_start) / xres, 0)
                    nlat = self.res_rounder(abs(self.spatial_conf.y_end - row['decimalLatitude']) / yres, 0)
                    rst[int(nlat), int(nlong)] = 1

                date_span = f"{date_s.strftime('%Y')}_{date_s.strftime('%m')}_{date_s.strftime('%d')}"
                sp_data_span_tif = f"{sp}/{sp}_{date_span}.tif"

                with rasterio.open(
                    f"{self.sp_raster_out}/{sp_data_span_tif}", 
                    'w', 
                    height = extent_binary.shape[0], 
                    width = extent_binary.shape[1],
                    count = 1, 
                    nodata = -9, 
                    crs = extent_crs, 
                    dtype = rasterio.int16, 
                    transform = extent_transform
                ) as dst:
                    dst.write(rst * extent_binary, 1)

                file_name[sp][date_span] = sp_data_span_tif

                date_s = self.time_step(date_s)
                t_s = (date_s - self.date_start).days
                date_e = self.time_span(date_s)
                t_e = (date_e - self.date_start).days

        sp_inf['file_name'] = file_name
        with open('./workspace/species_information.json', 'w') as f:
            json.dump(sp_inf, f)
            
        self.sp_inf = sp_inf

    def align_env(self, dir_input='./raw/env'):

        self.env_aligned_out = './workspace/raster_data/env_aligned'
        if not os.path.exists(self.env_aligned_out):
            os.makedirs(self.env_aligned_out)
        
        extent_binary = gdal.Open('./workspace/extent_binary.tif')
        extent_transform = extent_binary.GetGeoTransform()
        extent_projection = extent_binary.GetProjection()

        for env in self.env_conf.env_list:
            env_raster_dir = os.path.join(self.env_aligned_out, env)
            if not os.path.exists(env_raster_dir):
                os.makedirs(env_raster_dir)
            file_names = os.listdir(os.path.join(dir_input, env))
            for file_name in file_names:
                if not file_name.endswith(f'.{self.env_conf.ext}'):
                    print (f'File name does not end with .{self.env_conf.ext}, ignoring file {file_name}.')
                    continue
                    
                print(file_name, end='\r')
                    
                ds  = gdal.Open(os.path.join(dir_input, env, file_name), gdalconst.GA_ReadOnly)
                in_trans = ds.GetGeoTransform()
                in_proj = ds.GetProjection()

                driver= gdal.GetDriverByName('GTiff')
                output = driver.Create(os.path.join(env_raster_dir, file_name), 
                                       extent_binary.GetRasterBand(1).XSize, 
                                       extent_binary.GetRasterBand(1).YSize, 
                                       1, 
                                       gdalconst.GDT_Float32)

                # 设置输出文件地理仿射变换参数与投影
                output.SetGeoTransform(extent_transform)
                output.SetProjection(extent_projection)

                # 重投影，插值方法为双线性内插法
                gdal.ReprojectImage(ds, output, in_proj, extent_projection, gdalconst.GRA_Bilinear)

                ds = None
                driver  = None
                output = None
                
        extent_binary = None

    def avg_timespan_env(self):

        self.env_aligned_timespan_avg_out = './workspace/raster_data/env_aligned_timespan_avg'
        if not os.path.exists(self.env_aligned_timespan_avg_out):
            os.makedirs(self.env_aligned_timespan_avg_out)
        
        env_filename_templates = self.env_conf.env_filename_templates
        
        with rasterio.open('./workspace/extent_binary.tif') as raster_:
            extent_transform = raster_.transform
            extent_binary = raster_.read(1)
            extent_crs = raster_.crs

        env_info = dict()
        env_info['dir_base'] = self.env_aligned_timespan_avg_out
        env_info['info'] = dict()

        for env in self.env_conf.env_list:

            env_info['info'][env] = dict()
            env_info['info'][env]['file_name'] = dict()

            out_path_endswith_env_name = os.path.join(self.env_aligned_timespan_avg_out, env)
            if not os.path.exists(out_path_endswith_env_name):
                os.makedirs(out_path_endswith_env_name)

            if env not in self.env_conf.env_no_need_avg:

                date_target_start = self.date_start
                date_target_end = date_target_start + relativedelta(years = self.year_span, months = (self.month_span-1), days = self.day_span)

                raster_value_alltime = np.empty(0)
                while date_target_end <= self.date_end:

                    date_target_log = f'{date_target_start.year}_{date_target_start.month:02d}_{date_target_start.day:02d}'
                    env_info['info'][env]['file_name'][date_target_log] = f'{env}/{env}_{date_target_log}_avg.tif'

                    raster_avg = np.zeros(extent_binary.shape)
                    for month_add in range(self.month_span):
                        date_target_element = date_target_start + relativedelta(years = self.year_span, months = month_add, days = self.day_span)

#                         env_aligned_file = f'wc2.1_2.5m_{env}_{date_target_element.year}-{date_target_element.month:02d}.tif'
                        env_aligned_file = env_filename_templates[env].replace('[YYYY]', f'{date_target_element.year}').replace('[MM]', f'{date_target_element.month:02d}')
                        
                        with rasterio.open(os.path.join(self.env_aligned_out, f'{env}/{env_aligned_file}.tif')) as img:
                            value = img.read(1)
                    
                        raster_avg = raster_avg + value
                        
                    raster_avg = raster_avg / self.month_span

                    with rasterio.open(
                        os.path.join(out_path_endswith_env_name,f'{env}_{date_target_log}_avg.tif'), 
                        'w',
                        height = extent_binary.shape[0],
                        width = extent_binary.shape[1], 
                        count = 1, 
                        nodata = -9, 
                        crs = extent_crs, 
                        dtype = rasterio.float32,
                        transform = extent_transform
                    ) as img:
                        img.write(raster_avg * extent_binary, 1)

                    raster_value_alltime = np.concatenate((raster_value_alltime, raster_avg[extent_binary == 1]))
                    
                    date_target_start = self.time_step(date_target_start)
                    date_target_end = date_target_start  + relativedelta(years = self.year_span, months = (self.month_span-1), days = self.day_span)

                env_info['info'][env]['mean'] = np.mean(raster_value_alltime)
                env_info['info'][env]['sd'] = np.std(raster_value_alltime)
            else:
                date_target_start = self.date_start
                date_target_end = date_target_start + relativedelta(years = self.year_span, months = (self.month_span-1), days = self.day_span)

                while date_target_end <= self.date_end:
                    
                    date_target_log = f'{date_target_start.year}_{date_target_start.month:02d}_{date_target_start.day:02d}'
                    env_info['info'][env]['file_name'][date_target_log] = f'{env}/{env}_avg.tif'

                    date_target_start = self.time_step(date_target_start)
                    date_target_end = date_target_start  + relativedelta(years = self.year_span, months = (self.month_span-1), days = self.day_span)  

                env_aligned_file = env_filename_templates[env]
                with rasterio.open(os.path.join(self.env_aligned_out, f'{env}/{env_aligned_file}.tif')) as img:
                    value = img.read(1)

                with rasterio.open(
                    os.path.join(out_path_endswith_env_name, f'{env}_avg.tif'), 
                    'w', 
                    height = extent_binary.shape[0], 
                    width = extent_binary.shape[1], 
                    count = 1, 
                    nodata = -9, 
                    crs = extent_crs, 
                    dtype = rasterio.float32, 
                    transform = extent_transform
                ) as img:
                    img.write(value * extent_binary, 1)

                env_info['info'][env]['mean'] = np.mean(value[extent_binary == 1]).astype(float)
                env_info['info'][env]['sd'] = np.std(value[extent_binary == 1]).astype(float)

        with open('./workspace/env_information.json', 'w') as f:
            json.dump(env_info, f)        
    
    #########################################
        
    def create_k_info(self):

        self.k_out = './workspace/raster_data/k'
        if not os.path.exists(self.k_out):
            os.makedirs(self.k_out)
        
        k_info = dict()
        k_info['dir_base'] = self.k_out
        k_info['file_name'] = dict()

        with rasterio.open('./workspace/extent_binary.tif') as raster_:
            extent_transform = raster_.transform
            extent_binary = raster_.read(1)
            extent_crs = raster_.crs

        xres = abs(extent_transform[0])
        yres = abs(extent_transform[4])

        if self.species_filter is None:
            self.species_filter = pd.read_csv('./workspace/species_data/occurrence_data/species_occurrence_filter.csv')
            self.species_list = np.unique(self.species_filter.species)

        species_filter = self.species_filter.copy()
        species_filter['week'] = species_filter.daysincebegin // 7

        date_target_start = self.date_start
        date_target_end = self.time_span(date_target_start)
        day_target_start = (date_target_start - self.date_start).days
        day_target_end = (date_target_end - self.date_start).days


        while date_target_start <= self.date_end:
            species_filter_day = species_filter[(species_filter.daysincebegin < day_target_end) & (species_filter.daysincebegin >= day_target_start)]
            week_list = list(set(species_filter_day.week))

            rst_time_span = np.zeros([extent_binary.shape[0], extent_binary.shape[1]])
            
            for week in week_list:
                rst_week = np.zeros([extent_binary.shape[0], extent_binary.shape[1]])
                species_filter_week = species_filter_day[species_filter_day.week == week]
                for i, row in species_filter_week.iterrows():
                    nlong = self.res_rounder(abs(row['decimalLongitude'] - self.spatial_conf.x_start) / xres, 0)
                    nlat = self.res_rounder(abs(self.spatial_conf.y_end - row['decimalLatitude']) / yres, 0)
                    rst_week[int(nlat), int(nlong)] = 1
                rst_time_span = rst_time_span + rst_week

            rst_result = np.where(extent_binary == 0, -9, rst_time_span / len(week_list))
            date_target_log = f'{date_target_start.year}_{date_target_start.month:02d}_{date_target_start.day:02d}'
            with rasterio.open(
                os.path.join(self.k_out, f'k_{date_target_log}.tif'),
                'w', 
                height = extent_binary.shape[0], 
                width = extent_binary.shape[1], 
                count = 1, 
                nodata = -9, 
                crs = extent_crs, 
                dtype = rasterio.float32, 
                transform = extent_transform
            ) as img:
                img.write(rst_result, 1)

            k_info['file_name'][date_target_log] = f'k_{date_target_log}.tif'

            date_target_start = self.time_step(date_target_start)
            date_target_end = self.time_span(date_target_start)
            day_target_start = (date_target_start - self.date_start).days
            day_target_end = (date_target_end - self.date_start).days

        with open('./workspace/k_information.json', 'w') as f:
            json.dump(k_info, f)        

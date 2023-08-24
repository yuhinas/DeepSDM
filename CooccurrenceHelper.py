import os
import pandas as pd
from datetime import datetime
import pickle as pkl
import numpy as np
import torch

class CooccurrenceHelper():
    def __init__ (self,
                  spatial_conf,
                  temporal_conf,
                  target_species=[],
                  distance_function=lambda x1, x2, y1, y2: ((x1-x2)**2 + (y1-y2)**2)**0.5,
                  output_dir='./workspace/species_data'):
        
        super().__init__()

        self.spatial_conf = spatial_conf
        self.temporal_conf = temporal_conf
        
        self.target_species = target_species
        self.x_start = spatial_conf.x_start
        self.y_start = spatial_conf.y_start
        self.x_end = spatial_conf.x_end
        self.y_end = spatial_conf.y_end
        
        # for 2 dimension
        self.cooccurrence_xy_limit = 2 * spatial_conf.out_res
        
        # for 1 dimension
        self.cooccurrence_day_limit = temporal_conf.cooccurrence_day_limit        
        
        # convert to datetime format
        self.date_start = datetime.strptime(temporal_conf.date_start, '%Y-%m-%d')
        self.date_end = datetime.strptime(temporal_conf.date_end, '%Y-%m-%d')
        
        # 
        self.day_first = 0
        self.day_last = (self.date_end - self.date_start).days
        
        #
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.occurrence_dir = os.path.join(self.output_dir, 'occurrence_data')
        if not os.path.exists(self.occurrence_dir):
            os.makedirs(self.occurrence_dir)

        self.cooccurrence_dir = os.path.join(self.output_dir, 'cooccurrence_data')
        if not os.path.exists(self.cooccurrence_dir):
            os.makedirs(self.cooccurrence_dir)

        self.get_distance = distance_function

    def filter_occurrence(self, gbif_occurrence_csv='./raw/species_occurrence_raw_50k.csv', nrows=None):

        self.gbif_occurrence_csv = gbif_occurrence_csv
        
        # read raw data directly download from gbif
        self.species_raw = pd.read_csv(self.gbif_occurrence_csv, 
                                       sep = '\t', 
                                       usecols = [
                                           'species',
                                           'decimalLatitude',
                                           'decimalLongitude',
                                           'day',
                                           'month',
                                           'year'],
                                       nrows=nrows,
                                      )
        
        # drop na values
        species_filter = self.species_raw.dropna().reset_index(drop = True)

        # filter species 
        # change species name
        species_filter['species'] = species_filter.apply(lambda x: f"{x['species'].split(' ')[0]}_{x['species'].split(' ')[1]}", axis = 1)
        if len(self.target_species) != 0:
            species_filter = species_filter[species_filter['species'].isin(self.target_species)].reset_index(drop = True)

        # add column 'date' with values of 'year-month-day'
        species_filter['date'] = species_filter['year'].astype(int).astype(str) + "-" + species_filter['month'].astype(int).astype(str) + "-" + species_filter['day'].astype(int).astype(str)
        species_filter['day'] = species_filter['day'].astype(int)

        # add column 'daysincebegin' with values of days after the begin of the first date
        # number of days since date_start
        species_filter['daysincebegin'] = species_filter.apply(lambda x: (datetime.strptime(x.date, '%Y-%m-%d') - self.date_start).days, axis = 1)

        # filter records by date
        species_filter = species_filter[(species_filter['daysincebegin'].values >= self.day_first) & (species_filter['daysincebegin'].values <= self.day_last)].reset_index(drop = True)
        
        
        # save filtered csv
        species_filter.to_csv(os.path.join(self.occurrence_dir, 'species_occurrence_filter.csv'), index = None)
        print(f"File: {os.path.join(self.occurrence_dir, 'species_occurrence_filter.csv')} saved.")
        
        self.species_filter = species_filter


    def aggregate_cooccurrence_units(self):

        def res_divider(a, b):
            digit_parts = str(b).split('.')
            assert(len(digit_parts) == 2)
            num_digits_after_decimal = min(len(digit_parts[1]), 6)
            return round(a / b, num_digits_after_decimal)

        
        sp_filter = self.species_filter.copy()
        self.data_unit = dict()
        
        sp_filter['daysincebeginUnit'] = ((sp_filter.daysincebegin - sp_filter.daysincebegin.min()) // self.cooccurrence_day_limit).astype(int)        
        sp_filter['decimalLongitudeUnit'] = res_divider(sp_filter.decimalLongitude - self.x_start, self.spatial_conf.out_res).astype(int)
        sp_filter['decimalLatitudeUnit'] = res_divider(sp_filter.decimalLatitude - self.y_start, self.spatial_conf.out_res).astype(int)

        def coocurr_agg (df, data_unit):
            t, x, y = df.name
            if t not in data_unit:
                data_unit[t] = dict()
            if round(x, 2) not in data_unit[t]:
                data_unit[t][x] = dict()
            self.data_unit[t][x][y] = df

        # order: t, x, y
        sp_filter.groupby(['daysincebeginUnit', 'decimalLongitudeUnit', 'decimalLatitudeUnit']).apply(coocurr_agg, self.data_unit)

    def count_cooccurrence(self, cooccurrence_counts_file='cooccurrence.csv'):
        
        data_unit = self.data_unit
        
        cooccurrence = dict()

        for t in data_unit:
            for x in data_unit[t]:
                for y in data_unit[t][x]:
                    for _, obsrv1 in data_unit[t][x][y].iterrows():

                        neighbor_units = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype=int).T.reshape(-1, 3)
                        
                        # Observations in the possibly co-occurrence temporal-spatial units
                        for dt, dx, dy in neighbor_units:
                            neighbor_t = t + dt
                            if neighbor_t not in data_unit:
                                continue

                            neighbor_x = x + dx
                            if neighbor_x not in data_unit[neighbor_t]:
                                continue

                            neighbor_y = y + dy
                            if neighbor_y not in data_unit[neighbor_t][neighbor_x]:
                                continue
                                
                            for _, obsrv2 in data_unit[neighbor_t][neighbor_x][neighbor_y].iterrows():
                                if obsrv1['species'] == obsrv2['species']:
                                    continue
                                if ((self.get_distance(float(obsrv1['decimalLongitude']), 
                                                       float(obsrv2['decimalLongitude']), 
                                                       float(obsrv1['decimalLatitude']), 
                                                       float(obsrv2['decimalLatitude'])) > self.cooccurrence_xy_limit) | 
                                    (abs(obsrv1['daysincebegin'] - obsrv2['daysincebegin']) > self.cooccurrence_day_limit)):
                                    continue
                                    
                                if (obsrv1['species'], obsrv2['species']) in cooccurrence:
                                    cooccurrence[(obsrv1['species'], obsrv2['species'])] += 1
                                elif (obsrv2['species'], obsrv1['species']) in cooccurrence:
                                    cooccurrence[(obsrv2['species'], obsrv1['species'])] += 1
                                else:
                                    cooccurrence[(obsrv1['species'], obsrv2['species'])] = 1

        # save the cooccurrence csv
        with open(os.path.join(self.cooccurrence_dir, cooccurrence_counts_file), 'w', encoding='utf-8') as out_file:
            out_file.write(f"sp1\tsp2\tcounts\n")
            for key in cooccurrence:
                try:
                    out_file.write(f"{key[0]}\t{key[1]}\t{cooccurrence[key]}\n")
                except:
                    # WHAT is `mat` ???
                    print(key, mat[key])
        print(f'File: {os.path.join(self.cooccurrence_dir, cooccurrence_counts_file)} saved.')

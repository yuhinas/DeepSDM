import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import rasterio
from TaxaDataset_smoothviz import TaxaDataset_smoothviz
from TaxaDataset import TaxaDataset
import matplotlib.pyplot as plt
import time
import pytorch_lightning as pl
import mlflow

class LitDeepSDMData(pl.LightningDataModule):
    def __init__ (self, info, conf,
                  dir_result = 'result'):
        
        super().__init__()

        self.info = info
        self.conf = conf
        self.dir_result = dir_result
        self.batch_size = self.conf.base_batch_size * self.conf.num_train_subsample_stacks

        if not os.path.isdir(f'./tmp'):
            os.makedirs(f'./tmp')
        
        self.geo_extent_file = './workspace/extent_binary.tif'
        meta_json_files = {
            'env_inf': './workspace/env_information.json',
            'sp_inf': './workspace/species_information.json',
            'k_inf': './workspace/k_information.json', 
            'co_vec': './workspace/cooccurrence_vector.json',
        }
        
        # define self.env_inf, self.sp_inf, self.co_vec
        for item_ in meta_json_files.items():
            print(item_[1])
            with open(item_[1]) as f:
                setattr(self, item_[0], json.load(f))

        with rasterio.open(self.geo_extent_file) as f:
            self.geo_extent = ToTensor()(f.read(1))
            self.geo_transform = f.transform
            self.geo_crs = f.crs

    def _save_trainval_split(self, dataset):

        self.partition_extent = dataset.split_tif[:dataset.height_original, :dataset.width_original].numpy()
        
        fig_ = plt.figure()
        plt.imshow(self.geo_extent.squeeze() * (1 + self.partition_extent.squeeze()) * .25)
        mlflow.log_figure(fig_, './workspace/train_val_geo_extent.png')
        plt.close()
        
#         current_experiment=dict(mlflow.get_experiment_by_name(self.conf.experiment_name))
#         run_name = current_experiment['run_name']
        
        active_run_ = mlflow.active_run()
        run_name = active_run_.info.run_name
        
        if not os.path.isdir(f'./tmp/{self.conf.experiment_name}'):
            os.makedirs(f'./tmp/{self.conf.experiment_name}')

        tmp_output_file = f'./tmp/{self.conf.experiment_name}/{run_name}_partition_extent.tif'
        with rasterio.open(tmp_output_file,
                           'w', 
                           crs = self.geo_crs,
                           transform = self.geo_transform,
                           height = dataset.height_original, 
                           width = dataset.width_original, 
                           count = 1, 
                           nodata = 0, 
                           dtype = rasterio.int16) as img_towrite:
            
            img_towrite.write(self.partition_extent, 1)
        mlflow.log_artifact(tmp_output_file)
        
    def _load_env_list(self, stage_date_list, stage_env_list, stage_species_list):
        # env
        ### The order is date x env
        env_stack = {
            'date_list' : stage_date_list,
            'env_list' : stage_env_list,
        }
        env_tensor_list = []
        for date_ in stage_date_list:
            date_env_list = []
            y_m = '-'.join(date_.split('-')[:-1])
            for env_ in stage_env_list:
                with rasterio.open(os.path.join(self.env_inf['dir_base'], f"{self.env_inf['info'][env_][y_m]['tif_span_avg']}")) as f:
                    img_ = ToTensor()(f.read(1)).cuda()
                img = img_.where(self.geo_extent.cuda() == 1, torch.normal(self.env_inf['info'][env_]['mean'], self.env_inf['info'][env_]['sd'], img_.shape).cuda())
                if env_ not in self.info.non_normalize_env_list:
                    img_norm = (img - self.env_inf['info'][env_]['mean']) / self.env_inf['info'][env_]['sd']
                else:
                    img_norm = img
                date_env_list.append(img_norm)
            env_tensor_list.append(torch.cat(date_env_list)[None, ])
        env_stack['tensor'] = torch.cat(env_tensor_list)
        
        return env_stack
    
    def _load_label_list(self, stage_date_list, stage_env_list, stage_species_list):
        # label
        ########### WARNING, check later if the label order is aligned with the env order
        ### The order is species x date
        label_stack = {
            'species_date' : [],
            'species' : [],
            'date' : [],
        }
        label_tensor_list = []
        for species_ in stage_species_list:
            for date_ in stage_date_list:
                label_stack['species_date'].append(f'{species_}_{date_}')
                label_stack['species'].append(f'{species_}')
                label_stack['date'].append(f'{date_}')
                with rasterio.open(os.path.join(self.sp_inf['dir_base'], f"{self.sp_inf['file_name'][species_][date_]}")) as f:
                    label_tensor_list.append(ToTensor()(f.read(1)))
                                             
        t = torch.cat(label_tensor_list)
        label_stack['tensor'] = torch.where(t < 0, 0, t)
        
        return label_stack

    def _load_k2_list(self, stage_date_list, stage_env_list, stage_species_list):        
        # k2 value
        k2_stack = {
            'date_list' : stage_date_list,
        }
        k2_tensor_list = []
        for date_ in stage_date_list:
            with rasterio.open(os.path.join(self.k_inf['dir_base'], f"{self.k_inf['file_name'][date_]}")) as f:
                k2_tensor_list.append(ToTensor()(f.read(1)))
        k2_stack['tensor'] = torch.cat(k2_tensor_list)
        
        return k2_stack

    def _load_embedding_list(self, stage_date_list, stage_env_list, stage_species_list):        
        # embeddings
        embedding = dict()
        for species_ in stage_species_list:
            embedding[species_] = self.co_vec[species_]
            
        return embedding
            
    def _load_meta_list(self, stage_date_list, stage_env_list, stage_species_list, stage):
        torch.save(self._load_env_list(stage_date_list, stage_env_list, stage_species_list), f'./tmp/env_stack_{stage}.pth')
        torch.save(self._load_embedding_list(stage_date_list, stage_env_list, stage_species_list), f'./tmp/embedding_{stage}.pth')
        torch.save(self._load_label_list(stage_date_list, stage_env_list, stage_species_list), f'./tmp/label_stack_{stage}.pth')
        torch.save(self._load_k2_list(stage_date_list, stage_env_list, stage_species_list), f'./tmp/k2_stack_{stage}.pth')

    def prepare_data(self):
        
        torch.cuda.synchronize()
        start_time = time.time()

        file_missing = False
        for prefix_ in ['env_stack', 'embedding', 'label_stack', 'k2_stack']:
            for stage_ in ['train', 'val', 'smoothviz']:
                if not os.path.exists(f'./tmp/{prefix_}_{stage_}.pth'):
                    file_missing = True
                if file_missing:
                    break
            if file_missing:
                break

        if file_missing:
            print("Missing some cached files, re-caching...")
            env_list = self.info.env_list
            
            date_list_train = self.info.date_list
            species_list_train = self.info.species_list

            date_list_val = self.info.date_list_val
            species_list_val = self.info.species_list_val

            date_list_smoothviz = self.info.date_list_smoothviz
            species_list_smoothviz = self.info.species_list_smoothviz

            ### caching staged lists
            # train
            print("train")
            self._load_meta_list(
                date_list_train, 
                env_list, 
                species_list_train,
                'train'
            )

            # val
            print("val")
            self._load_meta_list(
                date_list_val, 
                env_list, 
                species_list_val,
                'val'
            )

            # smoothviz
            print("smoothviz")
            self._load_meta_list(
                date_list_smoothviz, 
                env_list, 
                species_list_smoothviz,
                'smoothviz'
            )

        torch.cuda.synchronize()
        print(f'Data prepared in {time.time() - start_time} seconds.')


    def setup(self, stage=None):

        ###
        torch.cuda.synchronize()
        start_time = time.time()
        self.env_stack_train = torch.load('./tmp/env_stack_train.pth', map_location='cpu')
        self.embedding_train = torch.load('./tmp/embedding_train.pth', map_location='cpu')
        self.label_stack_train = torch.load('./tmp/label_stack_train.pth', map_location='cpu')
        self.k2_stack_train = torch.load('./tmp/k2_stack_train.pth', map_location='cpu')
        print(f'train ############################################## {self.trainer.global_rank}')
        if self.trainer.global_rank == 0:
            dataset_train = TaxaDataset(
                self.env_stack_train, 
                self.embedding_train, 
                self.label_stack_train, 
                self.k2_stack_train, 'train', self.conf, self.trainer.global_rank
            )
        else:
            dataset_train = None
        self.trainer.strategy.barrier()
        dataset_train = self.trainer.strategy.broadcast(dataset_train, src=0)
        self.dataset_train = dataset_train
        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data train loaded in {time.time() - start_time} seconds.')

        ###
        torch.cuda.synchronize()
        start_time = time.time()
        self.label_stack_val = torch.load('./tmp/label_stack_val.pth', map_location='cpu')
        print(f'val ############################################## {self.trainer.global_rank}')
        if self.trainer.global_rank == 0:
            dataset_val = TaxaDataset(
                torch.load('./tmp/env_stack_val.pth', map_location='cpu'), 
                torch.load('./tmp/embedding_val.pth', map_location='cpu'), 
                self.label_stack_val, 
                torch.load('./tmp/k2_stack_val.pth', map_location='cpu'), 'val', self.conf, self.trainer.global_rank
            )
        else:
            dataset_val = None
        dataset_val = self.trainer.strategy.broadcast(dataset_val, src=0)
        self.dataset_val = dataset_val
        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data val loaded in {time.time() - start_time} seconds.')

        ###
        torch.cuda.synchronize()
        start_time = time.time()
        print(f'train on val ############################################## {self.trainer.global_rank}')
        if self.trainer.global_rank == 0:
            dataset_train_on_val = TaxaDataset(
                self.env_stack_train, 
                self.embedding_train, 
                self.label_stack_train, 
                self.k2_stack_train, 'val', self.conf, self.trainer.global_rank
            )
        else:
            dataset_train_on_val = None
        dataset_train_on_val = self.trainer.strategy.broadcast(dataset_train_on_val, src=0)
        self.dataset_train_on_val = dataset_train_on_val
        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data train_on_val loaded in {time.time() - start_time} seconds.')
        
        ###
        ##########################################################        
        torch.cuda.synchronize()
        start_time = time.time()
        self.env_stack_smoothviz = torch.load('./tmp/env_stack_smoothviz.pth', map_location='cpu')
        self.embedding_smoothviz = torch.load('./tmp/embedding_smoothviz.pth', map_location='cpu')
        self.label_stack_smoothviz = torch.load('./tmp/label_stack_smoothviz.pth', map_location='cpu')
        self.k2_stack_smoothviz = torch.load('./tmp/k2_stack_smoothviz.pth', map_location='cpu')
        
        if self.trainer.global_rank == 0:
            self.datasets_smoothviz = []
            print ("Setting up dataset in log_img...")
            for idx_species_date in range(len(self.label_stack_smoothviz['species_date'])):

                self.datasets_smoothviz.append(
                    TaxaDataset_smoothviz(
                        idx_species_date, 
                        self.env_stack_smoothviz, self.embedding_smoothviz, self.label_stack_smoothviz, 
                        self.conf.subsample_height, self.conf.subsample_width, self.conf.num_smoothviz_steps
                    )
                )

        self.trainer.strategy.barrier()

        torch.cuda.synchronize()
        print(f'({self.trainer.global_rank}) Data smoothviz loaded in {time.time() - start_time} seconds.')

        ##########################################################
        
        if self.trainer.global_rank == 0:
            self._save_trainval_split(self.dataset_train)
            
        self.trainer.strategy.barrier()
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, self.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return [
            DataLoader(self.dataset_train_on_val, self.batch_size, shuffle=False, num_workers=16, pin_memory=True), # eval train dataset first
            DataLoader(self.dataset_val, self.batch_size, shuffle=False, num_workers=16, pin_memory=True), # Why shuffling here?
        ]

    def smoothviz_dataloader(self):
        return [DataLoader(dataset_smoothviz, batch_size=self.batch_size, shuffle=False, num_workers=0) for dataset_smoothviz in self.datasets_smoothviz]

    def predict_dataloader(
        self,
        date_list = ['2020_01_01', '2020_04_01', '2020_07_01', '2020_10_01'],
        species_list = ['Psilopogon_nuchalis', 'Yuhina_brunneiceps', 'Corvus_macrorhynchos']
    ):
        
        env_list = self.info.env_list
        
        self._load_meta_list(
            date_list, 
            env_list, 
            species_list,
            'predict',
        )

        self.env_stack_predict = torch.load('./tmp/env_stack_predict.pth', map_location='cpu')
        self.embedding_predict = torch.load('./tmp/embedding_predict.pth', map_location='cpu')
        self.label_stack_predict = torch.load('./tmp/label_stack_predict.pth', map_location='cpu')
        self.k2_stack_predict = torch.load('./tmp/k2_stack_predict.pth', map_location='cpu')
        
        self.datasets_predict = []
        print ("Setting up dataset for prediction...")
        for idx_species_date in range(len(self.label_stack_predict['species_date'])):

            self.datasets_predict.append(
                TaxaDataset_smoothviz(
                    idx_species_date, 
                    self.env_stack_predict, self.embedding_predict, self.label_stack_predict, 
                    self.conf.subsample_height, self.conf.subsample_width, self.conf.num_predict_steps
                )
            )
        
        return [DataLoader(dataset_predict, batch_size=self.batch_size, shuffle=False, num_workers=0) for dataset_predict in self.datasets_predict]
    
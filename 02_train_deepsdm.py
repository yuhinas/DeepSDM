import time
import pytorch_lightning as pl
from types import SimpleNamespace
import mlflow
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from LitDeepSDMData import LitDeepSDMData
from LitUNetSDM import LitUNetSDM

# List those environment rasters for training, must be equal to or be a subset of env_list in step 01_prepare_data.
env_list = ['tmax', 'tmin', 'prec', 'elev', 'evi']
# Those environment rasters that don't need normalization or are already normalized, e.g. PCA results.
non_normalize_env_list = []

date_list = []
# lists of selected dates for training
# format: YYYY_MM_01
# The python range exclude the stop value (here e.g. 2020)
# So here we generate from 2016_01_01 to 2019_12_01
# We keep data of 2020 for validation/prediction
for y_ in range(2016, 2020):
    for m_ in range(1, 13):
        date_list.append(f'{y_:04d}-{m_:02d}-01')

# list of species that selected for training
species_list = ['Psilopogon_nuchalis', 'Yuhina_brunneiceps', 'Corvus_macrorhynchos', 'Zosterops_simplex', 'Passer_montanus', 'Spilopelia_chinensis', 'Acridotheres_javanicus']

# lists of species and dates for smooth visualization preview
date_list_smoothviz = ['2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01']
species_list_smoothviz = ['Psilopogon_nuchalis', 'Yuhina_brunneiceps', 'Corvus_macrorhynchos']

# packed the species lists and date lists for training
info = SimpleNamespace(**dict(
    env_list = sorted(env_list),
    non_normalize_env_list = sorted(non_normalize_env_list),
    species_list = sorted(species_list),
    species_list_val = sorted(species_list),
    species_list_smoothviz = sorted(species_list_smoothviz),
    date_list = sorted(date_list),
    date_list_val = sorted(date_list),
    date_list_smoothviz = sorted(date_list_smoothviz)
))


# We should find a relatively decent effective batch size
# The effective batch size = base_batch_size * num_train_subsample_stacks * num_of_devices * num_of_nodes
conf = SimpleNamespace(**dict(
    experiment_name = "DeepSDM DEMO",
    epochs = 300, # Change the values according to any baseline result. For example, if the training stops at epoch 300 and the metric monitored is still ascending (descending), extend this value
    base_batch_size = 64, # modify this value if OOM (out of cude memory) encountered
    num_vector = 64, # number of vectors for the embeddings of species co-occurrence
    num_env = len(env_list), # number of environmental layers
    num_train_subsample_stacks = 2, # the number of random training subsamples that sampled from the training grids
    num_val_subsample_stacks = 1, # the number of validation subsamples that sampled from the validation grids
    subsample_height = 56, # must be 8 * N, for the requirements of UNET
    subsample_width = 56, # must be 8 * N, for the requirements of UNET
    num_smoothviz_steps = 7, # the steps for smooth visualization, must be a factor of 56, the higher the smoother, and slower
    num_predict_steps = 14, # the steps for smooth visualization, must be a factor of 56, the higher the smoother, and slower
    learning_rate = 1e-4, # Usually you don't have to change this
    k2 = 1, # constant weight of k2
    k3 = 1/12, # constant value of k3, confidence of absence
    p = 1/3 # the exponential decay rate 
))

### LOGGER
# Use mlflow to auto logging pytorch lightning everything
mlflow.set_experiment(conf.experiment_name)
mlflow.pytorch.autolog()
timelog = time.strftime('%Y%m%d%H%M%S', time.localtime())

### DATA
# initialize the lightning data module
deep_sdm_data = LitDeepSDMData(info=info, conf=conf)

### MODLE
# initialize the DeepSDM lightning module
model = LitUNetSDM(info=info, conf=conf)

# Check the metrics monitored and save models (checkpoints)
checkpoint_callback = ModelCheckpoint(
    save_last=True,
    save_top_k=1,
    verbose=True,
    monitor='f1_train',
    mode='max'
)

# setup early stopper for reaching a plateau
early_stop_callback = EarlyStopping(monitor="f1_train", min_delta=0.00, patience=100, 
                                    verbose=True, mode="max")

### TRAINER (all kinds of loops)
# change the devices number if you have only 1 GPU or more GPUs
# We use half precision for less memory usage and faster calculations
trainer = pl.Trainer(
    max_epochs=conf.epochs, devices=2, accelerator="gpu", check_val_every_n_epoch=1,
    strategy=DDPStrategy(static_graph=True), # use 'ddp_fork_find_unused_parameters_true' instead on jupyter or colab
    precision=16, callbacks=[checkpoint_callback, early_stop_callback]
)

# Start the training!
with mlflow.start_run(run_name=f"{timelog}-{trainer.global_rank}"):
    ### START
    trainer.fit(model, datamodule=deep_sdm_data)
    
    
# run `mlflow ui` in the console at the path that contains 'mlruns' for monitoring the trainig
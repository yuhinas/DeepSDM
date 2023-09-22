import time
import pytorch_lightning as pl
from types import SimpleNamespace
import mlflow
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from LitDeepSDMData import LitDeepSDMData
from LitUNetSDM import LitUNetSDM
import yaml

# Define timelog
timelog = time.strftime('%Y%m%d%H%M%S', time.localtime())

# load configurations
with open('DeepSDM_conf.yaml', 'r') as f:
    DeepSDM_conf = yaml.load(f, Loader = yaml.FullLoader)
DeepSDM_conf = SimpleNamespace(**DeepSDM_conf)

# lists of selected dates for training
# format: YYYY_MM_01
# The python range exclude the stop value (here e.g. 2020)
# So here we generate from 2016_01_01 to 2019_12_01
# We keep data of 2020 for validation/prediction
# date_list_train = []
# for y_ in range(2000, 2019):
#     for m_ in range(1, 13):
#         date_list_train.append(f'{y_:04d}-{m_:02d}-01')

        
# packed the species lists and date lists for training
info = SimpleNamespace(**dict(
    env_list = sorted(DeepSDM_conf.env_list),
    non_normalize_env_list = sorted(DeepSDM_conf.non_normalize_env_list),
    species_list = sorted(DeepSDM_conf.species_list_train),
    species_list_val = sorted(DeepSDM_conf.species_list_train),
    species_list_smoothviz = sorted(DeepSDM_conf.species_list_smoothviz),
    date_list = sorted(DeepSDM_conf.date_list_train),
    date_list_val = sorted(DeepSDM_conf.date_list_train),
    date_list_smoothviz = sorted(DeepSDM_conf.date_list_smoothviz)
))
conf = SimpleNamespace(**DeepSDM_conf.conf)
conf.num_env = len(DeepSDM_conf.env_list) # number of environmental layers
conf.num_vector = DeepSDM_conf.embedding_conf['num_vector'] # number of vectors for the embeddings of species co-occurrence

# update configurations

### LOGGER
# Use mlflow to auto logging pytorch lightning everything
mlflow.set_experiment(conf.experiment_name)
mlflow.pytorch.autolog()

### DATA
# initialize the lightning data module
deep_sdm_data = LitDeepSDMData(info=info, conf=conf)

### MODEL
# initialize the DeepSDM lightning module
model = LitUNetSDM(info=info, conf=conf)

# Check the metrics monitored and save models (checkpoints)
model_checkpoint_conf = SimpleNamespace(**DeepSDM_conf.model_checkpoint_conf)
checkpoint_callback = ModelCheckpoint(
    save_last = model_checkpoint_conf.save_last,
    save_top_k = model_checkpoint_conf.save_top_k,
    verbose = model_checkpoint_conf.verbose,
    monitor = model_checkpoint_conf.monitor,
    mode = model_checkpoint_conf.mode
)

# setup early stopper for reaching a plateau
earlystopping_conf = SimpleNamespace(**DeepSDM_conf.earlystopping_conf)
early_stop_callback = EarlyStopping(
    monitor = earlystopping_conf.monitor,                                 
    min_delta = earlystopping_conf.min_delta, 
    patience = earlystopping_conf.patience,                                 
    verbose = earlystopping_conf.verbose, 
    mode = earlystopping_conf.mode)

### TRAINER (all kinds of loops)
# change the devices number if you have only 1 GPU or more GPUs
# We use half precision for less memory usage and faster calculations
trainer_conf = SimpleNamespace(**DeepSDM_conf.trainer_conf)
trainer = pl.Trainer(
    max_epochs = conf.epochs, 
    devices = trainer_conf.devices, 
    accelerator = trainer_conf.accelerator, 
    check_val_every_n_epoch = trainer_conf.check_val_every_n_epoch,
    strategy = DDPStrategy(static_graph=True), # use 'ddp_fork_find_unused_parameters_true' instead on jupyter or colab
    precision = trainer_conf.precision, 
    callbacks = [checkpoint_callback, early_stop_callback], 
    logger = pl.loggers.TensorBoardLogger(save_dir = './', version = timelog)
)

# Start the training!
with mlflow.start_run(run_name=f"{timelog}-{trainer.global_rank}"):
    ### START
    trainer.fit(model, datamodule=deep_sdm_data)
    
    
# run `mlflow ui` in the console at the path that contains 'mlruns' for monitoring the trainig
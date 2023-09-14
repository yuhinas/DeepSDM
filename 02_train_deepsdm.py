import time
import pytorch_lightning as pl
from types import SimpleNamespace
import mlflow
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from LitDeepSDMData import LitDeepSDMData
from LitUNetSDM import LitUNetSDM

# List those environment rasters for training, must be equal to or be a subset of env_list in step 01_prepare_data.
env_list = ['clt', 'cmi', 'hurs', 'pet', 'pr', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin', 'vpd', 'landcover_PC00', 'landcover_PC01', 'landcover_PC02', 'landcover_PC03', 'landcover_PC04', 'landcover_PC05', 'landcover_PC06', 'landcover_PC07', 'ele']
# Those environment rasters that don't need normalization or are already normalized, e.g. PCA results.
non_normalize_env_list = ['landcover_PC00', 'landcover_PC01', 'landcover_PC02', 'landcover_PC03', 'landcover_PC04', 'landcover_PC05', 'landcover_PC06', 'landcover_PC07']

date_list = []
# lists of selected dates for training
# format: YYYY_MM_01
# The python range exclude the stop value (here e.g. 2020)
# So here we generate from 2016_01_01 to 2019_12_01
# We keep data of 2020 for validation/prediction
for y_ in range(2000, 2019):
    for m_ in range(1, 13):
        date_list.append(f'{y_:04d}-{m_:02d}-01')

# list of species that selected for training
# species_list = ['Abroscopus_albogularis', 'Accipiter_trivirgatus', 'Accipiter_virgatus', 'Acridotheres_cristatellus', 'Actinodura_morrisoniana', 'Aegithalos_concinnus', 'Aix_galericulata', 'Alauda_gulgula', 'Alcedo_atthis', 'Alcippe_morrisonia', 'Amaurornis_phoenicurus', 'Anas_poecilorhyncha', 'Apus_nipalensis', 'Arborophila_crudigularis', 'Ardea_alba', 'Ardea_purpurea', 'Bambusicola_sonorivox', 'Brachypteryx_goodfellowi', 'Bubulcus_ibis', 'Butorides_striata', 'Caprimulgus_affinis', 'Carpodacus_formosanus', 'Cecropis_striolata', 'Centropus_bengalensis', 'Chalcophaps_indica', 'Charadrius_alexandrinus', 'Charadrius_dubius', 'Cinclus_pallasii', 'Cisticola_exilis', 'Cisticola_juncidis', 'Columba_pulchricollis', 'Coracina_macei', 'Corvus_macrorhynchos', 'Stachyridopsis_ruficeps', 'Delichon_dasypus', 'Dendrocitta_formosae', 'Dendrocopos_leucotos', 'Dicaeum_ignipectus', 'Dicaeum_minullum', 'Dicrurus_aeneus', 'Dicrurus_macrocercus', 'Egretta_garzetta', 'Egretta_sacra', 'Elanus_caeruleus', 'Enicurus_scouleri', 'Erpornis_zantholeuca', 'Pomatorhinus_erythrocnemis', 'Falco_peregrinus', 'Ficedula_hyperythra', 'Fulvetta_formosana', 'Gallicrex_cinerea', 'Gallinula_chloropus', 'Garrulax_taewanus', 'Garrulus_glandarius', 'Gorsachius_melanolophus', 'Heterophasia_auricularis', 'Himantopus_himantopus', 'Hirundapus_cochinchinensis', 'Hirundo_tahitica', 'Horornis_acanthizoides', 'Horornis_fortipes', 'Hydrophasianus_chirurgus', 'Hypothymis_azurea', 'Hypsipetes_amaurotis', 'Hypsipetes_leucocephalus', 'Ictinaetus_malayensis', 'Ixobrychus_cinnamomeus', 'Ixobrychus_sinensis', 'Ketupa_flavipes', 'Lanius_schach', 'Gallirallus_striatus', 'Liocichla_steerii', 'Locustella_alishanensis', 'Lonchura_atricapilla', 'Lonchura_punctulata', 'Lonchura_striata', 'Lophura_swinhoii', 'Parus_holsti', 'Macropygia_tenuirostris', 'Milvus_migrans', 'Monticola_solitarius', 'Motacilla_alba', 'Myiomela_leucura', 'Myophonus_insularis', 'Niltava_vivida', 'Ninox_japonica', 'Nisaetus_nipalensis', 'Nucifraga_caryocatactes', 'Nycticorax_nycticorax', 'Oriolus_chinensis', 'Oriolus_traillii', 'Otus_elegans', 'Otus_lettia', 'Otus_spilocephalus', 'Parus_monticolus', 'Passer_cinnamomeus', 'Passer_montanus', 'Pericrocotus_solaris', 'Periparus_ater', 'Pernis_ptilorhynchus', 'Phasianus_colchicus', 'Phoenicurus_fuliginosus', 'Picus_canus', 'Pnoepyga_formosana', 'Pomatorhinus_musicus', 'Prinia_flaviventris', 'Prinia_inornata', 'Prinia_striata', 'Prunella_collaris', 'Psilopogon_nuchalis', 'Garrulax_poecilorhynchus', 'Garrulax_ruficeps', 'Pycnonotus_sinensis', 'Pycnonotus_taivanus', 'Pyrrhula_nipalensis', 'Pyrrhula_owstoni', 'Rallina_eurizonoides', 'Regulus_goodfellowi', 'Riparia_chinensis', 'Rostratula_benghalensis', 'Alcippe_brunnea', 'Sinosuthora_webbiana', 'Sitta_europaea', 'Poecile_varius', 'Spilopelia_chinensis', 'Spilornis_cheela', 'Spizixos_semitorques', 'Sternula_albifrons', 'Streptopelia_orientalis', 'Streptopelia_tranquebarica', 'Strix_leptogrammica', 'Strix_nivicolum', 'Suthora_verreauxi', 'Coturnix_chinensis', 'Syrmaticus_mikado', 'Tachybaptus_ruficollis', 'Glaucidium_brodiei', 'Tarsiger_indicus', 'Tarsiger_johnstoniae', 'Terpsiphone_atrocaudata', 'Treron_formosae', 'Treron_sieboldii', 'Trochalopteron_morrisonianum', 'Troglodytes_troglodytes', 'Turdus_mandarinus', 'Turdus_poliocephalus', 'Turnix_suscitator', 'Turnix_sylvaticus', 'Tyto_longimembris', 'Urocissa_caerulea', 'Yuhina_brunneiceps', 'Yungipicus_canicapillus', 'Porzana_fusca', 'Zoothera_dauma', 'Zosterops_japonicus', 'Zosterops_meyeni', 'Zosterops_simplex']
species_list = ['Passer_cinnamomeus', 'Carpodacus_formosanus', 'Acridotheres_cristatellus', 'Nisaetus_nipalensis', 'Corvus_macrorhynchos', 'Pycnonotus_taivanus', 'Glaucidium_brodiei', 'Psilopogon_nuchalis', 'Treron_sieboldii', 'Lonchura_atricapilla', 'Prunella_collaris', 'Hirundapus_cochinchinensis', 'Alauda_gulgula', 'Pycnonotus_sinensis', 'Suthora_verreauxi', 'Garrulax_ruficeps', 'Centropus_bengalensis', 'Treron_formosae', 'Otus_lettia', 'Enicurus_scouleri', 'Spizixos_semitorques', 'Ketupa_flavipes', 'Garrulax_taewanus', 'Motacilla_alba', 'Sitta_europaea', 'Tarsiger_johnstoniae', 'Poecile_varius', 'Elanus_caeruleus', 'Alcippe_morrisonia', 'Oriolus_traillii', 'Ardea_purpurea', 'Phasianus_colchicus', 'Horornis_acanthizoides', 'Strix_nivicolum', 'Lophura_swinhoii', 'Delichon_dasypus', 'Lonchura_striata', 'Yuhina_brunneiceps', 'Myophonus_insularis', 'Periparus_ater', 'Zosterops_simplex', 'Syrmaticus_mikado', 'Prinia_flaviventris']
# species_list = ['Psilopogon_nuchalis', 'Yuhina_brunneiceps', 'Corvus_macrorhynchos', 'Syrmaticus_mikado']


# lists of species and dates for smooth visualization preview
date_list_smoothviz = ['2018-01-01', '2018-04-01', '2018-07-01', '2018-10-01']
species_list_smoothviz = ['Psilopogon_nuchalis', 'Yuhina_brunneiceps', 'Corvus_macrorhynchos', 'Syrmaticus_mikado']

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
    max_epochs=conf.epochs, devices=4, accelerator="gpu", check_val_every_n_epoch=1,
    strategy=DDPStrategy(static_graph=True), # use 'ddp_fork_find_unused_parameters_true' instead on jupyter or colab
    precision=32, callbacks=[checkpoint_callback, early_stop_callback]
)

# Start the training!
with mlflow.start_run(run_name=f"{timelog}-{trainer.global_rank}"):
    ### START
    trainer.fit(model, datamodule=deep_sdm_data)
    
    
# run `mlflow ui` in the console at the path that contains 'mlruns' for monitoring the trainig
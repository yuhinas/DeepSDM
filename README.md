## Running instructions

### 1. Run the 01_prepare_data.ipynb in the jupyterlab.

### 2. Run the 02_train_deepsdm.py in the console.
```console
$ python 02_train_deepsdm.py
```

### 3. You can run a mlflow ui for monitoring the training process.
Open a new console and run the command under deepsdm_ebbe_nielsen directory.
```console
$ mlflow ui
```
Access the monitoring at https://127.0.0.1:5000

### 4. Run the 03_make_prediction.ipynb in the jupyterlab.

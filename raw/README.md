## In the directory `raw`:

### Put the species occurrence csv at the root.
For example:
```
./species_occurrence_raw_50k.csv
```

### Put each type of environment rasters at ./env/{env_type} seperate sub directories.

Rasters with temporal resolution of month
```
./tmax/wc2.1_2.5m_tmax_2016-01.tif
./tmax/wc2.1_2.5m_tmax_2016-02.tif
...
./tmax/wc2.1_2.5m_tmax_2020-12.tif
...
./prec/wc2.1_2.5m_prec_2016-01.tif
./prec/wc2.1_2.5m_pred_2016-01.tif
./prec/wc2.1_2.5m_prec_2016-01.tif
```
Static Rasters
```
./elev/wc2.1_2.5m_elev.tif
```

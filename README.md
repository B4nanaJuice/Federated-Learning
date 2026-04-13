# Tests on attacking and defending centralized machine learning
---
## 📑 Project overview
---
## 📚 Data preprocessing phase
### Dataset structure
The main file used here for the data preprocessing is `data/input/Final_Energy_Dataset_with_weather.csv`, each line of the file corresponding to a time point.
#### Main columns
The dataset contains :
##### Time variable
| Variable | Description |
|--------|-------------|
| date | Timestamp of the measurement |
##### Weather variables
Those variables are the same for each building.
| Variable | Description |
|----------|-------------|
| weekday | Day of the week From 0 (Monday) to 6 (Sunday) |
| tod_sin | Cyclic encoding of the time of the day |
| tod_cos | Cyclic encoding of the time of the day |
| temp | Temperature |
| rhum | Relative humidity |
| wspd | Wind speed |
| wdir | Wind direction |

Those variables are called exogenous variables because they influence energy consumption but are not directly energy data.
##### Energy variables
For each building (i)
| Variable | Description |
|----------|-------------|
| $load_{i}$ | Building's energy consumption |
| $pv_{i}$ | Photovoltaic energy production |
| $net_{i}$ | Net consumption ($pv_i-load_i$) |

Each building represents a client in the federated learning process.
#### Preprocessing pipeline
When calling the preprocessing script (in `data/preprocessing.py` or directly importing with `from data import run_preprocessing`), multiple steps are being done :
1. Dataset loading, the script reads the input file located in `data/input`
2. Data extraction for each client (building)
3. Date preprocessing, by computing the weekday and the cyclic time of the day (`tod_sin` and `tod_cos`)
4. Net consumption computation
5. Column reordering, for easier readability
6. Data normalization [TODO]
7. Time-based data splitting (`train`, `validation` and `test`) [TODO]
8. Converting the object from a `pandas.DataFrame` to a `torch.Tensor`
9. Saving the `torch.Tensor` into the `data/processed` directory
#### Data extraction for each client

#### Date preprocessing

#### Net consumption computation

#### Column reordering

#### Data normalization

#### Time-based data splitting

#### Converting and saving

---
## 📥 Installation guide
---

[Ctrl+ K V] pour l'overview

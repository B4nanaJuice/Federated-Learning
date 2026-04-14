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
| weekday | Day of the week From 0 (Monday) to 6 (Sunday) |
| tod_sin | Cyclic encoding of the time of the day |
| tod_cos | Cyclic encoding of the time of the day |

##### Weather variables
Those variables are the same for each building.
| Variable | Description |
|----------|-------------|
| temp | Temperature |
| rhum | Relative humidity |
| wspd | Wind speed |
| wdir | Wind direction |
| pres | Pressure |

Those variables are called exogenous variables because they influence energy consumption but are not directly energy data.

##### Energy variables
For each building (i) :
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
6. Data normalization
7. Time-based data splitting (`train`, `validation` and `test`)
8. Converting the object from a `pandas.DataFrame` to a `torch.Tensor`
9. Saving the `torch.Tensor` into the `data/processed` directory
<details>
<summary>See more about the preprocessing pipeline</summary>

#### Data extraction for each client
Each client represents a building. The preprocessing script selects the following columns :
```
date
weather related columns (temp, rhum, wdir, wspd, and pres)
energy targets (load_X and pv_X where X is the building id)
```
The target variable depends on the chosen task : `target = load` or `target = pv` or `target = net`, where the target column will be renamed into `target`. 

#### Date preprocessing
The `date` column is converted into three features:
- `weekday`: Integer from 0 (Monday) to 6 (Sunday)
- `tod_sin` and `tod_cos`: Cyclic encodings of the time of day using sine and cosine transformations

Since measurements are recorded every 30 minutes, there are 48 time points per day. The cyclic encoding preserves the circular nature of time by mapping each time point to a position on a unit circle, ensuring that time point 0 and time point 47 are close in feature space.

The transformation is computed as:
```
tod_sin = sin(2π × time_point / 48)
tod_cos = cos(2π × time_point / 48)
```

#### Net consumption computation
The net consumption is computed for each client by subtracting the photovoltaic energy production from the building's energy consumption:

```
net_i = load_i - pv_i
```

This represents the energy that each building needs to draw from or feed back to the grid. Positive values indicate net consumption (more energy used than produced), while negative values indicate net production (more energy produced than consumed).

#### Column reordering
The columns are reordered for improved readability and logical grouping:

1. **Date-related columns** (`date`, `weekday`, `tod_sin`, `tod_cos`) - temporal information
2. **Weather-related columns** (`temp`, `rhum`, `wspd`, `wdir`, `pres`) - exogenous variables
3. **Energy-related columns** (`load`, `pv`, `net`) - target and related energy data

This ordering ensures that temporal context comes first, followed by environmental factors, and finally the energy consumption data to be modeled.

#### Data normalization
In order to stabilize the training process, data is normalized. Two methods are used here :
##### Energy normalization
Consumption and production are normalized with the `MinMaxScaler` where the data is normalized by doing :
```
x_scaled = (x - min) / (max - min)
```
The values are then between `0` and `1`. It is important to use the same scaler for the stored energy values and the energy target to ensure coherence between the input and the output.

##### Weather-related variables normalization
Exogenous variables are normalized using the `StandardScaler` where the data is normalized by doing :
```
x_scaled = (x - mean) / std
```
Data has then the following distribution :
```
mean ≈ 0
standard deviation ≈ 1
```
This normalization is coherent for weather variables.

#### Time-based data splitting
The dataset is divided into three disjoint sets using a time-based split strategy :

- **Training set** (70%): Used to train the model on historical data
- **Validation set** (20%): Used to tune hyperparameters and monitor performance during training
- **Test set** (10%): Used for final evaluation of model performance on unseen data

The split respects the temporal ordering of measurements, ensuring that the model is always trained on past data and evaluated on future data, which is crucial for time series forecasting tasks.

#### Converting and saving
Once preprocessing is complete, the `pandas.DataFrame` is converted into a `torch.Tensor` format for efficient computation on GPUs. This conversion enables seamless integration with PyTorch-based models during training and inference.

The conversion process:
1. Extracts numerical data from the DataFrame using `.values`
2. Creates a `torch.Tensor` object with appropriate data type
3. Serializes the tensor using `torch.save()` in `.pt` format

The processed tensors are saved to the `data/processed` directory with the following naming convention:
```
{split}/building_{client_id}.pt
```

Where:
- `client_id`: Building identifier
- `split`: Dataset partition (`train`, `val`, or `test`)

**Example**: `train/building_1.pt`

These `.pt` files can be directly loaded during model training and testing using `torch.load()`, eliminating the need for runtime preprocessing and significantly accelerating the training pipeline.
</details>

---
## 📥 Installation guide
---

[Ctrl+ K V] pour l'overview

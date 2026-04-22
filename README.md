# Tests on attacking and defending centralized machine learning
---
## 📑 Project overview
This project is dedicated to testing attack and defense mechanisms for federated learning systems in the context of smart grid energy forecasting. The federated learning framework keeps sensitive building-level energy data local while enabling collaborative model training. However, the training process remains vulnerable to various attacks including data poisoning, model inversion, and Byzantine failures.

**Project Status**: 🚧 Early Development

The project is currently in its early stages. The foundational data preprocessing pipeline and project structure are in place, but significant work remains to fully implement and validate attack and defense mechanisms.

### 📋 Development Roadmap

- [x] **Attack Mechanisms Implementation**
    - [x] Model poisoning attack module at client's side
    - [x] Model poisoning attack module at server's side
    
- [ ] **Defense Strategies**
    - [ ] Scoring method for clients and server
    - [ ] Anomaly detection for poisoned updates
    - [ ] Degraded federated learning mode
    
- [x] **Federated Learning Core**
    - [x] Complete server-client communication protocol
    - [x] Model update aggregation logic
    - [x] Round-based training loop
    
- [ ] **Testing & Evaluation**
    - [ ] Comprehensive test suite for attacks
    - [ ] Defense mechanism validation
    - [ ] Performance benchmarking
    - [x] Data visualisation
    
- [ ] **Documentation**
    - [ ] API documentation
    - [ ] Usage examples and tutorials
    - [ ] Results and findings report

### Context: Smart Grids and Energy Forecasting
Smart grids integrate distributed energy resources, IoT sensors, and machine learning to optimize energy distribution. Building-level energy consumption and photovoltaic production forecasting is critical for grid stability and efficient resource allocation. Each building (client) generates sensitive consumption data that reveals occupancy patterns and behavior, making data privacy a primary concern.

### Why Test Defense Mechanisms?
Federated learning preserves privacy by keeping data local, but the training process remains vulnerable to:
- **Data poisoning attacks**: Malicious clients modify model's weights to compromise overall performance
- **Byzantine failures**: Compromised clients send adversarial gradients

Testing robust defense mechanisms ensures the system maintains both **model accuracy** and **data privacy** under realistic threat scenarios.

### Project Structure
```
Federated-Learning/
├── app/
│   ├── models/
│   │   ├── malicious_entity.py  # Class for malicious entity
│   │   ├── client.py            # Class for federated client
│   │   ├── malicious_client.py  # Class for malicious federated client
│   │   ├── dataloader.py        # Dataloader class for data splitting and batchs 
│   │   ├── model.py             # ML models creation
│   │   ├── server.py            # Class for centralized server
│   │   └── attacked_server.py   # Class for attacked centralized server
│   └── simulation.py            # File with simulations
├── config/
│   ├── logger.py                # Configuration for a logger
│   └── settings.py              # Project configuration
├── data/
│   ├── input/                   # Input dataset in CSV format
│   ├── processed/               # Preprocessed data
│   │   ├── train/
│   │   │   └── building_*.pt
│   │   ├── val/
│   │   │   └── building_*.pt
│   │   └── test/
│   │       └── building_*.pt
│   └── preprocessing.py         # Methods for preprocessing raw data
├── logs/                        # Logs 
├── save/                        # Directory for saving server state and gobal model
├── README.md
├── requirements.txt             # Libraries requirement for the project
└── run.py                       # Entry point for simulations, checks and data preprocessing
```

**Key directories:**
- **app/**: Core implementation including ML models, client and server model
- **config/**: Configuration for the project and logger
- **data/**: Raw input datasets, preprocessing logic, and processed tensors organized by train/val/test splits
- **Root files**: Project configuration and entry points
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
6. Data sanitazing, to replace NaN values by 0
7. Data normalization
8. Time-based data splitting (`train`, `validation` and `test`)
9. Converting the object from a `pandas.DataFrame` to a `torch.Tensor`
10. Saving the `torch.Tensor` into the `data/processed` directory
<details>
<summary>See more about the preprocessing pipeline</summary>

#### Data extraction for each client
Each client represents a building. The preprocessing script selects the following columns :
```
date
weather related columns (temp, rhum, wdir, and wspd)
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
2. **Weather-related columns** (`temp`, `rhum`, `wspd`, `wdir`) - exogenous variables
3. **Energy-related columns** (`load`, `pv`, `net`) - target and related energy data

This ordering ensures that temporal context comes first, followed by environmental factors, and finally the energy consumption data to be modeled.

#### Data sanitizing
NaN values in the dataset are replaced with 0. This approach is well-suited for energy data because:

- **Missing energy measurements**: A NaN value for energy consumption or production is treated as zero energy, representing periods where no data was recorded or equipment was inactive
- **Weather variables**: Missing weather readings are set to 0, which serves as a neutral baseline that doesn't artificially bias the model toward extreme values
- **Consistency with domain logic**: In energy systems, missing data during low-activity periods is reasonably approximated as zero consumption/production rather than being omitted, which could create gaps in the time series and break temporal continuity

This strategy preserves the sequential nature of the data while handling missing values in a physically interpretable way.

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
## 📥 Installation and setup guide
### Installation part
In order to install and use the repository, you'll need `python 3.14.0`, you can check your python version by doing :
```
python --version
```
You can then clone the repository with :
```
git clone https://github.com/B4nanaJuice/Federated-Learning.git
```
Once the repository cloned, you can create your virtual environment and activate it with :
```
python -m venv venv
./venv/Scripts/activate
```
After enabling your virtual environment, you can install all required libraries :
```
pip install -r ./requirements.txt
```
### Setup part
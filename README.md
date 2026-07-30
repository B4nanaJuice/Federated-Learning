# Federated-Learning

Attack and defense testing of entities in a **Federated Learning** process applied to a **Smart Grid** context (prediction of electricity consumption, photovoltaic production, and net consumption).

This repository implements a complete federated learning simulation in PyTorch: baseline training (FedAvg), client-side *data poisoning* and *model poisoning* attacks, server-side attacks, robust defense mechanisms (Krum, Multi-Krum, Trimmed Mean, RFA, FLTrust, etc.), a scoring/decay mechanism for detecting malicious clients, and an "offline/degraded" training mode simulating clients with intermittent network connectivity.

## Table of contents

- [Context](#context)
- [Project architecture](#project-architecture)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Usage](#usage)
  - [Available commands](#available-commands)
  - [Examples](#examples)
- [Models](#models)
- [Implemented attacks](#implemented-attacks)
- [Implemented defenses](#implemented-defenses)
- [Scoring and trust decay](#scoring-and-trust-decay)
- [Degraded (offline) mode](#degraded-offline-mode)
- [Results and metrics](#results-and-metrics)
- [SLURM jobs](#slurm-jobs)
- [License](#license)

## Context

In a **Smart Grid** scenario, several buildings (clients) hold local data on electricity consumption (`load`), photovoltaic production (`pv`), and weather data. A central server coordinates the training of a global prediction model without raw data from each building ever leaving the client, in line with the federated learning principle.

This project studies the robustness of this process against malicious clients or servers (data poisoning, model poisoning, aggregation server attacks) and evaluates different defense and client trust-scoring strategies.

## Project architecture

```
Federated-Learning/
├── app/
│   ├── models/                  # ML models, client, base FedAvg server
│   │   ├── model.py              # NormalMLP, SoftGatedMoE
│   │   ├── client.py             # Federated client (local training)
│   │   ├── server.py             # Aggregation server (FedAvg)
│   │   ├── dataloader.py         # PyTorch dataset with lookback window
│   │   └── utils.py              # Utilities (early stopping, etc.)
│   ├── attacking_models/        # Malicious entities
│   │   ├── malicious_entity.py   # Base mixin for malicious behavior
│   │   ├── malicious_client.py   # Client poisoning data/model
│   │   └── attacked_server.py    # Server broadcasting a poisoned model
│   ├── scoring/                 # Robust defenses and trust scoring
│   │   ├── scoring_entity.py     # Scoring metrics (distance, dataset, ...)
│   │   ├── scoring_client.py     # Self-assessing client
│   │   ├── scoring_server.py     # Server evaluating clients
│   │   └── defense_server.py     # Krum, Multi-Krum, Trimmed Mean, RFA, FLTrust, CLRA, ...
│   ├── degraded/                 # Degraded network / offline client simulation
│   │   ├── network.py
│   │   ├── network_interface.py
│   │   ├── offline_client.py
│   │   └── offline_server.py
│   └── services/
│       ├── simulation_service.py # Orchestration of all simulations
│       ├── aggregation_service.py# Aggregation algorithms (FedAvg, ...)
│       └── plot_service.py       # Results plotting
├── config/
│   ├── settings.py               # Global settings (device, hyperparameters, paths)
│   └── logger.py                 # Logging configuration
├── data/
│   └── preprocessing.py          # Cleaning, normalization, and data splitting
├── jobs/                         # Shell / SLURM scripts to launch simulations on an HPC cluster
│   ├── attacks/                  # Data/model/server poisoning attack jobs
│   ├── defense/                  # Defense mechanism jobs
│   ├── scoring/                  # Client/server scoring jobs
│   ├── sigma/                    # Trust decay measurement jobs (sigma decay)
│   ├── offline/                  # Degraded-mode training job
│   └── utils/                    # Utility jobs (preprocessing, data grouping)
├── requirements.txt              # Python dependencies
├── simulation results.zip        # JSON files containing training loss, test MAE/MSE/RMSE for simulations
└── run.py                        # Command-line entry point
```

## Installation

**Requirements**: Python 3.11+ recommended, `pip`, and ideally a CUDA-compatible GPU to speed up training (the `cuda`/`cpu` device is selected automatically).

```bash
git clone https://github.com/B4nanaJuice/Federated-Learning.git
cd Federated-Learning

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> `requirements.txt` references the PyTorch CUDA 12.6 index (`--find-links https://download.pytorch.org/whl/cu126`). If you don't have a compatible GPU, install a CPU version of PyTorch/torchvision matching your setup before installing the rest of the dependencies.

## Data preparation

The project expects an input file `Final_Energy_Dataset_with_weather.csv` located in `data/input/` (configurable path in `config/settings.py` via `INPUT_DATA_PATH` and `INPUT_DATA_FILENAME`). This file must contain, for each building `X`, columns `load_X` (consumption) and `pv_X` (photovoltaic production), as well as weather columns (`temp`, `rhum`, `wspd`, `wdir`) and a `date` column.

The preprocessing step (`data/preprocessing.py`) performs, per building:

1. Selection of relevant columns (`load`, `pv`, weather, `date`).
2. Extraction of cyclical time features (`weekday`, `tod_sin`, `tod_cos`).
3. Computation of net consumption (`net = load - pv`).
4. Min-max normalization of energy variables and standardization (z-score) of weather variables.
5. Replacement of missing values with `0`.
6. Chronological split into training / validation / test sets (70% / 20% / 10%).
7. Conversion to PyTorch tensors and saving (`data/processed/`).

To run preprocessing (IID or non-IID data depending on the desired distribution across clients):

```bash
python run.py preprocess iid
python run.py preprocess noniid
```

## Usage

All simulations are launched via `run.py`:

```bash
python run.py <command> [--option value ...]
```

Without arguments, the list of available commands is displayed:

```
Usage: python run.py <command>
Available commands: preprocess, check, run-simulation, test, group-data, show-results
```

### Available commands

| Command | Purpose |
|---|---|
| `preprocess` | Preprocesses raw data (`iid` / `noniid`). |
| `baseline` | Baseline FedAvg simulation, no attack. |
| `data --malicious <0-100>` | Simulation with data poisoning by a percentage of malicious clients. |
| `model --malicious <0-100> --attack <method>` | Simulation with model poisoning. |
| `server --partial <true/false>` | Simulation of an attacked aggregation server (partial or total attack). |
| `long-server --partial <true/false>` | Extended variant of the server attack. |
| `client-scoring --metric <metric>` | Simulation with client-side trust scoring. |
| `server-scoring --metric <metric>` | Simulation with server-side trust scoring. |
| `client-decay --metric <metric>` | Client-side scoring with temporal trust decay. |
| `server-decay --metric <metric>` | Server-side scoring with temporal trust decay. |
| `client-defense --defense <method>` | Applies a robust client-side defense mechanism against an attacked server. |
| `server-defense --defense <method> --malicious <0-100>` | Applies a robust server-side defense mechanism against malicious clients. |
| `run-decay` | Measures trust score decay (sigma decay). |
| `offline --metric <metric>` | Simulation in degraded mode (clients with intermittent connectivity). |
| `group-data --save-filename <name>` | Aggregates several result runs (JSON) into a single file for analysis. |

Model poisoning methods (`--attack`): `gaussian_noise`, `gaussian_weights`, `uniform_noise`, `uniform_weights`, `sign_flip`, `gradient_amplification`.

Defense methods (`--defense`): `fedavg`, `krum`, `mkrum`, `norm`, `cbaa`, `tmean`, `rfa`, `fltrust`, `clra`.

Scoring metrics (`--metric`): `distance`, `dataset` (see `ScoringMetric` in `app/scoring/scoring_entity.py`).

### Examples

```bash
# Baseline simulation (no attack)
python run.py baseline

# 20% of clients poisoning their data
python run.py data --malicious 20

# 20% of clients poisoning their model with gradient sign flipping
python run.py model --malicious 20 --attack sign_flip

# Full attack on the aggregation server
python run.py server --partial false

# Multi-Krum defense against 30% malicious clients
python run.py server-defense --defense mkrum --malicious 30

# Trimmed Mean defense on the client side against an attacked server
python run.py client-defense --defense tmean

# Training over a degraded network, distance-based scoring
python run.py offline --metric distance

# Group all "20_data_*" runs into a single results file
python run.py group-data --save-filename 20_data
```

Each simulation runs 10 independent runs of 20 federated rounds by default (20 clients, 50% selected each round), and saves the metrics (training loss, MAE, MSE, RMSE on `load`, `pv`, and `net`) in JSON format in the `save/` folder (`SAVE_DATA_PATH`).

## Models

Two neural network architectures are available (`app/models/model.py`), taking as input a time window of `LOOKBACK` time steps (48 by default, i.e. 24h at a 30-minute step) and `NUM_FEATURES` variables (day of week, time of day encoded as sine/cosine, temperature, humidity, wind speed, and wind direction):

- **`NormalMLP`**: a simple multilayer perceptron (Linear → ReLU → Linear → ReLU → Dropout → Linear), used by default in most simulations.
- **`SoftGatedMoE`**: a soft-gated Mixture of Experts, with several expert sub-networks and a shared network combining their weighted outputs.

Both models produce a 3-dimensional output (`load`, `pv`, `net`).

## Implemented attacks

- **Data poisoning (`data`)**: malicious clients corrupt their training batches (Gaussian noise, replacement with Gaussian/uniform noise, etc.).
- **Model poisoning (`model`)**: malicious clients alter their local model weights before sending them to the server (Gaussian/uniform noise on weights, sign flipping, gradient amplification).
- **Aggregation server attack (`server` / `long-server`)**: the server itself broadcasts a poisoned global model to clients, either partially (a subset of rounds) or fully.

## Implemented defenses

The `app/scoring/defense_server.py` module provides several robust aggregation server variants (and their "attacked" counterparts for testing their resilience):

- **Weighted FedAvg** (`WeightedFedAvgServer`)
- **Krum** / **Multi-Krum** (`KrumServer`, `MKrumServer`)
- **Norm-bounding aggregation** (`NormAggServer`)
- **Certified Byzantine-robust Aggregation** (`CBAAFedAvgServer`)
- **Trimmed Mean** (`TMeanServer`)
- **Robust Federated Averaging (RFA)** (`RFAServer`)
- **FLTrust** (`FLTrustServer`)
- **CLRA** (`CLRAServer`)

## Scoring and trust decay

The `app/scoring/` module implements self-assessment and client rating mechanisms (`ScoringClient`, `ScoringServer`) based on different metrics (distance of updates, comparison against a reference dataset, etc.), with a trust threshold (`threshold`) below which a client is excluded from aggregation. A "decay" variant makes this trust score decrease over time to detect intermittent malicious behavior.

## Degraded (offline) mode

The `app/degraded/` module simulates a federated network with intermittent connectivity: a `Network` object manages client availability (`OfflineClient`), and an `OfflineServer` adapts the aggregation process accordingly, combined with trust scoring to keep detecting malicious clients despite disconnections.

## Results and metrics

After each run, the following metrics are computed on the test set and saved as JSON in `save/`:

- **`training_loss`**: training loss (MSE) per round and per client.
- **`MAE`, `MSE`, `RMSE`**: prediction errors for `load`, `pv`, and `net` on the test set.

The `group-data` command aggregates several runs sharing the same filename prefix into a single consolidated file (`save/grouping/<name>.json`), making it easier to compute means/standard deviations across multiple repetitions. The `app/services/plot_service.py` service can then be used to visualize these results.

## SLURM jobs

The `jobs/` folder contains shell scripts (some in `#SBATCH` format, intended for an HPC compute cluster) that automate the launch of simulation campaigns: increasing percentages of malicious clients, various attack and defense methods, trust decay measurements, etc. These scripts are provided as an example/reference for reproducing the experiments on a SLURM cluster and contain paths specific to the original environment (account, `spack` modules, virtualenv path) that should be adapted to your own infrastructure.

## License

This project is distributed under the [MIT](LICENSE) license.

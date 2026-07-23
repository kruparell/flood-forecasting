# Flood Forecasting Tutorials & Baselines

This directory contains configuration files, basin lists, trained model runs, evaluation notebooks, and baseline documentation for Google Hydrology flood forecasting models.

## Baseline Reference Document
See [`BASELINES.md`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/BASELINES.md) for full performance metrics, configuration settings, and metric distributions of the official baseline models:

- **Official Primary Baseline:** **100-Basin Generic Mean Embedding LSTM (Epoch 11)**
  - **Loss:** `0.1069`
  - **Median NSE:** `0.5626` (NeuralHydrology Framework Median NSE: `0.5791`)
  - **Median KGE:** `0.3450`
  - **Weights Checkpoint:** [`tutorial/model-runs/generic-meanembedding-100basin_2207_144648/continue_training_from_epoch003/model_epoch011.pt`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/model-runs/generic-meanembedding-100basin_2207_144648/continue_training_from_epoch003/model_epoch011.pt)
  - **Config File:** [`tutorial/configs/train-100basin-generic.yml`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/configs/train-100basin-generic.yml)
  - **Basin List:** [`tutorial/basin-lists/100-basin-train.txt`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/basin-lists/100-basin-train.txt)

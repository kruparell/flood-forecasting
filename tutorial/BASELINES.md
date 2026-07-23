# Hydrological Baseline Models Reference

> [!IMPORTANT]
> **Official Baseline Specification**: Use the 100-basin extended-epoch generic mean-embedding model detailed below as the primary baseline for general-purpose multi-basin streamflow forecasting experiments across CAMELS catchments.

---

## 1. Primary Generic Baseline: 100-Basin Mean Embedding LSTM (Epoch 11)

- **Model Architecture:** `MeanEmbeddingForecastLSTM`
- **Basin Coverage:** 100 CAMELS catchments ([`tutorial/basin-lists/100-basin-train.txt`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/basin-lists/100-basin-train.txt))
- **Configuration File:** [`tutorial/configs/train-100basin-generic.yml`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/configs/train-100basin-generic.yml)
- **Run Directory:** [`tutorial/model-runs/generic-meanembedding-100basin_2207_144648/continue_training_from_epoch003`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/model-runs/generic-meanembedding-100basin_2207_144648/continue_training_from_epoch003)
- **Best Model Checkpoint:** [`model_epoch011.pt`](file:///usr/local/google/home/kruparell/flood-forecasting/tutorial/model-runs/generic-meanembedding-100basin_2207_144648/continue_training_from_epoch003/model_epoch011.pt)

### Training Loss Progression
- **Epoch 1:** Loss = `0.3596`
- **Epoch 3:** Loss = `0.2332`
- **Epoch 6:** Loss = `0.1555`
- **Epoch 11:** **Loss = `0.1069`**

### Benchmark Performance Metrics (Across 100 Test Catchments)

| Metric | Lower Quartile (LQ / 25th %) | Median (50th %) | Upper Quartile (UQ / 75th %) |
| :--- | :---: | :---: | :---: |
| **NSE (Nash-Sutcliffe Efficiency)** | **0.3324** | **0.5626** | **0.6764** |
| **KGE (Kling-Gupta Efficiency)** | **-0.2747** | **0.3450** | **0.5775** |

#### Standard NeuralHydrology Framework Metric Distribution:
- **Official 1D NSE Median:** **0.5791**
- **Official 1D NSE Lower Quartile (LQ):** **0.4372**
- **Official 1D NSE Upper Quartile (UQ):** **0.6990**

---

## 2. Model Settings & Hyperparameters

```yaml
model: mean_embedding_forecast_lstm
dataset: multimet
hidden_size: 32
seq_length: 365
lead_time: 7
predict_n_hindcast: 5
predict_last_n: 12
batch_size: 256
learning_rate: 0.01 (Epochs 1-5) -> 0.005 (Epochs 6-11)
```

---

## 3. Comparison with Legacy 50-Basin Model Runs

| Run Configuration | Training Catchments | Epochs | Median NSE | Median KGE |
| :--- | :---: | :---: | :---: | :---: |
| **Legacy 50-Basin Quick Run** | 50 | 2 | ~0.42 | ~0.21 |
| **100-Basin Extended Run (Baseline)** | **100** | **11** | **0.5626** | **0.3450** |

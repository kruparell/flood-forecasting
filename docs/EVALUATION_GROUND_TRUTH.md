# Hydrology ML Evaluation & Model Ground Truth (`EVALUATION_GROUND_TRUTH.md`)

> [!IMPORTANT]
> **To All AI Agents & Collaborators**:
> This document defines the canonical specifications for the **GoogleHydrology Baseline Foundation Model (`MeanEmbeddingForecastLSTM`)** and the **Data Assimilation (DA)** project.
> - **DO NOT** assume tensor dimensions or feature keys from prose.
> - **DO NOT** modify scaler normalization logic without passing the verification probe below.
> - Any code change where the unassimilated baseline score drops below the reference threshold is **strictly invalid**.

---

## 1. System Navigation & Directory Map

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Google3 (Piper Workspace)                                                       │
│ Path: /google/src/cloud/kruparell/googlehydrology_rebased/google3/             │
│  ├── third_party/py/googlehydrology/                                            │
│  │    ├── modelzoo/               # MeanEmbeddingForecastLSTM foundation model   │
│  │    ├── datasetzoo/             # Multimet & Zarr dataset loaders             │
│  │    ├── datautils/              # Scaler & normalization utilities            │
│  │    ├── evaluation/             # assimilation.py, batched_assimilation.py    │
│  │    └── pipelines/              # 1_data_assimilation/ (DA runners & Beam)    │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────────┐
│ CNS Distributed Storage (Large Datasets, Zarr Stores, Checkpoints)              │
│ Cell: /cns/jn-d/home/floods/hydro_model/                                        │
│  ├── datasets/external/Caravans_MultiMet/  # MultiMet Dynamic Met Zarr Archives │
│  ├── work/kruparell/Caravans_V2/           # Caravans V2 Timeseries & Attributes│
│  ├── work/kruparell/scalers/               # Precomputed Zarr Scaler Caches     │
│  └── work/kruparell/cross_validation/      # 5-fold CV splits & basin groups    │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────────┐
│ Cloudtop Local Environment                                                      │
│ Home: /usr/local/google/home/kruparell/                                         │
│  ├── openhydronets_next/          # Local research git repo & exploratory code  │
│  │    └── tutorial/notebooks/     # Interactive DA & diagnostic notebooks       │
│  └── .gemini/jetski/brain/        # Session transcripts, logs & scratch scripts │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Critical File Paths Quick-Reference

| Component | Path Location | System / Tier |
| :--- | :--- | :--- |
| **Data Assimilation Engine** | `google3/third_party/py/googlehydrology/evaluation/assimilation.py` | Google3 (Live Workspace) |
| **Batched Vectorized DA Engine** | `google3/third_party/py/googlehydrology/evaluation/batched_assimilation.py` | [DEPRECATED / PROHIBITED - USE tester.py] |
| **DA Parameter Selection Pipelines** | `google3/third_party/py/googlehydrology/pipelines/1_data_assimilation/` | Google3 (Live Workspace) |
| **Foundation Model Code** | `google3/third_party/py/googlehydrology/modelzoo/mean_embedding_forecast_lstm.py` | Google3 (Live Workspace) |
| **MultiMet Zarr Datasets** | `/cns/jn-d/home/floods/hydro_model/datasets/external/Caravans_MultiMet` | CNS (`jn-d`) |
| **Caravans V2 Base Directory** | `/cns/jn-d/home/floods/hydro_model/work/kruparell/Caravans_V2` | CNS (`jn-d`) |
| **Precomputed Scaler Caches** | `/cns/jn-d/home/floods/hydro_model/work/kruparell/scalers/` | CNS (`jn-d`) |
| **Interactive Tutorial Notebooks** | `/usr/local/google/home/kruparell/openhydronets_next/tutorial/notebooks/` | Cloudtop Local Disk |

---

## 2. Canonical Baseline Foundation Model Checkpoint

| Attribute | Specification |
| :--- | :--- |
| **Model Class** | `MeanEmbeddingForecastLSTM` (`googlehydrology.modelzoo.mean_embedding_forecast_lstm`) |
| **Checkpoint Path** | `/cns/jn-d/home/floods/hydro_model/work/kruparell/checkpoints/...` |
| **Training Config** | `configs/train/multimet_foundation_85ep.yml` |
| **Warmup Sequence Length** | `365` days (Full sequence) |
| **Forecast Lead Time Horizon** | `7` days (`lead_time: 0` to `lead_time: 6`) |

---

## 3. Data Assimilation (DA) Three Optimization Modes

Data Assimilation optimizes representations and states using recent observed streamflow over historical sub-windows:

### Mode 1: LSTM State Updating (`h_n`, `c_n`)
* **Targets**: `['c_n_forecast']`, `['c_n_hindcast']`, `['c_both']`, `['h_n']`, `['c_n']`.
* **Mechanism**: Directly optimizes the LSTM cell state $c_t$ and/or hidden state $h_t$ at the end of the historical assimilation window to correct accumulated internal hydrological storage errors.

### Mode 2: Embedding Vector Optimization (`e_dyn`, `e_stat`)
* **Targets**: `['e_dyn']` (temporal dynamic embeddings), `['e_stat']` (static catchment embeddings), `['embedded_both']`.
* **Mechanism**: Optimizes the intermediate latent embeddings produced by the multi-provider dynamic embedding networks or static feature network before passing into the recurrent backbone.

### Mode 3: Precipitation Vector Optimization (`precip`, `total_precipitation`)
* **Targets**: `['precip']`, `['total_precipitation']`, `['precipitation_forcing']`.
* **Mechanism**: Directly optimizes the input precipitation forcing vectors (`x_d['total_precipitation']` / `x_d_forecast`) within the assimilation window to correct for meteorological precipitation forecast/gauge bias.

### Standard DA Hyperparameter Grid
* `assimilation_window`: `[30, 90, 180, 365]` days (or `[1, 3, 5, 7, 10]` for short-window probing)
* `assimilation_lead_time`: `0` (or `0..6` for multi-lead testing)
* `learning_rate`: `[0.05, 0.10, 0.20]`
* `epochs` / `steps`: `[25, 50, 75, 100]`
* `loss`: `MSE` or Peak-Weighted MSE
* `optimizer`: `Adam` or `SGD`

### Canonical Assimilation YAML Configuration Keys (`AssimilationConfig`)

All YAML configs passed to `--assimilation_config` are strictly validated against `@property` attributes of `AssimilationConfig` (`googlehydrology.utils.assimilationconfig`). Any key not listed below will raise `ValueError: ['...'] are not recognized config keys`:

| Category | Canonical YAML Config Key | Type | Description / Notes |
| :--- | :--- | :---: | :--- |
| **Optimization** | `learning_rate` | `float` or `dict` | Base learning rate (e.g. `0.10`) or step schedule `{0: 0.1, 50: 0.01}`. |
| | `learning_rate_drop_factor` | `float` | Multiplier when decaying LR (default `0.9`). |
| | `learning_rate_epoch_drop` | `int` | Interval of epochs between LR drops (default `5`). |
| | `epochs` | `int` | Optimization iterations per date (e.g. `25`, `50`, `100`). |
| | `optimizer` | `str` | Name of optimizer (`Adam`, `SGD`). |
| | `loss` | `str` | Objective loss function (`MSE`, `RMSE`). |
| | `clip_gradient_norm` | `float` | Gradient clipping threshold (default `1.0`). |
| | `early_stopping_patience` | `int` | Steps of loss plateau before early stop (default `10`). |
| | `early_stopping_min_loss` | `float` | Minimum loss threshold (default `1e-5`). |
| | `early_stopping_min_lr` | `float` | Minimum LR before early stop (default `1e-5`). |
| **Window & Horizon** | `assimilation_window` | `int` | History assimilation window length in days (e.g. `30`, `90`, `180`, `365`). Alias: `assimilation_window_length`. |
| | `assimilation_lead_time` | `int` | Target forecast lead time (0 = same-day / lead 1). |
| | `history` | `int` | Context lag (typically `1`). |
| | `seq_length` | `int` | Model sequence length (typically `365`). |
| | `predict_last_n` | `int` | Steps to predict. |
| | `predict_n_hindcast` | `int` | Hindcast steps (default `5`). |
| | `use_per_step_updates` | `bool` | Sequential iterative daily updating (default `True`). |
| | `state_anchor` | `str` | Where the recurrent-state control variables (`c_0_*`, `h_0_*`) are injected. `window_start` (default) = the LSTM state entering the first timestep of the assimilation window (standard 4D-Var state update). `sequence_start` = the state at `t=0` of the input sequence (legacy; retained only to reproduce pre-anchor sweeps). No effect on embedding or precipitation targets. |
| **Targets & Components** | `assimilation_targets` | `list[str]` | Target representations to optimize (see Target Catalog below). |
| | `assimilation_components` | `dict` | Explicit per-component dict `{name: {weight: float, lr: float}}`. |
| **Regularization** | `regularization_weight` | `float` | Global / fallback L2 penalty towards unassimilated baseline. |
| | `static_embedding_regularization_weight` | `float` | Penalty for `static_embedding` (alias: `bg_stat_weight`, default `1e-6`). |
| | `hindcast_embedding_regularization_weight` | `float` | Penalty for `hindcast_embedding` (alias: `bg_dyn_weight`, default `0.01`). |
| | `forecast_embedding_regularization_weight` | `float` | Penalty for `forecast_embedding` (falls back to `bg_dyn_weight`). |
| | `recurrent_state_regularization_weight` | `float` | Penalty for LSTM cell/hidden states (alias: `bg_regularization_weight`). |
| | `regularization` | `list[str]` | Regularizer types list. |
| **Features & Model** | `target_variables` | `list[str]` | Target variable names (e.g. `['streamflow']` or `['total_precipitation']`). |
| | `target_loss_weights` | `list[float]` | Loss weighting per target variable. |
| | `model_dropout` | `bool` | Enable dropout during assimilation (default `False`). |
| | `timestep_dropout` | `float` | Timestep dropout rate (default `0.0`). |
| | `no_loss_frequencies` | `list[str]` | Frequencies to exclude from loss calculation. |
| | `precip_forcing_keys` | `list[str]` | Forcing feature keys for precipitation DA. |
| | `precip_min_clip` | `float` | Minimum clip value for precipitation updates (default `-3.0`). |

#### Supported Target Catalog & Aliases
- **Component Names**:
  - `'static_embedding'`: Static catchment physical attribute embedding.
  - `'hindcast_embedding'`: Historical multi-provider dynamic meteorological embedding.
  - `'forecast_embedding'`: Forecast meteorological embedding.
  - `'c_0_hindcast'`, `'h_0_hindcast'`: LSTM cell and hidden state at start of hindcast.
  - `'c_0_forecast'`, `'h_0_forecast'`: LSTM cell and hidden state at start of forecast.
  - `'total_precipitation'`: Input precipitation forcing vectors.
- **Predefined Aliases**:
  - `'embedded_both'` / `'both_embeddings'`: `['static_embedding', 'hindcast_embedding']`
  - `'embedded_all'` / `'all_embeddings'`: `['static_embedding', 'hindcast_embedding', 'forecast_embedding']`
  - `'embedded_dynamics'` / `'embedded_dyn'`: `['hindcast_embedding']`
  - `'embedded_statics'` / `'embedded_stat'`: `['static_embedding']`
  - `'c_both'`: `['c_0_hindcast', 'c_0_forecast']`
  - `'h_both'`: `['h_0_hindcast', 'h_0_forecast']`
  - `'precip'`: `['total_precipitation']`

---

## 4. Exact Tensor Dimensions & Feature Contracts

### A. Dynamic Meteorological Forcings (`x_dyn`)
MultiMet multi-provider meteorological forcings:

* **Tensor Shape**: `[batch_size, sequence_length (365), num_providers (4), num_features]`
* **Canonical Dynamic Inputs (11 active features across providers)**:
  - `hres`: `['hres_surface_net_solar_radiation', 'hres_surface_net_thermal_radiation', 'hres_surface_pressure', 'hres_temperature_2m', 'hres_total_precipitation']`
  - `graphcast`: `['graphcast_temperature_2m', 'graphcast_total_precipitation', 'graphcast_u_component_of_wind_10m', 'graphcast_v_component_of_wind_10m']`
  - `imerg`: `['imerg_precipitation']`
  - `cpc`: `['cpc_precipitation']`

### B. Static Catchment Attributes (`x_stat` - Exactly 84 Features)
The foundation model consumes **84 static catchment features** (`STATIC_ATTRS`):

* **Tensor Shape**: `[batch_size, num_static_features (84)]`
* **Full 84 Feature List**:
  ```python
  STATIC_ATTRS = [
      "p_mean",
      "pet_mean_ERA5_LAND",
      "aridity_ERA5_LAND",
      "frac_snow",
      "moisture_index_ERA5_LAND",
      "seasonality_ERA5_LAND",
      "high_prec_freq",
      "high_prec_dur",
      "low_prec_freq",
      "low_prec_dur",
      "aet_mm_syr",
      "ari_ix_sav",
      "crp_pc_sse",
      "ele_mt_sav",
      "ero_kh_sav",
      "for_pc_sse",
      "gdp_ud_ssu",
      "gla_pc_sse",
      "glc_pc_s01",
      "glc_pc_s02",
      "glc_pc_s03",
      "glc_pc_s04",
      "glc_pc_s06",
      "glc_pc_s07",
      "glc_pc_s08",
      "glc_pc_s09",
      "glc_pc_s10",
      "glc_pc_s11",
      "glc_pc_s12",
      "glc_pc_s13",
      "glc_pc_s14",
      "glc_pc_s15",
      "glc_pc_s16",
      "glc_pc_s17",
      "glc_pc_s18",
      "glc_pc_s19",
      "glc_pc_s20",
      "glc_pc_s21",
      "glc_pc_s22",
      "hft_ix_s09",
      "hft_ix_s93",
      "inu_pc_slt",
      "inu_pc_smn",
      "inu_pc_smx",
      "ire_pc_sse",
      "kar_pc_sse",
      "lka_pc_sse",
      "nli_ix_sav",
      "pac_pc_sse",
      "pet_mm_syr",
      "pnv_pc_s01",
      "pnv_pc_s02",
      "pnv_pc_s03",
      "pnv_pc_s04",
      "pnv_pc_s05",
      "pnv_pc_s06",
      "pnv_pc_s07",
      "pnv_pc_s08",
      "pnv_pc_s09",
      "pnv_pc_s10",
      "pnv_pc_s11",
      "pnv_pc_s12",
      "pnv_pc_s13",
      "pnv_pc_s14",
      "pnv_pc_s15",
      "ppd_pk_sav",
      "pre_mm_syr",
      "prm_pc_sse",
      "rdd_mk_sav",
      "snw_pc_syr",
      "swc_pc_syr",
      "tmp_dc_syr",
      "urb_pc_sse",
      "wet_pc_s01",
      "wet_pc_s02",
      "wet_pc_s03",
      "wet_pc_s04",
      "wet_pc_s05",
      "wet_pc_s06",
      "wet_pc_s07",
      "wet_pc_s08",
      "wet_pc_s09",
      "wet_pc_sg1",
      "wet_pc_sg2",
  ]
  ```

### C. Streamflow Target (`y_obs`)
* **Tensor Shape**: `[batch_size, lead_time_horizon (7)]`
* **Target Variables**: `['streamflow']` in mm/day

---

## 5. Data Normalization & Scaler Ground Truth

1. **Scaler Cache Location**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/scalers/`
2. **Critical Rule**: During inference/evaluation in `ZarrDatasetReader` or `assimilation.py`, the reader **must** load the pre-computed scaler from CNS.
3. **Prohibited**: Dynamically calculating mean/std on the evaluation test period is strictly forbidden.

---

## 6. Benchmark Baseline "North Star" Metrics

Before evaluating Data Assimilation (DA) updates, the unassimilated baseline model **must** match these verified scores:

### A. Canonical Single-Basin Benchmark (`hysets_12101500`)
* **Evaluation Period**: 2017-01-01 to 2019-12-31 (3-Year Continuous)
* **Expected Unassimilated Baseline NSE**: `0.7025` $\pm 0.01$
* **Expected Post-DA NSE**: $\ge 0.78$ (Significant improvement on lead times 0–2 across all three DA modes)

### B. Multi-Basin Benchmark (50 Caravan Basins)
* **Expected Unassimilated Median NSE**: `0.7180`
* **Lead-Time Monotonicity Constraint**:
  $$\text{NSE}(L_0) \ge \text{NSE}(L_1) \ge \dots \ge \text{NSE}(L_6)$$
  *(Lead-0 predictions must always outperform or match Lead-1 predictions).*

---

## 7. Mandatory Local Verification Unit Tests

Run these unit tests locally **before** launching cluster experiments or submitting code:

```bash
# 1. Test Sequential Assimilation Engine
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test //third_party/py/googlehydrology:assimilation_test

# 2. Test Vectorized Batched Assimilation Engine (State, Embedding, and Precip Modes)
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test //third_party/py/googlehydrology:batched_assimilation_test
```

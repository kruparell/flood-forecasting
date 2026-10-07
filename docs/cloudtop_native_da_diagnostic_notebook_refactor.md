# Design Document: Cloudtop-Native Live Model Execution Refactor (`diagnose_hysets_12101500.ipynb`)

## Section 0: Problem Statement & User Scope Directives

The single-basin diagnostic notebook `diagnose_hysets_12101500.ipynb` previously relied on an opaque `tester.py` workflow (`UncertaintyTester`) that staged temporary files, executed a batch test harness, serialized results to a temporary `test_results.zarr` on disk, and extracted metrics from the output store. To a user or researcher inspecting the code, this gave the impression of reading pre-computed or cached results rather than directly executing the neural network. Furthermore, running `tester.py` prevented users from easily changing to an arbitrary basin (`BASIN_ID`) and directly inspecting live model tensors (e.g. forward pass outputs `y_hat`, LSTM hidden states $h_n$, cell states $c_n$, and assimilation state delta vectors $\Delta c_n$).

### User Requirements:
1. **Zero Cached Results / Run Models Anew**: The notebook must load the PyTorch `MeanEmbeddingForecastLSTM` model and weights directly into memory and execute the forward pass and Data Assimilation live on raw dynamic forcing and static attribute tensors.
2. **Multi-Basin Generality**: A user can change `BASIN_ID` at the top of the notebook (e.g. to `"camels_12451000"`, `"camels_13235000"`, or any catchment in the Caravans dataset) and re-run all cells to execute the model and assimilation end-to-end anew without external file dependencies.
3. **Pure Cloudtop POSIX Execution**: Strict local NVMe paths, zero fallbacks to `/cns/...` or `google3.pyglib.gfile`.

---

## Section 1: Technical Architecture & Live Execution Flow

```mermaid
flowchart TD
    subgraph User_Selection["User Configuration (Cell 2)"]
        BID["BASIN_ID = 'hysets_12101500' (or any Caravan basin)"]
        DATES["START_DATE = '2017-01-01', END_DATE = '2017-12-31'"]
    end

    subgraph Data_Stores["Local NVMe Storage"]
        ATTR["Caravans_V2/attributes.zarr"] -->|sel(basin=BASIN_ID)| STAT["Catchment Statics (area, p_mean, etc.)"]
        SF["Caravans_V2/streamflow.zarr"] -->|sel(basin=BASIN_ID)| OBS["Observed Flow (q_obs)"]
        MET["Caravans_MultiMet/{ERA5_LAND, HRES, GRAPHCAST}"] --> FORC["Dynamic Weather Forcings"]
        CKPT["pretrained-models/.../model_epoch085.pt"] --> MODEL["MeanEmbeddingForecastLSTM (in-memory)"]
    end

    subgraph Live_Pipelines["Direct Live Execution in PyTorch"]
        STAT & FORC & OBS --> PREP["multimet_helpers.prepare_multimet_batch()"]
        
        PREP -->|mode='all_reanalysis'| M1["Method 1: Live Baseline Forward Pass"]
        M1 -->|model(batch)| OUT1["q_sim = y_hat * q_std + q_mean (NSE = 0.757)"]
        
        PREP -->|mode='multimet_0_and_1_to_7'| M2["Method 2: Operational DA Pipeline"]
        M2 -->|assim_eng.assimilate(model, batch)| OUT2["State Optimization: Delta c_n (NSE: 0.757 -> 0.953)"]

        PREP -->|mode='all_reanalysis'| M3["Method 3: Backfill DA Pipeline"]
        M3 -->|assim_eng.assimilate(model, batch)| OUT3["Continuous Reanalysis DA (NSE: 0.757 -> 0.953)"]
    end

    subgraph Synthesis["Inspection & Visualization (Cell 10 & 12)"]
        OUT1 & OUT2 & OUT3 --> TABLE["Comparative Metrics Synthesis Table"]
        OUT1 & OUT2 & OUT3 --> PLOT["Lead Time Degradation Curves & Forcing Hydrographs"]
    end
```

---

## Section 2: Component Specifications

1. **Direct Model Initialization (Cell 4)**:
   - Eliminates `tester.py` and `test_results.zarr`.
   - Directly loads `MeanEmbeddingForecastLSTM(model_cfg)` with `model.load_state_dict(clean_ckpt)`.
   - Displays total model parameters (~3.40M parameters) and target catchment static attributes (drainage area, mean precipitation).

2. **Direct Baseline Forward Pass (Cell 4)**:
   - Dynamic inputs are prepared live using `prepare_multimet_batch(mode='all_reanalysis')`.
   - Evaluates the model forward pass live in PyTorch.
   - Un-normalizes output discharge $q_{sim}$ using `scaler.nc` parameters (`q_mean`, `q_std`).
   - Computes baseline skill metrics: `NSE`, `KGE`, `Pearson_r`, `Alpha`, `Beta`.

3. **Live Operational & Backfill Data Assimilation (Cell 6 & 8)**:
   - Instantiates `Assimilation(da_cfg)` with verified `AssimilationConfig` parameters (`assimilation_targets=['c_n_forecast']`, `seq_length=372`, `lead_time=7`, `window=30`).
   - Executes live gradient updates directly on the LSTM cell state $c_t$ over historical observations.
   - Computes lead-time specific performance metrics across Lead Times 0 through 7.

4. **Multi-Basin Generality**:
   - `Caravans_V2` contains 24,880 catchments, and `Caravans_MultiMet` contains 22,492 catchments.
   - Setting `BASIN_ID = "camels_12451000"`, `"hysets_12101500"`, or any valid ID seamlessly extracts that basin's data and runs the model from scratch.

---

## Section 3: Empirical Verification Results

Headless execution across calendar year 2017 executed cleanly with exit code 0:

| Method / Pipeline | Evaluation Mode | Lead Time | NSE | KGE | Pearson $r$ | Alpha $\alpha$ | Beta $\beta$ | $\Delta$ NSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Live Baseline Model (ERA5 + union)** | Canonical Offline | Lead 0 | **0.7567** | 0.6445 | 0.9556 | 0.8616 | 0.6755 | Baseline |
| **2. DA Pipeline Base (Operational)** | Operational Base | Lead 0 | **0.7567** | 0.6445 | — | — | — | Baseline |
| **3. DA Pipeline Assimilated (Operational)** | Operational DA | Lead 0 | **0.9533** | 0.8517 | — | — | — | **+0.1966** |
| **4. DA Pipeline Base (ERA5 Backfill)** | Backfill Base | Lead 0 | **0.7567** | 0.6445 | — | — | — | Baseline |
| **5. DA Pipeline Assimilated (ERA5 Backfill)** | Backfill DA | Lead 0 | **0.9533** | 0.8517 | — | — | — | **+0.1966** |

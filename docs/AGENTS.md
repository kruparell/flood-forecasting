# GoogleHydrology & Data Assimilation Agent Guidelines

## 0. BANNED FILES LIST (STRICT PROHIBITION)
The following files are **STRICTLY BANNED** and must NEVER be imported, executed, referenced, or used in any script, pipeline, test, or XManager job:
- `third_party/py/googlehydrology/evaluation/batched_assimilation.py` (`batch_assimilation.py`)
- `third_party/py/googlehydrology/evaluation/redundant/batched_assimilation.py`
- `third_party/py/googlehydrology/pipelines/1_data_assimilation/batched_da_param_selection.py`
- `third_party/py/googlehydrology/pipelines/1_data_assimilation/beam_da_param_selection.py`

> [!CAUTION] **ALL EVALUATION & DATA ASSIMILATION MUST RUN STRICTLY VIA `tester.py`**
> Always run evaluation and Data Assimilation through `googlehydrology/evaluation/tester.py` (`UncertaintyTester` / `RegressionTester` -> `Assimilation` in `assimilation.py`), invoked via `start_evaluation()` and `scripts/run_eval_borg.py`.

## 1. Mandatory Single Source of Truth
Before proposing, modifying, or auditing any code related to hydrological models, data loaders, scalers, or Data Assimilation (DA) pipelines, you MUST read and strictly comply with:
- `google3/third_party/py/googlehydrology/EVALUATION_GROUND_TRUTH.md` (Ground truth contracts & baselines)
- `google3/third_party/py/googlehydrology/XMANAGER_EXPERIMENT_HISTORY.md` (Historical XManager runs, XIDs, and cluster diagnoses)

## 2. Core Invariants & Contracts
- **Foundation Model**: `MeanEmbeddingForecastLSTM` (multi-provider dynamic embeddings + static embedding backbone).
- **Three DA Optimization Modes**:
  1. State Updating: LSTM hidden/cell states (`h_n`, `c_n`).
  2. Embedding Optimization: Latent dynamic/static embeddings (`e_dyn`, `e_stat`).
  3. Precipitation Optimization: Input precipitation vectors (`total_precipitation`).
- **Feature Contracts**:
  - Dynamic Inputs: 11 active features across ERA5, HRES, CPC, IMERG.
  - Static Attributes: Exactly 84 features (`STATIC_ATTRS`).
- **Normalization Ground Truth**:
  - Always use pre-fitted training scalers from `/cns/jn-d/home/floods/hydro_model/work/kruparell/scalers/`.
  - Re-fitting scalers dynamically on evaluation periods is strictly prohibited.
- **Reference Benchmark & Frozen Ground Truth Files (STRICTLY READ-ONLY & UNEDITABLE)**:
  - Single-basin unassimilated baseline NSE on `hysets_12101500` is `0.7025` (3-year continuous 2017–2019).
  - Any code change where the unassimilated baseline NSE drops below `0.65` is invalid.
  - **IMMUTABLE GROUND TRUTH BENCHMARKS**: All files under `pretrained-models/**` are permanently frozen:
    - `pretrained-models/google-floodhub-settings-55-epochs/test/test_metrics.csv` (10,137 global benchmark basins across Caravans)
    - `pretrained-models/google-floodhub-settings-55-epochs/test/model_epoch055/test_metrics.csv` (616 CAMELS benchmark basins)
    - Foundation weights `model_epoch055.pt`, `optimizer_state_epoch055.pt`, and `scaler.nc`
  - **STRICT PROHIBITION**: Agents, tools, and evaluation workflows must **NEVER** edit, overwrite, truncate, or direct evaluation outputs (`--run_dir`) to `pretrained-models/`.
  - All test and evaluation outputs must strictly be written to external CNS directories (`/cns/.../large_scale_param_selection_results/`) or temporary run folders (`/tmp/...`), never inside `pretrained-models/`.
- **Archived Experimental Directories (INACTIVE & OUT OF SCOPE)**:
  - All files under `3_arlstm_and_extra_da/` (including `arlstm.py` and AR-LSTM notebooks) are frozen experimental archives.
  - Agents must **NOT** import, edit, execute, or assume these modules are active in the core library.
- **Repository Architecture & 4-Pillar Layout**:
  - `1_base/`: Reference snapshot of the upstream PR 272 base containing ONLY the files modified for DA.
  - `2_pushed_standalone_da/`: Reference snapshot of the standalone DA PR containing ONLY the modified/added files.
  - `3_arlstm_and_extra_da/`: Isolated archive of experimental AR-LSTM code and reference notebooks (inactive).
  - `4_working_analysis/`: Active research diagnostics, hydrograph evaluation, Zenodo comparisons, and staging data.
  - **Active Package Authority**: The root `googlehydrology/` package contains the working standalone DA engine. All evaluations in `4_working_analysis/` import directly from `googlehydrology`.




## 3. Mandatory Verification
Always run local unit tests before proposing or committing changes:
- `/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test //third_party/py/googlehydrology:assimilation_test`

> The `batched_assimilation_test` target has been **deleted**. It was the only
> build target covering the banned `batched_assimilation.py` code path (see §0),
> which is now excluded from the `py_library` glob. Do not re-add it.

## 4. Data Assimilation (DA) Run Standards & Directory Registry

### A. Canonical CNS Base Directory
All large-scale parameter selection and DA evaluation outputs MUST be stored under:
`/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/`

### B. Standardized Experiment Naming Convention
All XManager experiment names and CNS subdirectories MUST strictly follow this standard format:
`da_<sweep_type>_st<stage>_<altered_params>_<YYYYMMDD>`
- `<sweep_type>`: `embeddings`, `cell_state`, or `precip`
- `st<stage>`: `st1`, `st2`, `st3`, `st4`, `st5`, `st6`
- `<altered_params>`: Standardized short codes indicating the swept hyperparameters:
  - `w`: Window length ($w$)
  - `lr`: Learning rate
  - `dec`: Decay factor
  - `ep`: Epoch count
  - `bgdyn`: Dynamic regularization weight
  - `bgstat`: Static regularization weight
  - `tgt`: Target state / embedding
- `<YYYYMMDD>`: Execution date in 8-digit ISO format (e.g. `20260831`)

**Examples**:
- `da_embeddings_st3_bgstat_bgdyn_20260831` (Phase 1 diagnostic probe)
- `da_embeddings_st4_w_lr_dec_ep_bgdyn_20260831` (Phase 2 optimization sweep)
- `da_cell_state_st6_tgt_w_bg_lr_20260829` (Stage 6 cell state sweep)

### C. Active & Historical Benchmark Runs Index
| Stage / Sweep Type | Experiment Name | XID | Configs | CNS Subdirectory | Focus / Benchmark Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Embeddings Stage 2** | `da_embeddings_stage2_78cfgs` | `xid/284712352` | 78 | `.../da_embeddings_stage2_78cfgs` | Full grid comparing static vs. decoupled dynamic embeddings ($w=60,90,180$). Global best: $t+1$ NSE 0.510. |
| **Cell State Stage 6** | `da_cell_state_stage6_substantial` | `xid/284715583` | 66 | `.../da_cell_state_stage6_substantial` | Comprehensive Cell State DA sweep across Grids A–D ($w=3,7,14$). |
| **Embeddings Stage 3** | `da_embeddings_stage3_dyn_probe` | `xid/285174329` | 6 | `.../da_embeddings_stage3_dyn_probe` | Phase 1 Diagnostic probe testing $bg_{stat} \in \{10^{-3}, 10^{-6}\}$ and $bg_{dyn} \in \{0.1, 0.01, 10^{-4}\}$. |
| **Embeddings Stage 4** | `da_embeddings_st4_w_lr_dec_ep_bgdyn_20260831` | `xid/285185982` | 32 | `.../da_embeddings_st4_w_lr_dec_ep_bgdyn_20260831` | Phase 2 Multi-window ($w=180,365$) & annealing ($\text{LR}=0.1,0.3$, decay $0.7,0.9$, ep $100,150$) grid across 25 shards. |
| **Embeddings Stage 5** | `da_embeddings_st5_snap_6k_20260901` | `xid/285498157` | 10 | `.../da_embeddings_st5_snap_6k_20260901` | Snapshot checkpoint sweep ($w=365$, LR 0.01–0.5, Ep 10/50/100) across 80 shards (6k basins). |
| **Embeddings Stage 8** | `da_embeddings_st8_tgt_bgdyn_50basins_20260902` | `xid/285840827` | 16 | `.../da_embeddings_st8_tgt_bgdyn_50basins_20260902` | 3 vs 2 embeddings comparison (`embedded_all` vs `embedded_both`, $bg_{dyn} \in \{0.01, 0.1\}$) across 50 clean basins. |
| **Embeddings Stage 9** | `da_embeddings_st9_tgt_bgdyn_50basins_20260902` | `xid/285909206` | 16 | `.../da_embeddings_st9_tgt_bgdyn_50basins_20260902` | 3 vs 2 embeddings comparison (`embedded_all` vs `embedded_both`, $bg_{dyn} \in \{0.01, 10^{-4}\}$) across 50 clean basins (50 shards, 1 basin/shard). |
| **Embeddings Stage 10** | `da_embeddings_st10_all_bgdyn_500basins_20260902` | `xid/285922665` | 16 | `.../da_embeddings_st10_all_bgdyn_500basins_20260902` | 500-basin regularization sweep for `embedded_all` (3 embeddings) testing $bg_{dyn} \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ across 50 shards (10 basins/shard). |
| **Embeddings Stage 11 (Filtered)** | `da_embeddings_st11_both_all_filtered_6k_20260903` | `xid/286159780` | 288 | `.../da_embeddings_st11_both_all_filtered_6k_20260903` | 6k-basin comprehensive grid comparing `embedded_both` vs `embedded_all` across $w \in \{30,180,365\}$, $\text{LR} \in \{0.01,0.1,0.5\}$, $bg_{dyn} \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ on filtered baseline model (80 shards). |
| **Embeddings Stage 11 (Unfiltered)** | `da_embeddings_st11_both_all_unfiltered_6k_20260903` | `xid/286160507` | 288 | `.../da_embeddings_st11_both_all_unfiltered_6k_20260903` | 6k-basin comprehensive grid comparing `embedded_both` vs `embedded_all` across $w \in \{30,180,365\}$, $\text{LR} \in \{0.01,0.1,0.5\}$, $bg_{dyn} \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ on unfiltered baseline model (80 shards). |
| **Embeddings Stage 11 (Filtered 2017–2020)** | `da_embeddings_st11_both_all_filtered_6k_2017_2020_20260908` | `xid/287639943` | 288 | `.../da_embeddings_st11_both_all_filtered_6k_2017_2020_20260908` | 6k-basin 4-year continuous benchmark (2017-01-01 to 2020-12-31) comparing `embedded_both` vs `embedded_all` with `use_union_mapping=False` across 200 shards. |
| **Embeddings Stage 16** | `da_embeddings_st16_80cfgs_100b_2017_20260916` | `xid/290040560` | 80 | `.../da_embeddings_st16_80cfgs_100b_2017_20260916` | 100-basin 80-shard rapid optimization sweep with 7-day forecast gap fix in place (`assimilation_lead_time=7`). |
| **Embeddings Stage 17** | `da_embeddings_st17_10b_2017_fix_20260916` | `xid/290079013` | 2 | `.../da_embeddings_st17_10b_2017_fix_20260916` | 10-basin 2017 verification of single-pass window-splicing DA fix (`baseline_0da` vs `both_w365_lr0.1_ep50_bg0.01_stat1e-06`). |
| **Embeddings Stage 17b** | `da_embeddings_st17b_10b_2017_fix_20260916` | `xid/290082742` | 2 | `.../da_embeddings_st17b_10b_2017_fix_20260916` | 10-basin 2017 verification rerun with cuDNN RNN backward fix (`baseline_0da` vs `both_w365_lr0.1_ep50_bg0.01_stat1e-06`). |
| **Embeddings Stage 20** | `da_embeddings_st20_20cfgs_64b_2017_20260917` | `xid/290384855` | 20 | `.../da_embeddings_st20_20cfgs_64b_2017_20260917` | 64-basin 20-config DA sweep (1 shard/config = 20 V100 Borg jobs; 1 baseline + 19 DA configs across `both`, `dyn`, `all`, `stat`, and reg probes) with canonical `shared_embeddings` fix restoring Lead 7 baseline NSE parity (`~0.50`). |
| **Embeddings Stage 22** | `da_embeddings_st22_20cfgs_4287b_2017_2018_20260918` | `xid/290616911` | 20 | `.../da_embeddings_st22_20cfgs_4287b_2017_2018_20260918` | 4287-basin (`filtered_basins_nse_gt_0.5.txt`) 2-year (2017-01-01..2018-12-31) 20-config sweep at 20 shards/config = 400 V100 jobs. First sweep to read a **pre-materialized eval subset** (`--subset_dir=.../eval_subsets/sweep4287_2017_2018`) instead of the full 22k-basin/75-year MultiMet archive, removing the ~37x Zarr read amplification that previously hung every work unit at `dask.compute(self._dataset)`. Baseline median NSE 0.6166 (Lead 1) / 0.4149 (Lead 7) / 0.5209 (mean Lead 1-7) over 2836 evaluable basins. |

### D. Results Analysis & Comparison Tool
To inspect, aggregate shard parquets, rank configurations, and compare metrics:
```bash
./blaze-bin/third_party/py/googlehydrology/pipelines/1_data_assimilation/analyze_da_results \
  --results_dir=/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/<experiment_name>
```

---

## 5. In-Flight XManager Job & Basin Progress Monitoring Protocol (MANDATORY)

Whenever queried about the status, progress, health, or latency of any active or completed Data Assimilation (DA) XManager experiment:

### A. Strict Anti-Pattern: Never Report Coarse Job Status Alone
- **STRICT PROHIBITION**: Do **NOT** merely report that an XManager experiment or Borg job is "RUNNING", "PENDING", or "COMPLETED".
- **Queued vs. Executing (`RUNNING` != Executing)**: A work unit reported `RUNNING` may still be queued awaiting progressive GPU admission. Always compute elapsed execution time from the job's own log timestamps, **never** from wall-clock-since-launch.
- **XManager Inspector Binary**: Use `/google/bin/releases/gemini-agents-xmanager/xmanager_tool` (`get_experiment --xid=...`). `/google/bin/releases/xmanager/xmanager_tool` does not exist.
- **State Projections Only From Direct Measurements**: Never extrapolate aggregate throughput from a single-stream measurement (degraded distributed storage is often latency-bound, where $N$ concurrent streams can each be faster than 1 lone stream). State an ETA only after measuring at actual concurrency, and explicitly cite the measurement it rests on.

### B. Basin Completion & Shard Progress on CNS (Mandatory Verification)
- **Actual `tester.py` CNS Output Layout**:
  `<exp>/<config_id>/shard_NN/test/model_epoch<NNN>/test_metrics_data_assimilation.csv` (or `test_metrics.csv` for open-loop `baseline_0da`).
  *(Note: Legacy `<exp>/shard_*/timeseries/*.parquet` paths do NOT exist under `tester.py` runs — never query them).*
- **`output.log` Is NOT Streamed**: `<exp>/<config_id>/shard_NN/output.log` is written to CNS **only upon shard completion**, so it is useless for monitoring in-flight progress.
- **How to Check Completed Shards & Incremental Basin Metrics**:
  1. **Completed Shards / Metrics CSVs**:
     ```bash
     fileutil ls "/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/<experiment_name>/*/shard_*/test/model_epoch*/test_metrics_data_assimilation.csv" | wc -l
     ```
  2. **Automated Shard & Basin Progress Script**:
     ```bash
     /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run //third_party/py/googlehydrology/pipelines/1_data_assimilation:check_xid_progress -- \
       --exp_dir=/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/<experiment_name>
     ```

### C. Live Worker Log Inspection (`google-analog`) & "Hung" Eval Diagnosis
1. **`google-analog` Raw Dump + Local Filtering**:
   - The binary is `/google/bin/releases/analog-cli/google-analog`.
   - Its server-side `--regex` filter frequently returns empty results on live jobs, and `--start_time` / `--end_time` are ignored.
   - Always scope with `--job_regex` (or it is very slow), dump raw logs to a local file (`--lookback` / `--max_results`), and filter locally with `grep`:
     ```bash
     /google/bin/releases/analog-cli/google-analog --logtostderr --minloglevel=2 --remote \
       --user=kruparell --job_regex='.*<XID>.*' --lookback=4h --max_results=500 > /tmp/analog_<XID>.log
     grep -E 'BASIN DA SUMMARY|Processing Basin|Evaluated' /tmp/analog_<XID>.log
     ```
2. **Diagnosing "Hung" Evals (`dask.compute(self._dataset)`)**:
   - Essentially all dataset I/O in GoogleHydrology evals happens at a single line: `dask.compute(self._dataset)` in `datasetzoo/multimet.py`. Everything before it is lazy and finishes in seconds, so an eval that appears "hung" is blocked there.
   - Surface canonical phase markers by setting config key `logging_level: DEBUG` rather than instrumenting/editing frozen canonical code.
   - Root cause is almost always **Zarr read amplification**: MultiMet Zarr stores hold the entire ~75-year date axis (27,333 days) in **one chunk**, so a 2-year eval still downloads all 27,333 days per basin block once per work unit. Fix by pre-materializing a subset (`--subset_dir`), never by editing canonical loaders.

---

## 6. Architecture & Refactoring Standards (STRICT INVARIANTS)

### A. True Component-Agnostic Solvers (No "Façade" Branching)
- **STRICT PROHIBITION**: Solver or evaluation loops (`batched_assimilation.py`, `assimilation.py`) MUST NOT contain internal `if mode == 'cell_state': ... elif mode == 'embeddings': ...` branches.
- Generic solvers must operate on generic parameter dictionaries (`opt_targets: Dict[str, torch.Tensor]`).
- Domain variable names (`c_0`, `h_0`, `static_embedding`, `x_d`) and regex patterns (`_PRECIP_PATTERN`) belong strictly in model definitions (`AssimilationTargetSpec`) or configs, NEVER inside solver optimization loops.

### B. Canonical Loss & Regularization Delegation
- **STRICT PROHIBITION**: Do NOT calculate loss formulas (`diff_sq`, `weighted_diff_sq`) or regularization penalties (`penalty_c_hc`, `penalty_stat`) inline inside solver classes.
- All evaluation solvers MUST delegate loss and regularization evaluation to canonical framework factories:
  ```python
  loss = self._loss_obj(pred_dict, batch_data) + self._regularization_obj(opt_targets, ref_targets)
  ```

---

## 7. Canonical Evaluation Pipeline & Multi-Leadtime Metric Contract

### A. Tester Pipeline as Sole & Canonical Evaluation Engine
- **Canonical Evaluation Authority**: The `Tester` pipeline (`googlehydrology/evaluation/tester.py`, invoked via `run.py evaluate --assimilate` or `scripts/run_eval_borg.py --assimilate`) is the **sole canonical authority** for all evaluation runs, benchmark comparisons, sensitivity analyses, and parity checks.
- **STRICT PROHIBITION OF BATCHED_ASSIMILATION**: `batched_assimilation.py` and custom solver loops are **STRICTLY PROHIBITED**. No script, tool, or agent is permitted to import, call, or execute `batched_assimilation.py`. All evaluations and Data Assimilation MUST execute strictly via `tester.py` (`RegressionTester` / `start_evaluation()` / `run.py evaluate --assimilate`).

### B. Multi-Leadtime Forecast Metric Contract
For daily multi-step forecast models (e.g. 7-day lead time horizons):
- **Unprefixed Metrics (`NSE`, `KGE`, `Alpha-NSE`, etc.)**: Represent **Lead 1** ($t+1$). This ensures strict backward compatibility with historical benchmarks and unassimilated baseline comparisons (e.g. in `test_metrics.csv`).
- **Per-Leadtime Metrics (`{metric}_lead{L}`)**: Emitted for each lead time $L \in [1 \dots \text{lead\_time}]$, e.g.:
  - `NSE_lead1`, `NSE_lead2`, `NSE_lead3`, `NSE_lead4`, `NSE_lead5`, `NSE_lead6`, `NSE_lead7`
  - `KGE_lead1` ... `KGE_lead7`
- **Forecast Horizon Mean (`{metric}_mean_lead1_{num_leads}`)**: Emitted as the unweighted arithmetic mean across all lead times in the horizon (e.g. `NSE_mean_lead1_7`).
- **Data Assimilation Outputs**:
  - Incremental metrics are saved to `{period}_metrics_data_assimilation.csv`.
  - Predictions and observations are saved to `{period}_results_data_assimilation.zarr` with metadata consolidated.

---

## 8. Data Assimilation Config Schema & Target Contracts (MANDATORY)
- **Allowed YAML Config Keys**: All YAML configurations passed to `--assimilation_config` are strictly validated against `@property` attributes on `AssimilationConfig` (`googlehydrology.utils.assimilationconfig.AssimilationConfig`). Unknown keys raise `ValueError: ['...'] are not recognized config keys`.
- **Allowed Keys Reference**: Consult `AssimilationConfig.get_allowed_keys()` or Section 3 of `EVALUATION_GROUND_TRUTH.md` for the exhaustive list of recognized keys:
  - Optimization: `learning_rate`, `learning_rate_drop_factor`, `learning_rate_epoch_drop`, `epochs`, `optimizer`, `loss`, `clip_gradient_norm`, `early_stopping_patience`, `early_stopping_min_loss`, `early_stopping_min_lr`.
  - Horizon & Window: `assimilation_window` (alias: `assimilation_window_length`), `assimilation_lead_time`, `history`, `seq_length`, `predict_last_n`, `predict_n_hindcast`, `use_per_step_updates`, `state_anchor`.
  - Targets & Regularization: `assimilation_targets`, `assimilation_components`, `regularization_weight`, `static_embedding_regularization_weight` (alias: `bg_stat_weight`), `hindcast_embedding_regularization_weight` (alias: `bg_dyn_weight`), `forecast_embedding_regularization_weight`, `recurrent_state_regularization_weight` (alias: `bg_regularization_weight`).
  - Inputs & Model: `target_variables`, `target_loss_weights`, `model_dropout`, `timestep_dropout`, `no_loss_frequencies`, `precip_forcing_keys`, `precip_min_clip`.
- **Target Component Names**: Must match `MeanEmbeddingForecastLSTM.supported_assimilation_targets`:
  `'static_embedding'`, `'hindcast_embedding'`, `'forecast_embedding'`, `'c_0_hindcast'`, `'h_0_hindcast'`, `'c_0_forecast'`, `'h_0_forecast'`, `'total_precipitation'`.
- **Standard Target Aliases**:
  - `'embedded_both'`: `['static_embedding', 'hindcast_embedding']`
  - `'embedded_all'`: `['static_embedding', 'hindcast_embedding', 'forecast_embedding']`
  - `'embedded_dynamics'`: `['hindcast_embedding']`
  - `'embedded_statics'`: `['static_embedding']`



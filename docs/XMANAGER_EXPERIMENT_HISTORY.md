# Historical XManager Experiments & Diagnosis Reference (`XMANAGER_EXPERIMENT_HISTORY.md`)

> [!NOTE]
> This document archives the diagnoses, failure root causes, resource requirements, and benchmark findings from historical XManager runs. AI agents should consult this file before launching new experiments or diagnosing Borg/XManager failures.

---

## 1. Key Experiment Runs & Diagnostic Log

### A. Scaler Zero-Variance & Division-by-Zero Fixes
* **Failed Run**: [`xid/279505123`](https://xmanager.corp.google.com/experiments/279505123)
* **Succeeded Baseline**: [`xid/279256158`](https://xmanager.corp.google.com/experiments/279256158)
* **Root Cause Diagnosed**: Constant/zero-variance static features produced zero standard deviation ($\sigma = 0$) in `scaler.py`, triggering `FloatingPointError` / `NaN` loss during training.
* **Resolution**: Added epsilon regularization ($\sigma + 10^{-7}$) and constant feature masking in `datautils/scaler.py`.

---

### B. Mean Embedding Cross-Validation Sweep
* **Experiment ID**: [`xid/279566061`](https://xmanager.corp.google.com/experiments/279566061)
* **Configuration**: 5-Fold Cross Validation sweep across 56 work units.
* **Key Finding**: Foundation model convergence requires $\ge 85$ epochs for MultiMet embeddings to stabilize across all 4 meteorological providers.

---

### C. Large-Scale Data Assimilation (DA) 400-Basin Benchmark
* **Experiment ID**: [`xid/283649031`](https://xmanager.corp.google.com/experiments/283649031)
* **Evaluation Period**: 3-Year Continuous Timeseries (2017-01-01 to 2019-12-31)
* **Goal**: Benchmark DA state updating and embedding vector optimization against the unassimilated baseline model across 400 Caravan catchments.
* **Findings**:
  - Unassimilated baseline median NSE on clean basins: `~0.70 - 0.72`.
  - Significant performance boost observed for lead times 0–2 when optimizing `[h_n, c_n]` with `assimilation_window=3` or `5`.

---

### D. Single-Basin & Sharded DA Parameter Selection
* **Experiment ID**: [`xid/283692435`](https://xmanager.corp.google.com/experiments/283692435)
* **Target Script**: `googlehydrology/pipelines/1_data_assimilation/batched_da_param_selection.py`
* **Observation**: Single-basin runs hanging or exceeding 40 minutes per basin was traced to unvectorized sequential disk reads on CNS NetCDFs.
* **Resolution**: Consolidated all timeseries into unified Zarr stores with `consolidated=True`, reducing runtime to < 2 minutes per shard.

---

### E. Embeddings Stage 2 (78-Config Full Sweep)
* **Experiment ID**: [`xid/284712352`](https://xmanager.corp.google.com/experiments/284712352)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_stage2_78cfgs`
* **Findings**:
  - Global Best: `cfg_embedded_both_w180_lr0.1_ep100_bg0.005_bgdyn0.02_decay0.9s5` ($t+1$ NSE 0.510 vs 0.318 baseline).
  - Decoupled regularization ($bg_{stat}=0.005, bg_{dyn}=0.02$) decisively outperforms static alone and uniform regularization.
  - Window length $w=180$ days dominates $w=90$ and $w=60$.

---

### F. Cell State Stage 6 (66-Config Substantial Sweep)
* **Experiment ID**: [`xid/284715583`](https://xmanager.corp.google.com/experiments/284715583)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_cell_state_stage6_substantial`
* **Focus**: Systematic combinations of target states (`c_both`, `c_n_forecast`), windows ($w=3,7,14$), history windows ($h=2,4$), and deep learning rate/decay schedules.

---

### G. Embeddings Stage 3 (Dynamic Weight Diagnostic Probe)
* **Experiment ID**: [`xid/285174329`](https://xmanager.corp.google.com/experiments/285174329)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_stage3_dyn_probe`
* **Focus**: 6-config probe isolating $bg_{stat} \in \{10^{-3}, 10^{-6}\}$ against $bg_{dyn} \in \{0.1, 0.01, 10^{-4}\}$ on $w=180$, $\text{LR}=0.10$, $\text{Epochs}=100$.
* **Preliminary Findings**: $bg_{stat}=10^{-6}$ substantially outperforms $10^{-3}$; $bg_{dyn} \ge 0.01$ outperforms loose dynamic regularization ($10^{-4}$).

---

### H. Embeddings Stage 4 (Focused Multi-Window & Annealing Optimization)
* **Experiment ID**: [`xid/285185982`](https://xmanager.corp.google.com/experiments/285185982)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st4_w_lr_dec_ep_bgdyn_20260831`
* **Allocation**: 25 Borg worker shards $\times$ 1 V100 GPU (32 configs in batch).
* **Focus**: Exploring interactions between $w \in \{180, 365\}$, $\text{LR} \in \{0.1, 0.3\}$, $\text{Decay} \in \{0.7, 0.9\}$, $\text{Epochs} \in \{100, 150\}$, with $bg_{stat}=10^{-6}$ and $bg_{dyn} \in \{0.1, 0.01\}$.

---

### I. Embeddings Stage 8 (3 Embeddings vs 2 Embeddings Diagnostic Probe)
* **Experiment ID**: [`xid/285840827`](https://xmanager.corp.google.com/experiments/285840827)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st8_tgt_bgdyn_50basins_20260902`
* **Allocation**: 10 Borg worker shards $\times$ 1 V100 GPU (50 basins total, 5 basins/shard).
* **Focus**: Comparing 3 embeddings optimization (`static_embedding`, `hindcast_embedding`, `forecast_embedding`) vs 2 embeddings (`static_embedding`, `hindcast_embedding`) across $\lambda_{\text{dyn}} \in \{0.01, 0.1\}$ with $w=180$, $\text{LR}=0.10$, $\text{Epochs}=100$, and multi-epoch snapshotting at $[5, 10, 50, 100]$.

---

### J. 20-Basin Multi-Lead DA Hyperparameter Verification Sweep
* **Experiment ID**: [`xid/289732205`](https://xmanager.corp.google.com/experiments/289732205)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/eval_results/da_sweep_20basins_v1`
* **Allocation**: 18 Borg worker shards $\times$ 1 V100 GPU (18 configs).
* **Focus**: Verification sweep on 20 debug catchments for 2017 testing `tester.py` fix and comparing 17 DA hyperparameter variants against unassimilated baseline (0 DA).
* **Findings**:
  - Baseline ($0$ DA): Lead 1 NSE `0.7000`, decaying to `0.5121` at Lead 7.
  - Global Best (`both_w180_lr10_s25`): Lead 1 NSE `0.7738` (+0.1267 gain), maintaining `0.7788` at Lead 7 (+0.2667 gain over baseline).

---

### K. Embeddings Stage 14 (1,000-Basin Comprehensive 6-Year Benchmark: 2017–2022)
* **Experiment ID**: [`xid/289964920`](https://xmanager.corp.google.com/experiments/289964920)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st14_w_lr_ep_bgdyn_bgstat_1000b_20260916`
* **Allocation**: 19 Borg worker shards $\times$ 1 V100 GPU (18 DA configs + 1 open-loop baseline).
* **Focus**: Comprehensive 6-year continuous benchmark (2017-01-01 to 2022-12-31) across 1,000 high-quality catchments evaluating:
  - Windows: $w \in \{30, 180, 365\}$ days.
  - Learning Rates: $\text{LR} \in \{0.01, 0.05, 0.10\}$.
  - Regularization Sets: Balanced ($bg_{stat}=10^{-6}, bg_{dyn}=0.01$) vs. High Regularization ($bg_{stat}=10^{-4}, bg_{dyn}=0.10$).
  - Epochs: 50 optimization steps.

---

### L. Embeddings Stage 15 (100-Basin Rapid 80-Shard Optimization Sweep: 2017)
* **Experiment IDs**:
  - Main Sweep (76 Shards): [`xid/289978191`](https://xmanager.corp.google.com/experiments/289978191)
  - 4-Shard Rerun (`embedded_all`): [`xid/289983153`](https://xmanager.corp.google.com/experiments/289983153)
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st15_80cfgs_100b_2017_20260916`
* **Allocation**: 80 concurrent Borg worker shards $\times$ 1 V100 GPU (100 stratified basins across Caravans).
* **Focus**: Rapid 1-year (2017) hyperparameter search exploring 80 configurations:
  - 48 Core Factorial: $w \in \{30, 90, 180, 365\} \times \text{LR} \in \{0.05, 0.10, 0.20\} \times \text{ep} \in \{25, 50, 75, 100\}$.
  - 20 Regularization Sensitivity: $w \in \{180, 365\} \times \text{LR} \in \{0.10, 0.20\} \times \text{ep}=50$ over 5 $(bg_{stat}, bg_{dyn})$ pairs.
  - 11 Alternative Targets: `embedded_dynamics` (4), `embedded_all` (4), `embedded_statics` (3).
  - 1 Baseline Anchor: Unassimilated 0 DA open-loop reference on identical 100 basins.

---

### M. Embeddings Stage 16 (100-Basin Rapid 80-Shard Optimization Sweep with 7-Day Forecast Gap Fix: 2017)
* **Experiment ID**: [`xid/290040560`](https://xmanager.corp.google.com/experiments/290040560)
* **Experiment Name**: `da_embeddings_st16_80cfgs_100b_2017_20260916`
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st16_80cfgs_100b_2017_20260916`
* **Allocation**: 80 concurrent Borg worker shards $\times$ 1 V100 GPU (100 stratified basins across Caravans).
* **Focus**: Rerun of the 80-configuration grid with the 7-day forecast gap fix in place (`assimilation_lead_time = 7`). Ensures Data Assimilation optimizes strictly up to issue date $T$ ($t \le 358$), leaving the 7-day forecast horizon ($T+1 \dots T+7$) unassimilated to evaluate true lead-time degradation.

---

### N. Embeddings Stage 17 (10-Basin 2017 Single-Pass Window-Splicing DA Fix Verification)
* **Experiment ID**: [`xid/290079013`](https://xmanager.corp.google.com/experiments/290079013)
* **Experiment Name**: `da_embeddings_st17_10b_2017_fix_20260916`
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st17_10b_2017_fix_20260916`
* **Allocation**: 2 Borg worker shards $\times$ 1 V100 GPU (10 basins from `sanity_10_filtered_basins.txt`, year 2017).
* **Focus**: 10-basin, 1-year (2017) verification of the single-pass full-sequence window-splicing Data Assimilation fix (`assimilation.py` reverted to Commit 16 base + single continuous forward pass + canonical `loss.py` / `regularization.py` delegation + `tester.py` lead indexing fix). Compares `baseline_0da` vs `both_w365_lr0.1_ep50_bg0.01_stat1e-06`.
* **Findings & Diagnosis**:
  - `baseline_0da` completed successfully (median Lead 1 NSE `0.5870`, mean Lead 1–7 `0.4031`).
  - `both_w365_lr0.1_ep50_bg0.01_stat1e-06` failed with `RuntimeError: cudnn RNN backward can only be called in training mode` because PyTorch cuDNN LSTM requires `.train()` mode during backward passes. Resolved by toggling model training mode during DA inner loop optimization in `assimilation.py`.

---

### O. Embeddings Stage 17b (10-Basin 2017 DA Fix + cuDNN RNN Backward Fix Verification)
* **Experiment ID**: [`xid/290082742`](https://xmanager.corp.google.com/experiments/290082742)
* **Experiment Name**: `da_embeddings_st17b_10b_2017_fix_20260916`
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_embeddings_st17b_10b_2017_fix_20260916`
* **Allocation**: 2 Borg worker shards $\times$ 1 V100 GPU (10 basins from `sanity_10_filtered_basins.txt`, year 2017).
* **Focus**: Rerun of Stage 17 verification with cuDNN RNN backward fix in `assimilation.py` comparing `baseline_0da` vs `both_w365_lr0.1_ep50_bg0.01_stat1e-06`.

### P. Embeddings Stage 18 (20-Basin Multi-Horizon Lead-Time Decay & Open-Loop Baseline Probe: 2017)
* **Experiment ID**: [`xid/290214910`](https://xmanager.corp.google.com/experiments/290214910)
* **Experiment Name**: `da_20basins_decay_probe_2017_20260916`
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_20basins_decay_probe_2017_20260916`
* **Allocation**: 11 concurrent Borg workers $\times$ 1 V100 GPU (20 basins from `/cns/jn-d/home/floods/hydro_model/work/kruparell/eval_configs/20_basins_debug.txt`, 2017).
* **Focus**: Rigorous testing of lead-time forecast decay across leads 1 to 7 comparing unassimilated open-loop model inference (`baseline_0da`) against 10 diverse DA parameter configurations:
  - Joint embeddings: `both_w30_lr001_s25`, `both_w30_lr05_s25`, `both_w180_lr001_s25`, `both_w180_lr05_s25`, `both_w180_lr10_s25`
  - Dynamics-only: `dyn_w14_lr05_s25`, `dyn_w30_lr05_s25`, `dyn_w180_lr05_s25`
  - Statics-only: `stat_w30_lr05_s25`, `stat_w180_lr05_s25`
* **Purpose**: Determines whether the assimilation period and window lengths preserve or improve forecast skill without excessive lead-time degradation across the 7-day horizon.
* **Findings & Decay Diagnosis (All 11 Configs Evaluated)**:
  - **Dynamics-Only ($w \in \{14, 30\}\text{d}$)** preserves high baseline skill and maintains gentle decay: `dyn_w14_lr05_s25` achieved Lead 1 NSE of $0.824$ on Canadian basins, Lead 7 NSE of $0.778$ (decay delta vs baseline: $-0.011$ over 7 days), while outperforming baseline on all 16 valid basins (Lead 1 NSE $0.680$ vs $0.673$, Lead 7 NSE $0.490$ vs $0.475$). On KGE, `dyn_w14_lr05_s25` gained $+0.052$ at Lead 1 ($0.815$ vs $0.763$).
  - **Aggressive Joint Optimization ($w=180\text{d}, \text{LR}=0.10$ or $w=365\text{d}$)** causes steep lead degradation on temperate basins: `both_w180_lr10_s25` drops Canadian Lead 1 NSE from $0.824 \to 0.758$ ($-0.067$) and Lead 7 to $0.598$ ($-0.191$). Large static adjustments over long windows drift away from short-term forecast dynamics.
  - **Arid Southwestern Basins Benefit Across All Horizons**: All DA configurations improved NSE by $+0.11$ to $+0.38$ across all 7 leads on Southwestern basins, with seasonal windows (`dyn_w180_lr05` and `both_w180`) producing the highest gains.
  - **Key Recommendation**: Conservative learning rates ($\text{LR} \le 0.01$) for joint embeddings (`both_w30_lr001`) or short-window dynamics-only optimization (`dyn_w14_lr05` / `dyn_w30_lr05`) completely resolve the excessive decay problem.

---

### Q. Embeddings Stage 19 (100-Configuration Hyperparameter Sweep on 64 Filtered Basins: 2017)
* **Experiment ID**: [`xid/290331415`](https://xmanager.corp.google.com/experiments/290331415)
* **Experiment Name**: `da_100cfg_64b_st19_2017_20260917`
* **CNS Output Directory**: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/da_100cfg_64b_st19_2017_20260917`
* **Allocation**: 99 Borg workers $\times$ 1 V100 GPU (64 filtered global catchments from `filtered_64_global_basins.txt`, year 2017, single-batch per config).
* **Focus**: Comprehensive grid across:
  - Dynamics-only: $w \in \{7, 14, 21, 30, 60, 90\}\text{d}$, $\text{LR} \in \{0.01, 0.05, 0.10\}$, $\text{ep} \in \{10, 25\}$ (36 configs)
  - Statics-only: $w \in \{90, 180, 365\}\text{d}$, $\text{LR} \in \{0.0001, 0.001, 0.01\}$, $\text{ep} \in \{10, 25\}$ (18 configs)
  - Decoupled joint: $w \in \{14, 30, 90\}\text{d}$, $\text{LR}_{dyn} \in \{0.02, 0.05\}$, $\text{LR}_{stat} \in \{0.0005, 0.001, 0.005\}$, $\text{ep} \in \{10, 20\}$ (36 configs)
  - Regularized dynamics: $w \in \{14, 30\}\text{d}$, $\text{LR}=0.05$, $\text{ep}=20$, $\lambda_{bg} \in \{0.01, 0.1, 1.0, 10.0\}$ (8 configs)
  - Baseline anchor: `baseline_0da` (1 config)

---

## 2. Mandatory Borg & XManager Launch Guidelines

When launching XManager experiments in this repository:

1. **Resource Allocation Flags (Mandatory)**:
   ```bash
   --xm_resource_pool=research-dynamic \
   --xm_resource_alloc=group:research-dynamic/idrim-dynamic-shared-user
   ```
2. **Environment Override**:
   Set `overrides.env_vars.TMPDIR = "/tmp"` in `xm_abc.Borg` execution settings to avoid Borglet sandbox permission errors.
3. **TorchDynamo Restriction**:
   `torch.compile` fails in Borglet sandboxes; always ensure `torch._dynamo.config.disable = True` at entrypoint.

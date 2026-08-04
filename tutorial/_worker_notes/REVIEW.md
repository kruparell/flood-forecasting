# Review of Prior Episode Attempts (Layer-2 Synthesis)

## 1. Layer-1 Worker 1 (`/google/src/cloud/kruparell/subagent-Layer-1-Synthesis-Worker-1-DeepCoderWorkerSynthesis-a1ea708d`)
### Strengths
- Configured 10 real Caravan/CAMELS catchments (`camels_01195100`, `camels_02077200`, `camels_03500240`, `camels_04216418`, `camels_04221000`, `camels_12115000`, `camels_12377150`, `camels_12451000`, `camels_14222500`, `camels_14236200`).
- Trained 5 epochs (`model_epoch005.pt`) and ran evaluation in both Closed-Loop ($Q_{obs}$ provided) and Open-Loop ($Q_{obs}$ missing) modes using `evaluate_10basin_arlstm.py`.
- Populated Cells 11 and 12 of `ARLSTM_Bugs_Demonstration.ipynb` with real 10-basin summary tables and 5-panel hydrograph/NSE comparison figures.
- Executed `nbconvert` headlessly with 0 errors across all 4 target paths.
- Wrote `_worker_notes` in its monorepo workspace path.

### Weaknesses / Gaps
- Cell 12 `candidate_dirs` contained candidate path hardcoding prior worker directories.

---

## 2. Layer-1 Worker 2 (`/google/src/cloud/kruparell/subagent-Layer-1-Synthesis-Worker-2-DeepCoderWorkerSynthesis-0ee4289e`)
### Strengths
- Trained ARLSTM for 5 epochs (`model_epoch005.pt`) with MultiMet dynamic features and Caravan target streamflow.
- Evaluated Closed-Loop and Open-Loop modes, creating `test_results_data_assimilation.zarr` and `test_results.zarr`.
- Computed 10-basin metrics (NSE, RMSE, MAE) and $\Delta\text{NSE (Gain)}$.
- Headless execution with `nbconvert` passed with 0 errors.

### Weaknesses / Gaps
- Wrote its `_worker_notes` in `/usr/local/google/home/kruparell/flood-forecasting/tutorial/_worker_notes/` instead of inside its CitC monorepo workspace root directory.
- Cell 12 `candidate_dirs` hardcoded the workspace path of Layer-1 Worker 1 (`subagent-Layer-1-Synthesis-Worker-1-DeepCoderWorkerSynthesis-a1ea708d`).

---

## 3. Synthesis Strategy
1. **Dynamic Path Resolution**: Refactor Cell 12 in `ARLSTM_Bugs_Demonstration.ipynb` to dynamically resolve `model-runs` by searching upward from the current working directory, workspace root, and home directory without referencing any specific worker attempt directory names.
2. **Comprehensive Verification**: Verify model evaluation results (`model_epoch005.pt` Zarr stores) produce complete metrics and clear hydrograph figures for all 10 basins.
3. **Headless Execution**: Execute `jupyter nbconvert --to notebook --execute --inplace` across all 4 notebook target paths with 0 errors.
4. **Full Workspace Synchronization**: Ensure all notebook files in both home directory and google3 monorepo workspace are 100% updated and synchronized.

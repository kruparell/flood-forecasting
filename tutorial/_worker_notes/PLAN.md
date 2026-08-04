# Synthesis Implementation Plan (Layer-2 Synthesis)

## Objectives
1. Eliminate hardcoded prior worker workspace paths in Cell 12 of `ARLSTM_Bugs_Demonstration.ipynb` by implementing dynamic upward path resolution for `model-runs`.
2. Confirm the 10-basin ARLSTM evaluation results in Closed-Loop ($Q_{obs}$ provided) and Open-Loop ($Q_{obs}$ missing) modes.
3. Repopulate `ARLSTM_Bugs_Demonstration.ipynb` with real 10-basin hydrographs and evaluation metric summary tables.
4. Execute the notebook headlessly with `jupyter nbconvert --to notebook --execute --inplace` to confirm 0 errors and rendered figure outputs.
5. Synchronize the executed notebook across all 4 target paths in the home directory and google3 monorepo workspace.

## Step-by-Step Execution
1. **Refactor Cell 12**: Update candidate path resolution to search parent directories dynamically up to 5 levels above `cwd` and `/usr/local/google/home/kruparell/flood-forecasting/tutorial/model-runs`.
2. **Execute Headlessly**: Run `jupyter nbconvert` using the `googlehydrology` python environment.
3. **Copy to All Target Paths**: Sync to all 4 notebook locations.
4. **Re-verify Headless Run**: Verify all 4 copies execute without errors.
5. **Document Handoff**: Write structured report to `_worker_notes/README.md` and send report to orchestrator via `send_message`.

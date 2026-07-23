# Synthesis Implementation Plan

## Goal
Add to `tutorial/notebooks/Evaluate_50_Basin_Trained_Model.ipynb` the evaluation and hydrograph plots of the ARLSTM model results under two distinct operational inference scenarios:
1. **Zero River Discharge Given as Input During Inference** ($Q_{\text{ar, zero}}$): Open-loop meteorological simulation without streamflow observation inputs ($Q_{t-1} = 0$).
2. **Continuous / Full River Discharge Given at All Timesteps During Inference** ($Q_{\text{ar, full}}$): Antecedent streamflow observations ($Q_{t-1}^{\text{obs}}$) available continuously at all timesteps.

## Implementation Steps
1. **Model & Data Evaluation**:
   - Evaluate the 10-epoch retrained ARLSTM model under both inference modes.
   - Apply physical unscaling ($\mu = 1.7772, \sigma = 3.3810$) from `scaler.nc` with non-negativity bounding ($Q_{\text{phys}} = \max(0, Q_{\text{sim}})$).
2. **Comparative Visualizations & Metrics**:
   - Figure 1: Empirical CDF curves (NSE & KGE), model distribution boxplots, and scatter comparisons across models and modes.
   - Figure 2: Multi-model streamflow hydrographs across representative CAMELS test catchments (`camels_01054200`, `camels_01195100`, `camels_01350000`, `camels_01413500`) over the complete 1-year test period (`2011-10-01` to `2012-09-30`).
3. **Execution & Serialization**:
   - Execute all 11 notebook cells cleanly using `ExecutePreprocessor`.
   - Serialize embedded high-resolution PNG plots into the notebook JSON output blocks.
   - Maintain perfect synchronization between `tutorial/notebooks/Evaluate_50_Basin_Trained_Model.ipynb` and `tutorial/Evaluate_50_Basin_Trained_Model.ipynb`.
4. **Verification**:
   - Run automated unit tests in `googlehydrology/evaluation/assimilation_test.py` (7/7 passed).
   - Validate notebook execution count, cell output keys, and figure payload serialization.

# Integration Plan: Autoregressive Rollout (Conditional-LSTM) in `googlehydrology`

This document outlines the plan to integrate the autoregressive rollout approach (from the `Ensemble_Forecasting-Rollout-and-Diffusion` repository) into `googlehydrology`. This will enable models to use their own past predictions (or past observations, if available) as inputs for subsequent timesteps during forecasting.

## Overview of the Rollout Approach

The rollout approach (referred to as "Conditional-LSTM" in the source repository) works by:
1.  **Training**: The model is trained with "masked discharge" inputs during the forecast period. With probability $p$, the true discharge is revealed as an input feature, and a corresponding mask feature is set to 1. Otherwise, the input is 0 and the mask is 0. This teaches the model to use the discharge observations when available to update its state, and to rely on forcings when they are not.
2.  **Inference (Rollout)**: At test time, the model is run iteratively.
    *   Initialize `previous_discharge` and `discharge_mask` to 0.
    *   For each forecast step $t$ from 0 to $N-1$:
        1.  Run the model forward pass over the sequence up to step $t$ (with future steps masked).
        2.  Extract the prediction at step $t$.
        3.  Sample from the prediction (or use mean) to get $y_t$.
        4.  Update the input sequence for the next iteration: set `previous_discharge[t] = y_t` and `discharge_mask[t] = 1`.
    *   This allows the LSTM to condition its prediction for $t+1$ on the prediction made for $t$.

## Proposed Changes

```mermaid
graph TD
    Config[1. Update Config] --> Dataset[2. Update Dataset (Multimet)]
    Config --> Model[3. Update Model (HandoffForecastLSTM)]
    Dataset --> Tester[4. Implement Rollout in Tester]
    Model --> Tester
```

### 1. Configuration Changes (`Config`)

Add the following parameters to the configuration schema (e.g., in `googlehydrology/utils/config.py`):
*   `autoregressive_inputs`: List of strings specifying the names of autoregressive features (e.g., `['previous_discharge', 'discharge_mask']`). These should *not* be listed in `forecast_inputs` to prevent the dataset loader from trying to read them from disk.
*   `autoregressive_probability`: Float (0.0 to 1.0), the probability $p$ of revealing the true discharge during training.

### 2. Dataset Changes (`Multimet` in `googlehydrology/datasetzoo/multimet.py`)

Modify the `Multimet` dataset to dynamically generate the autoregressive inputs during training:
*   In `__getitem__`, if `cfg.autoregressive_inputs` is defined and `self.is_train` is `True`:
    *   Extract the target variable `y` (true discharge).
    *   Generate a binary mask with probability `cfg.autoregressive_probability`.
    *   Create `previous_discharge` as `y * mask`.
    *   Add `previous_discharge` and `discharge_mask` (the mask itself) to `sample['x_d_forecast']` dictionary.
*   If `self.is_train` is `False` (validation/test):
    *   Initialize `previous_discharge` and `discharge_mask` to tensors of zeros (placeholders that will be updated during the rollout loop in the tester).

### 3. Model Changes (`HandoffForecastLSTM` in `googlehydrology/modelzoo/handoff_forecast_lstm.py`)

Modify the model to handle autoregressive inputs without embedding them:
*   In `__init__`:
    *   `self.forecast_inputs` should still be `cfg.forecast_inputs` (features to be read from disk and embedded).
    *   Store `self.autoregressive_inputs = cfg.autoregressive_inputs`.
    *   The `forecast_embedding_net` should be created with `input_size=len(self.forecast_inputs)` (excluding autoregressive inputs).
*   In `forward`:
    *   Extract `forecast_features` using `self.forecast_inputs` and embed them.
    *   Extract `autoregressive_features` using `self.autoregressive_inputs` (do *not* embed them).
    *   Concatenate the embedded forecast features, static embeddings, and raw autoregressive features before passing to the `forecast_lstm`.
    *   *Note*: The order of concatenation must match what the model expects. We should probably append autoregressive features at the very end.

### 4. Tester Changes (`BaseTester` in `googlehydrology/evaluation/tester.py`)

Implement the iterative rollout loop for evaluation when `autoregressive_inputs` are present:
*   In `_evaluate` (or a dedicated evaluation method):
    *   If `self.cfg.autoregressive_inputs` is not empty:
        *   Implement the rollout loop:
            ```python
            # Initialize placeholders in data['x_d_forecast'] to 0
            # for step in range(lead_time):
            #   1. Run model: predictions = model(data)
            #   2. Extract prediction at 'step'
            #   3. Sample/decode prediction to get discharge value
            #   4. Update data['x_d_forecast']['previous_discharge'][:, step] = value
            #   5. Update data['x_d_forecast']['discharge_mask'][:, step] = 1
            ```
        *   This will replace the single `model(data)` call.
        *   *Note*: We need to make sure we support batching correctly during this rollout.

## Open Questions & Considerations

*   **Teacher Forcing vs. Rollout**: During training, we feed the target at the *current* step (with probability $p$). This is slightly different from standard teacher forcing where we feed the target from the *previous* step. We should verify if this is indeed the intended behavior and if it performs well.
*   **Performance**: The rollout loop requires $N$ forward passes of the forecast LSTM (where $N$ is the lead time). This will be slower than the baseline single-pass evaluation. We should optimize the loop as much as possible (e.g. by only running the forecast LSTM, not the hindcast LSTM, repeatedly, if we can cache the handed-over states).
    *   *Optimization*: In `HandoffForecastLSTM`, we can split the forward pass: run hindcast once, get handed-over states, and then loop only the forecast LSTM. This would require exposing a way to run only the forecast part of the model.


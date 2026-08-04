# Implementation Plan - ARLSTM Bugs Demonstration Notebook (Synthesis)

## Objective
Create a clear, direct, and well-structured Jupyter Notebook at `~/flood-forecasting/tutorial/notebooks/ARLSTM_Bugs_Demonstration.ipynb` (and synchronized to `~/flood-forecasting/tutorial/ARLSTM_Bugs_Demonstration.ipynb`) that demonstrates all 4 identified ARLSTM bugs using minimal PyTorch dummy tensors and synthetic configs.

## Detailed Plan

### 1. Notebook Design & Structure
The notebook is structured into 6 logical sections across 11 markdown and code cells:

- **Cell 0 [Markdown]**: Title, Executive Summary, Prerequisites, and Overview of the 4 Bugs.
- **Cell 1 [Code]**: Setup & Base Helper Functions (`create_base_config`, imports, `RUN_DIR` resolution).
- **Cell 2 [Markdown]**: Section 1 - Target-to-AR Variable Order Mismatch.
- **Cell 3 [Code]**: Section 1 Code - Demonstrates silent channel substitution when target variable order differs from autoregressive input order.
- **Cell 4 [Markdown]**: Section 2 - PyTorch Autograd In-Place Mutation RuntimeError.
- **Cell 5 [Code]**: Section 2 Code - Demonstrates in-place slice mutation (`last_prediction[0] = prediction`) and autograd version counter mismatch during `backward()` when NaNs are present in AR inputs, plus the out-of-place fix.
- **Cell 6 [Markdown]**: Section 3 - Probabilistic Head (CMAL / GMM / UMAL) Incompatibility & KeyError.
- **Cell 7 [Code]**: Section 3 Code - Demonstrates initialization `ValueError` (`output_size != num_ar_inputs`) and forward pass `KeyError: 'y_hat'`.
- **Cell 8 [Markdown]**: Section 4 - Multi-Layer Hidden State Shape Mismatch.
- **Cell 9 [Code]**: Section 4 Code - Demonstrates `ARLSTM.__init__` ignoring `num_layers` (stuck at 1) and `torch.squeeze(..., dim=1)` shape mismatch when `num_layers > 1`.
- **Cell 10 [Markdown]**: Summary of Architectural Bugs & Recommended Fixes.

### 2. Implementation Execution
- Verified the notebook structure and execution cleanly.
- Saved to `/usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/ARLSTM_Bugs_Demonstration.ipynb`.

### 3. Verification & Synchronization
- Executed notebook using `/usr/local/google/home/kruparell/miniforge3/envs/googlehydrology/bin/jupyter nbconvert --to notebook --execute /usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/ARLSTM_Bugs_Demonstration.ipynb --output /usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/ARLSTM_Bugs_Demonstration.ipynb`.
- Synchronized notebook to `/usr/local/google/home/kruparell/flood-forecasting/tutorial/ARLSTM_Bugs_Demonstration.ipynb`.
- Verified that both notebooks exist and match.

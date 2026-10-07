# DA paper analysis — "Bridging global AI hydrologic models and local gauge observations via latent data assimilation"

Analysis code and basin definitions behind the paper. The DA engine itself lives in the public
[google-research/flood-forecasting](https://github.com/google-research/flood-forecasting) package
(`googlehydrology`), which must be installed in the environment (`conda activate googlehydrology`).

| Dir | Contents |
| :-- | :-- |
| `da_eval/` | Analysis library used by the notebooks (ingestion from CNS, AR(1) benchmark, risk/return, Köppen strata, tables, figures). |
| `notebooks/01_basin_selection/` | Basin pool (NSE > 0.5 ∧ 80 % HRES availability) and 5-group / 5-fold partitions. |
| `notebooks/02_results/` | `07_…Parquet_Visualizations.ipynb` produces every table/figure; `08_…Exploration_Tools.ipynb` is its companion. |
| `basin_lists/` | Outputs of the selection notebooks (`hres_80pct/`), Caravan-v2 partitions, filtered basin lists. |
| `models/` | Baseline model run dirs (`…-85-epochs/model_epoch085` produced the st53 sweep). Weights are git-ignored. |
| `docs/` | `XMANAGER_EXPERIMENT_HISTORY.md` (XIDs ↔ CNS dirs), `EVALUATION_GROUND_TRUTH.md`, `AGENTS.md`, design notes. |
| `data/` (git-ignored) | `eval_staging/<experiment>/` mirrors of CNS results; `koppen_geiger/`; `geo/` (HydroBASINS, Natural Earth, split CSVs). |

External data: `~/Caravans_V2/{attributes,streamflow}.zarr`, `~/Caravans_MultiMet/`, `~/zenodo_2024_paper/`.
CNS source of all results: `/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/`.

`notebooks/03_interactive_da/` holds the earlier interactive run-DA-locally notebooks (01/02) and their helper; they
pre-date the public `run infer --assimilate` CLI and were written against the google3-mirrored package, so expect to
adapt imports/paths before running them again.

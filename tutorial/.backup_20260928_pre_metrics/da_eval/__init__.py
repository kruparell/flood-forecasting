"""da_eval: Unified, clutter-free Data Assimilation evaluation and visualization package."""

from da_eval.session import DAAnalysisSession, DAEvaluationSession
from da_eval.ingestion import (
    load_valid_parquets,
    parse_da_config_id,
    load_basin_attributes,
    detect_lead_times,
)
from da_eval.benchmarks import (
    build_per_basin_optimal_eval,
    build_per_basin_optimal_timeseries,
    get_global_best_config,
    compute_anova_variance_decomposition,
)
from da_eval.physical_strata import (
    compute_consolidated_physical_strata,
    classify_aridity_regimes,
    classify_baseline_tiers,
    classify_seasonal_flow_regimes,
    classify_catchment_area,
    classify_snow_fraction,
    classify_terrain_slope,
)
from da_eval.tables import (
    build_top_n_benchmark_matrix,
    build_top_n_optimal_basin_distribution_table,
    style_top_n_optimal_basin_distribution_table,
    style_top_n_table,
    style_physical_characteristics_table,
)
from da_eval.static_plots import (
    plot_ecdf_suite,
    plot_hyperparameter_effects,
    plot_leadtime_decay,
    plot_basin_characteristics_efficacy,
    plot_top_n_optimal_basin_distribution,
)
from da_eval.dashboard import InteractiveDADashboard
from da_eval.zenodo_benchmark import (
    build_grdc_crosswalk_map,
    build_zenodo_comparison_table,
    build_zenodo_per_gauge_table,
    load_pretrained_metrics,
    load_zenodo_gauge_metrics,
    plot_zenodo_comparison_suite,
    style_zenodo_comparison_table,
    style_zenodo_per_gauge_table,
)

__all__ = [
    "DAAnalysisSession",
    "load_valid_parquets",
    "parse_da_config_id",
    "load_basin_attributes",
    "detect_lead_times",
    "build_per_basin_optimal_eval",
    "build_per_basin_optimal_timeseries",
    "get_global_best_config",
    "compute_anova_variance_decomposition",
    "compute_consolidated_physical_strata",
    "classify_aridity_regimes",
    "classify_baseline_tiers",
    "classify_seasonal_flow_regimes",
    "classify_catchment_area",
    "classify_snow_fraction",
    "classify_terrain_slope",
    "build_top_n_benchmark_matrix",
    "build_top_n_optimal_basin_distribution_table",
    "style_top_n_optimal_basin_distribution_table",
    "style_top_n_table",
    "style_physical_characteristics_table",
    "plot_ecdf_suite",
    "plot_hyperparameter_effects",
    "plot_leadtime_decay",
    "plot_basin_characteristics_efficacy",
    "plot_top_n_optimal_basin_distribution",
    "InteractiveDADashboard",
    "build_grdc_crosswalk_map",
    "load_pretrained_metrics",
    "load_zenodo_gauge_metrics",
    "build_zenodo_comparison_table",
    "style_zenodo_comparison_table",
    "build_zenodo_per_gauge_table",
    "style_zenodo_per_gauge_table",
    "plot_zenodo_comparison_suite",
]


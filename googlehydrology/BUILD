# Description:
#   Library for training deep learning models with environmental focus

load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

package_group(
    name = "default_visibility",
    packages = [
        "//third_party/py/googlehydrology/...",
    ],
)

license(name = "license")

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "googlehydrology",
    srcs = [
        "__about__.py",
        "__init__.py",
        "evaluate_mean_embedding_cv.py",
        "run.py",
        "run_scheduler.py",
    ] + glob(
        [
            "datasetzoo/**/*.py",
            "datautils/**/*.py",
            "evaluation/**/*.py",
            "modelzoo/**/*.py",
            "training/**/*.py",
            "utils/**/*.py",
        ],
        exclude = [
            "datasetzoo/zarr_dataset_reader_test.py",
            "datautils/utils_test.py",
            "evaluation/assimilation_test.py",
            # Banned per AGENTS.md section 0: all DA must run via tester.py.
            "evaluation/batched_assimilation.py",
            "evaluation/batched_assimilation_test.py",
            "evaluation/redundant/**/*.py",
            "utils/multimet_helpers_test.py",
        ],
    ),
    imports = [".."],
    strict_deps = False,
    deps = [
        "//base/python:pywrapbase",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/dask",
        "//third_party/py/dateutil:dateutil_core",
        "//third_party/py/etils/epath",
        "//third_party/py/h5netcdf",  # buildcleaner: keep
        "//third_party/py/matplotlib",
        "//third_party/py/more_itertools",
        "//third_party/py/netCDF4",
        "//third_party/py/numba",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pyarrow",
        "//third_party/py/pydantic",
        "//third_party/py/pydantic:pydantic_v2",
        "//third_party/py/ruamel",
        "//third_party/py/scipy",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch:tensorboard_support",  # buildcleaner: keep
        "//third_party/py/tqdm",
        "//third_party/py/xarray",
        "//third_party/py/yaml",
        "//third_party/py/zarr",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    main = "run.py",
    python_version = "PY3",
    srcs_version = "PY3",
    strict_deps = False,
    deps = [
        ":googlehydrology",
        "//base/python:pywrapbase",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/etils/epath",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
    ],
)

py_binary(
    name = "run_scheduler",
    srcs = ["run_scheduler.py"],
    main = "run_scheduler.py",
    strict_deps = False,
    deps = [
        ":googlehydrology",
    ],
)

py_binary(
    name = "xm_launch",
    srcs = ["xm_launch.py"],
    main = "xm_launch.py",
    strict_deps = False,
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/etils/epath",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/resource_selector",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)

py_test(
    name = "assimilation_test",
    size = "medium",
    srcs = ["evaluation/assimilation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":googlehydrology",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "zarr_dataset_reader_test",
    size = "small",
    srcs = ["datasetzoo/zarr_dataset_reader_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":googlehydrology",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
        "//third_party/py/zarr",
    ],
)

py_test(
    name = "tester_reduction_test",
    size = "small",
    srcs = ["evaluation/tester_reduction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":googlehydrology",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/xarray",
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["datautils/utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":googlehydrology",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/xarray",
    ],
)

py_test(
    name = "multimet_helpers_test",
    size = "small",
    srcs = ["utils/multimet_helpers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":googlehydrology",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/torch:pytorch",
        "//third_party/py/xarray",
    ],
)

pytest_multi_tests(
    name = "googlehydrology_tests",
    timeout = "long",
    srcs = [
        "test/__init__.py",
        "test/conftest.py",
        "test/test_config_runs.py",
    ],
    data = glob([
        "test/test_configs/*",
        "test/test_data/**",
        "test/test_data/**/.z*",
        "tutorial/Caravan-nc/**",
        "tutorial/Caravan-nc/**/.z*",
    ]),
    shard_count = 20,
    tests = glob(
        ["test/test_*.py"],
    ),
    deps = [
        ":googlehydrology",
        "//third_party/py/dask",
        "//third_party/py/etils/epath",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/h5netcdf",  # buildcleaner: keep
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/torch:pytorch",
        "//third_party/py/xarray",
    ],
)

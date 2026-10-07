# Design Document: Fault-Tolerant Model Evaluation with Incremental CNS Streaming

## Section 0: Problem Statement

When evaluating pretrained hydrological models (`floodhub_eval_base_vs_filtered`) over the full set of 4,287 basins across a 41-year period (1982–2023), evaluation jobs encounter container disk exhaustion (`OSError: [Errno 28] No space left on device`). The default Borg container `/tmp` volume is a 512 MB `tmpfs` RAM disk, whereas full-set evaluation generates over 12.1 GB of daily streamflow predictions, metrics, and logs. Moreover, multi-hour evaluation runs without incremental persistence risk losing all calculated metrics if the container is preempted or encounters an unhandled exception before completion. This design establishes a robust, disk-backed local scratch workflow (`/var/tmp` with explicit 25 GB host disk allocation) with periodic background CNS streaming (every 5 minutes, matching the Data Assimilation pipeline pattern), thread-safe delta-only file sync, Zarr metadata consolidation, and 10-basin dry-run verification.

---

## Section 1: Technical Architecture & Plan

### Architecture Overview

```mermaid
flowchart TD
    subgraph "Launcher Phase (launch_eval_borg.py)"
        A["launch_eval_borg.py"] -->|1. Request disk=25GB, cpu=8, ram=32GB| B["Package //.../scripts:run_eval_borg"]
        B -->|2. Dispatch Stage 1| C["10-Basin Dry Run (--sanity_10)"]
        C -->|3. Verify CNS Outputs| D{"test_metrics.csv &<br/>test_results.zarr valid?"}
        D -->|Pass| E["4. Dispatch Stage 2: Full Set (4,287 Basins)"]
        D -->|Fail| F["Abort & Report Error"]
    end

    subgraph "Borg Container Task (run_eval_borg.py)"
        E --> G["Disk-backed Host Workspace<br/>(/var/tmp/eval_runs/{run_id})"]
        G --> H["start_evaluation()<br/>(tester.py)"]
        H -->|Computes predictions| I["Local Zarr & CSV Store"]
        
        J["Background Daemon Streamer<br/>(interval_sec = 300s)"] -->|Delta Sync Cache (mtime/size)| I
        J -->|1MB Chunked gfile RPC| K["CNS Output Directory<br/>(/cns/.../eval_results)"]

        H -->|Completion| L["Thread Join & Final Sync"]
        L --> M["zarr.consolidate_metadata()"]
        M --> N["Write .EVAL_COMPLETE & Clean Local Scratch"]
    end
```

### Data Flow & Streaming Mechanics
1. **Container Workspace**: The job mounts `/var/tmp` with an explicit 25 GB host disk allocation (`disk=25 * xm.GiB`), bypassing the 512 MB `tmpfs` RAM limit.
2. **Incremental Execution**: `tester.py` iterates over basins, appending daily predictions to `test_results.zarr` and rows to `test_metrics.csv`.
3. **5-Minute Background Daemon Streamer (Delta-Sync Cache)**:
   - A daemon thread (`start_cns_streamer`) runs concurrently with `start_evaluation()` every 300 seconds.
   - It maintains an `mtime`/`size` cache (`_SYNCED_FILE_CACHE`) to only transfer modified or newly created Zarr chunks and metric lines, eliminating $O(N^2)$ re-copy overhead.
   - Streams files in 1 MB chunks to prevent RAM usage spikes and caches created CNS directories (`_CREATED_CNS_DIRS`) to eliminate redundant `gfile.MakeDirs` RPC calls.
4. **Finalization & Sentinel**:
   - Upon evaluation completion, the background streamer thread is stopped and joined (`thread.join(timeout=30.0)`).
   - Zarr metadata is consolidated via `zarr.consolidate_metadata()`.
   - An `.EVAL_COMPLETE` sentinel file is created on CNS before local scratch space is purged.

---

## Section 2: Alternatives Considered

1. **High-Frequency (10-second) Streaming**:
   - *Rejected*: Syncing to CNS every 10 seconds generates excessive RPC overhead for tiny file diffs and risks hitting CNS API rate limits.
   - *Chosen Strategy*: 5-minute (300-second) periodic delta-sync, matching the Data Assimilation (DA) pipeline batching standard.

2. **Pure Local `/var/tmp` Storage with Post-Run Bulk Copy**:
   - *Rejected*: Waiting until minute 90 to mirror results leaves zero output on CNS if the container crashes or is preempted near the end of evaluation.

3. **Direct Zarr CNS Store (`zarr.storage.FSStore`)**:
   - *Rejected*: Modifying `tester.py` to write Zarr directly to CNS requires editing frozen core repository files, violating strict repository rules (`AGENTS.md`).

---

## Section 3: Detailed Implementation Checklist

### 1. `googlehydrology/scripts/run_eval_borg.py`
- **Purpose**: Standalone container evaluation entrypoint.
- **Functions & Signatures**:
  - `start_cns_streamer(local_dir: Path, cns_dir: str, interval_sec: float = 300.0) -> tuple[threading.Event, threading.Thread]`
    - *Inputs*: `local_dir` (Path to container scratch directory), `cns_dir` (CNS destination string), `interval_sec` (polling frequency in seconds, default 300.0).
    - *Output*: `(stop_event, streamer_thread)` handle tuple to join thread on shutdown.
  - `mirror_local_to_cns(local_src: Path, cns_dst: str, max_workers: int = 16) -> None`
    - *Behavior*: Thread-safe delta sync checking `(mtime, size)`, using 1 MB chunked streaming and parent directory caching.
  - `main(_)`: Configures `/var/tmp` scratch space, launches background streamer, executes `start_evaluation()`, joins streamer thread, consolidates Zarr metadata, writes `.EVAL_COMPLETE`, and cleans up local scratch.

### 2. `googlehydrology/scripts/BUILD`
- **Purpose**: Defines Bazel target `//third_party/py/googlehydrology/scripts:run_eval_borg`.
- **Target Definition**:
  ```python
  load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
  package(default_visibility = ["//visibility:public"])

  py_binary(
      name = "run_eval_borg",
      srcs = ["run_eval_borg.py"],
      main = "run_eval_borg.py",
      deps = [
          "//third_party/py/googlehydrology",
          "//pyglib:gfile",
          "//third_party/py/absl:app",
          "//third_party/py/absl/flags",
          "//third_party/py/ruamel",
          "//third_party/py/torch:pytorch",
      ],
  )
  ```

### 3. `tools/launch_eval_borg.py`
- **Purpose**: XManager launcher for evaluation experiments.
- **Flags**:
  - `--sanity`: Runs 100-basin evaluation.
  - `--sanity_10`: Runs 10-basin dry-run evaluation.
- **Resource Spec**: Sets `cpu=8, ram=32*GiB, disk=25*GiB` and `TMPDIR='/var/tmp', EVAL_TMPDIR='/var/tmp'`.

---

## Section 4: Verification & Testing Strategy

1. **Stage 1: 10-Basin Dry Run Verification**:
   - Command:
     ```bash
     /google/bin/releases/xmanager/cli/xmanager.par launch tools/launch_eval_borg.py -- \
       --sanity_10 \
       --xm_resource_pool=research-dynamic \
       --xm_resource_alloc=group:research-dynamic/idrim-dynamic-shared-user
     ```
   - *Acceptance Criteria*: Job completes in < 2 minutes. CNS output folder `/cns/jn-d/home/floods/hydro_model/work/kruparell/eval_results/sanity_10_base_model` contains non-empty `test_metrics.csv`, valid `test_results.zarr`, consolidated Zarr metadata, and `.EVAL_COMPLETE` sentinel file.

2. **Stage 2: Full 4,287 Basin Set Dispatch**:
   - Command:
     ```bash
     /google/bin/releases/xmanager/cli/xmanager.par launch tools/launch_eval_borg.py -- \
       --xm_resource_pool=research-dynamic \
       --xm_resource_alloc=group:research-dynamic/idrim-dynamic-shared-user
     ```
   - *Acceptance Criteria*: Delta-sync Zarr chunks appear on CNS during evaluation every 5 minutes. Scratch disk usage on container host remains bounded under 25 GB limit. Complete consolidated `.EVAL_COMPLETE` sentinel written upon finish.

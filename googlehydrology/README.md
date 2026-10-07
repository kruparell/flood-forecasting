# googlehydrology

Google3 Python package for deep-learning rainfall-runoff and streamflow forecasting models.

## Documentation

Full internal documentation is available on g3doc:
* [go/g3doc/third_party/py/googlehydrology](http://g3doc/third_party/py/googlehydrology) (or see [g3doc/index.md](g3doc/index.md))

## Quick CLI Usage

```bash
# Train a model
SKYBUILD=1 blaze run //third_party/py/googlehydrology:run -- train --config-file <path/to/config.yml>

# Evaluate a model
SKYBUILD=1 blaze run //third_party/py/googlehydrology:run -- evaluate --config-file <path/to/config.yml> --run-dir <run_dir> --epoch <N> --period test
```

## Running Tests

```bash
SKYBUILD=1 blaze test //third_party/py/googlehydrology/test/...
```

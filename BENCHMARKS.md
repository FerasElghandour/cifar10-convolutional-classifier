# Reproduction and benchmark scope

The recovered compatible checkpoint reproduces **87.80% accuracy on all 10,000 CIFAR-10 test images** (cross-entropy0.37405). A strict parameter-name conversion preserves identical logits. The new reference trainer has fixed prepared splits, best-validation selection, resume/reload checks and held-out predictions.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `train` | New reference training protocol on supplied fixed splits; validation-selected checkpoint and held-out metrics | data, config |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The two-epoch1,000-image run validates the new training path; it does not recreate the historical25-epoch training procedure.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

Acquire the original `uoft-cs/cifar10` train/test Parquet files from the [dataset mirror](https://huggingface.co/datasets/uoft-cs/cifar10/tree/main/plain_text). Preparation checks the exact published hashes; it preserves the official test split and creates validation only from training.

```sh
python prepare_data.py --train train-00000-of-00001.parquet --test test-00000-of-00001.parquet --output data/cifar.npz
python train.py --data data/cifar.npz --epochs 25 --seed 7 --output runs/cifar
python train.py --data data/cifar.npz --checkpoint checkpoints/cifar10.pt --evaluate-only --output runs/recovered-evaluation
python predict.py --checkpoint checkpoints/cifar10.pt image.png
```

For an explicit small integration run add `--limit-train 1000 --limit-test 1000` during preparation and use two epochs. That is a different benchmark. The recovered model was evaluated on all 10,000 official test images. Resume an interrupted run with `--resume --checkpoint runs/cifar/best.pt`, the same prepared data, and a **new** output directory; `--epochs` is the total target epoch count.

## Recorded command definitions

### checks

```text
{python} smoke_test.py
{python} test_training_pipeline.py
```

### train

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.

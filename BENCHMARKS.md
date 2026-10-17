# Reproduction and benchmark scope

The recovered compatible checkpoint reproduces **87.80% accuracy on all 10,000 CIFAR-10 test images** (cross-entropy0.37405). A strict parameter-name conversion preserves identical logits. The new reference trainer has fixed prepared splits, best-validation selection, resume/reload checks and held-out predictions.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark | None |
| `train` | Configurable training on supplied fixed splits; default budget differs from named published reference protocols | data, config |

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


## Device selection

`train.py --device cpu` is the default reference path. `--device mps` explicitly selects an Apple Silicon GPU; `--device cuda` requires a CUDA GPU. Unsupported or unavailable devices fail explicitly. No automatic CPU fallback is enabled. Metrics record the device, runtime and elapsed time. CPU and GPU runs are separate numerical protocols; equal seeds do not imply equal weights or results across devices. Resume requires the checkpoint's original device type and saves its accelerator RNG state.

For a generated-data device integration check, set `TRAIN_TEST_DEVICE=mps` when running `python test_training_pipeline.py`. This checks training and checkpoint reload, not real-data model quality.

## Recorded command definitions

### checks

Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark

```text
{python} smoke_test.py
{python} test_training_pipeline.py
```

### train

Configurable training on supplied fixed splits; default budget differs from named published reference protocols

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.

## Pinned local reference installation

[requirements-reference-macos.txt](requirements-reference-macos.txt) pins the installed runtime/test dependency closure for Python 3.14.3 on macOS ARM64. It accompanies the observed environment record. It is a platform-specific reference snapshot, not a Linux/CUDA lock; a new empty-environment installation has not been independently certified. The GitHub checks use their separately declared Linux environment.

```sh
python3.14 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference-macos.txt
```

Use the recorded exact Python patch version when comparing numerical results. Notebook tooling and optional FID dependencies are outside this runtime/test snapshot.

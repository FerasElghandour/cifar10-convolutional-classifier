# CIFAR-10 Convolutional Classifier

Train a compact RGB image classifier with explicit separation of augmentation, model selection, and final test evaluation.

## Implementation

A three-stage RGB convolutional classifier with separate raw-logit training and probability inference interfaces.

![Implementation mechanism: RGB batch → Two convolutions per stage at 32 then 64 then 128 channels → Pool to 2048 flattened features → 256-unit head with dropout → 10 logits for loss or softmax for inference](assets/mechanism.svg)



## Train and evaluate the image model

The recovered compatible checkpoint reproduces **87.80% accuracy on all 10,000 CIFAR-10 test images** (cross-entropy 0.37405). A strict parameter-name conversion preserves identical logits. The new reference trainer has fixed prepared splits, best-validation selection, resume/reload checks and held-out predictions.

[Commands, artifacts and benchmark scope](BENCHMARKS.md) · [Versioned profiles](reproduction.json)


## Try a small example

Generated tensors exercise logits, probability normalization and backpropagation. The example does not reproduce the historical 25-epoch accuracy.

After the local setup below, run from the repository root:

```sh
python examples/quickstart.py
```

The [example source](examples/quickstart.py), [captured output](examples/expected-output.txt) and [execution record](examples/environment.json) keep this demonstration inspectable. Floating-point values may vary with library versions.

Observed output from this run:

```text
{
  "logits_shape": [
    2,
    10
  ],
  "loss": 2.333203077316284,
  "probability_row_sums": [
    1.0,
    1.0
  ]
}
```

## Completed study

The saved 25-epoch experiment reports 87.80% accuracy on all 10,000 CIFAR-10 test images, with cross-entropy 0.3741. Per-class recall ranges from 72.0% for cats to 94.7% for ships.

The recovered checkpoint has now been independently re-evaluated on the full test set, reproducing the reported accuracy. That verifies checkpoint inference, not a fresh 25-epoch training run. The new trainer and generated-data tests have separate scopes.

## Run locally

Python 3.11 or later.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
```

Prepare your own inputs as described in [DATA.md](DATA.md). Tests use generated fixtures and do not need credentials or provider calls. [VALIDATION.md](VALIDATION.md) records their scope.

## Practical relevance and limitations

These are saved results from one seed and one training budget. The trainer supports explicit CPU, CUDA or Apple MPS selection. The archived training hardware and current device checks have separate scope; this release does not claim a fresh 25-epoch reproduction.

## Reuse

Original reproduction and validation tooling is available under MIT; see [LICENSING.md](LICENSING.md) for the explicit file scope and preserved upstream, dataset and checkpoint terms.

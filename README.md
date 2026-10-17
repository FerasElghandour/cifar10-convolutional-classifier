# CIFAR-10 Convolutional Classifier

**Feras Elghandour** · Technical case study

Train a compact RGB image classifier with explicit separation of augmentation, model selection, and final test evaluation.

## Implementation

A three-stage RGB convolutional classifier with separate raw-logit training and probability inference interfaces.

```mermaid
flowchart LR
    A[Validated inputs] --> B[Portable implementation]
    B --> C[Evaluation and contract checks]
    C --> D[Inspectable results]
```

## Completed study

The saved 25-epoch experiment reports 87.80% accuracy on all 10,000 CIFAR-10 test images, with cross-entropy 0.3741. Per-class recall ranges from 72.0% for cats to 94.7% for ships.

These historical figures describe the completed study. This refactored standalone implementation has separate validation and has not been used to recertify the historical benchmark. The public release contains the portable core and focused tests.

## Run locally

Python 3.11 or later.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python smoke_test.py
```

Prepare your own inputs as described in [DATA.md](DATA.md). Tests use generated fixtures and do not need credentials or provider calls. [VALIDATION.md](VALIDATION.md) records their scope.

## Practical relevance and limitations

These are saved results from one seed and one training budget. The full run expects CUDA. The release includes a CPU architecture check, not a new 25-epoch training claim.

This edition makes the core behavior reusable and inspectable. Historical metrics and synthetic contract tests provide different kinds of evidence; neither establishes client deployment or commercial impact.

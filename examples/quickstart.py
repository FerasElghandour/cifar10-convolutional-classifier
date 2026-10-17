"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
from classifier import ImageClassifier
# Generated RGB input exercises an untrained network only.
torch.manual_seed(0)
torch.set_num_threads(2)
model = ImageClassifier().eval()
images = torch.zeros(2, 3, 32, 32)
logits = model(images)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
loss.backward()
result = {'logits_shape': list(logits.shape), 'loss': loss.item(), 'probability_row_sums': model.probabilities(images).sum(1).detach().tolist()}

import json
print(json.dumps(result, indent=2))


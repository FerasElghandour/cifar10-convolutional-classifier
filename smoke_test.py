import torch
from classifier import ImageClassifier
torch.set_num_threads(2);model=ImageClassifier().eval();x=torch.randn(4,3,32,32)
p=model.probabilities(x);assert p.shape==(4,10) and torch.allclose(p.sum(1),torch.ones(4))
torch.nn.functional.cross_entropy(model(x),torch.tensor([0,1,2,3])).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
try:model(torch.zeros(1,1,28,28))
except ValueError:pass
else:raise AssertionError('Wrong image shape accepted')
print('RGB shape contract, probability normalization, and finite gradients passed.')

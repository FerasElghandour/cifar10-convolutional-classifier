"""Compact RGB classifier with stable logits and explicit probability inference."""
import torch
from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self,classes=10):
        super().__init__()
        if classes<2:raise ValueError('At least two classes required')
        stages=[];channels=3
        for width in (32,64,128):
            for source in (channels,width):
                stages.extend([nn.Conv2d(source,width,3,padding=1,bias=False),nn.BatchNorm2d(width),nn.ReLU()])
            stages.append(nn.MaxPool2d(2));channels=width
        self.encoder=nn.Sequential(*stages)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(2048,256),nn.ReLU(),nn.Dropout(.35),nn.Linear(256,classes))
    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(3,32,32):raise ValueError('Expected N×3×32×32 RGB tensor')
        return self.head(self.encoder(x))
    def probabilities(self,x):return self(x).softmax(dim=-1)

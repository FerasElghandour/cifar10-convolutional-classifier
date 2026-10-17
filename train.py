"""Train or evaluate the released model using fixed prepared data splits."""
from classifier import ImageClassifier
from training import main

def build_model(c):
    return ImageClassifier(classes=c["classes"])

if __name__=="__main__":
    main(build_model,{'classes': 10},binary=False,tokens=False)

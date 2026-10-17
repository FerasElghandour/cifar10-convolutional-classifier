# Input contract

ImageClassifier expects N-by-3-by-32-by-32 floating-point RGB tensors. Its forward method returns logits; probabilities returns normalized class probabilities. The smoke test generates random tensors. No image collection, data-download helper, tokenizer, or trained model is bundled. Use a separately authorized dataset and a held-out evaluation to measure a new training run.

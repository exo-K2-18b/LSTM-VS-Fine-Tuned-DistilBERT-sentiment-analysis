# IMDB Sentiment: LSTM From Scratch vs. Fine-Tuned DistilBERT

The same task solved two ways: a small LSTM trained from scratch (TensorFlow/Keras) and a pretrained DistilBERT fine-tuned with Hugging Face `Trainer` (PyTorch). Both are trained on the same IMDB movie-review data and scored on the same 25,000-review test set, which each model sees exactly once, after training.

## Results

| | LSTM (from scratch) | DistilBERT (fine-tuned) |
|---|---|---|
| **Test accuracy** | **84.97%** | **90.34%** |
| Parameters | 353,281 | 66,955,010 |
| Training time | 1.0 min (5 epochs) | 6.2 min (3 epochs, Colab GPU, fp16) |
| Best validation epoch | 2 | 3 |

Confusion matrices (rows are the true label, columns the predicted label; 0 = negative, 1 = positive):

| LSTM | Pred. negative | Pred. positive |
|---|---|---|
| **True negative** | 10,232 | 2,268 |
| **True positive** | 1,490 | 11,010 |

| DistilBERT | Pred. negative | Pred. positive |
|---|---|---|
| **True negative** | 11,178 | 1,322 |
| **True positive** | 1,092 | 11,408 |

Both models make more false-positive than false-negative errors, and the LSTM leans further that way (2,268 vs 1,490).

### Per-epoch validation results

LSTM (training metrics are averaged over each epoch with dropout active):

| Epoch | Train acc. | Train loss | Val. acc. | Val. loss |
|---|---|---|---|---|
| 1 | 77.22% | 0.4773 | 75.92% | 0.5038 |
| 2 | 88.17% | 0.3011 | 85.56% | 0.3468 |
| 3 | 91.92% | 0.2160 | 84.64% | 0.3821 |
| 4 | 93.30% | 0.1841 | 83.48% | 0.4381 |
| 5 | 94.61% | 0.1499 | 83.72% | 0.5361 |

DistilBERT:

| Epoch | Train loss | Val. loss | Val. acc. |
|---|---|---|---|
| 1 | 0.2818 | 0.2637 | 89.04% |
| 2 | 0.1813 | 0.3359 | 88.84% |
| 3 | 0.1184 | 0.3855 | 90.32% |

## What the results show

- **DistilBERT is about 5.4 points more accurate.** With 25,000 test reviews, each accuracy carries roughly ±0.2–0.3 points of sampling error, so this gap is well outside noise.
- **The LSTM overfits quickly.** Its training accuracy keeps climbing to 94.6% while validation accuracy peaks at epoch 2 (85.6%) and validation loss rises from epoch 3 onward. The best-epoch weights (`restore_best_weights=True`) are the ones evaluated.
- **DistilBERT's validation loss also rises each epoch**, even though its validation accuracy ends highest at epoch 3. The model gets more confident, including on its mistakes.
- **The cost difference is large.** DistilBERT has about 190 times as many parameters as the LSTM and needed a GPU, while the LSTM trains in about a minute.
- These results are consistent with pretrained language representations helping a lot on this task, but this experiment does not separate the effect of pretraining from the effect of model size and architecture.

## Evaluation protocol

Shared by both models:

- 25,000 training reviews from IMDB, of which 10% (2,500) are held out for validation and model selection, leaving 22,500 for training
- The 25,000-review test set is used once, at the end, with no tuning against it
- Inputs limited to 200 tokens, keeping the start of long reviews
- Accuracy as the headline metric

Differences to keep in mind when reading the comparison:

- **Tokens are not the same unit.** The LSTM uses whole words from a 10,000-word vocabulary; DistilBERT uses sub-word pieces, so 200 of its tokens cover fewer words than 200 LSTM tokens
- **The validation sets are different 2,500-review samples.** Keras takes the last 10% of the training data, while the Hugging Face split is random with a fixed seed
- **Epochs and stopping differ.** The LSTM ran up to 5 epochs with early stopping on validation accuracy (patience 3); DistilBERT ran 3 epochs and kept the best-validation-accuracy checkpoint
- **Training-time hardware was not matched.** DistilBERT was trained on a Colab GPU with mixed precision; do not read the timing difference as an exact ratio
- Each result is a single run with a single seed

## Models

**LSTM** (`lstm_imdb.py`): embedding (10,000 words, 32 dimensions) → LSTM (64 units) → dense (128, ReLU) → dropout (0.5) → sigmoid output. Adam optimizer, binary cross-entropy. Also saves the model and the word index, and includes a small helper that encodes custom text the way the training data was encoded (including Keras's index offset of 3 and the start token).

**DistilBERT** (`finetune_distilbert_imdb.py`): `distilbert-base-uncased` with a new 2-class head, full fine-tuning (no frozen layers), batch size 16, learning rate 2e-5, weight decay 0.01, 3 epochs, seed 42.

## How to run

```bash
# LSTM
pip install tensorflow scikit-learn
python lstm_imdb.py

# DistilBERT (a GPU is strongly recommended; a free Colab GPU is enough)
pip install torch transformers datasets evaluate scikit-learn
python finetune_distilbert_imdb.py
```

Both scripts print the final test accuracy and confusion matrix. The DistilBERT script saves its model to `./distilbert-imdb`, which is large and should not be committed.

## Known limitations

- **No simple baseline.** There is no classical baseline (for example TF-IDF with logistic regression) in this comparison, so it does not show how much of the gap to DistilBERT is attributable to the LSTM being a weak model for this task
- **Single run per model**, with no error bars across seeds
- **Truncation to 200 tokens** cuts long reviews for both models, and the effect on accuracy was not measured
- **Small validation set** (2,500 reviews): differences of a point or so between epochs are within noise
- **Accuracy only**, with no precision/recall breakdown or calibration analysis beyond the confusion matrices
- The LSTM is a deliberately small, simple architecture; a stronger recurrent baseline (bidirectional layers, pretrained word embeddings) was not tried

## Next steps

- Add a TF-IDF + logistic regression baseline on the same data and splits
- Run several seeds per model and report mean and spread
- Try longer inputs (up to 512 tokens) for DistilBERT and measure what truncation costs
- Read through the reviews both models get wrong, and the ones only the LSTM gets wrong
- Match hardware for the timing comparison
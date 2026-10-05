
import time
import torch
import evaluate
from datasets import DatasetDict, load_dataset
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 200
SEED = 42

# Hold out 10% of train for model selection, so the test set is used only once, at the end
raw = load_dataset("stanfordnlp/imdb")
split = raw["train"].train_test_split(test_size=0.1, seed=SEED)
data = DatasetDict({"train": split["train"], "validation": split["test"], "test": raw["test"]})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)


tokenized = data.map(tokenize_fn, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(output):
    preds = output.predictions.argmax(-1)
    return accuracy_metric.compute(predictions=preds, references=output.label_ids)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

start = time.time()
trainer.train()
print(f"Training time: {(time.time() - start) / 60:.1f} min")

# Final evaluation: the test set is touched exactly once
output = trainer.predict(tokenized["test"])
preds = output.predictions.argmax(-1)
print(f"Test accuracy: {(preds == output.label_ids).mean():.4f}")
print(confusion_matrix(output.label_ids, preds))

trainer.save_model("./distilbert-imdb")
tokenizer.save_pretrained("./distilbert-imdb")
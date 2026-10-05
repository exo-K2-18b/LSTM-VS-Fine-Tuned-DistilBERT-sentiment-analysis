import json
import re
import time

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_WORDS = 10000
MAX_LEN = 200
INDEX_FROM = 3
SEED = 42

tf.keras.utils.set_random_seed(SEED)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=NUM_WORDS)
# Keep the FIRST 200 tokens of long reviews, the same window the DistilBERT script sees
x_train = pad_sequences(x_train, maxlen=MAX_LEN, truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, truncating="post")

model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(NUM_WORDS, 32),
    LSTM(64),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

start = time.time()
# 10% of train held out for early stopping, so the test set is used only once, below
model.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[early_stop])
print(f"Training time: {(time.time() - start) / 60:.1f} min")

probs = model.predict(x_test).ravel()
preds = (probs > 0.5).astype(int)
print(f"Test accuracy: {(preds == y_test).mean():.4f}")
print(confusion_matrix(y_test, preds))

model.save("sentiment_analyzer.keras")
word_index = tf.keras.datasets.imdb.get_word_index()
with open("word_index.json", "w") as f:
    json.dump(word_index, f)


def encode(review):
    tokens = re.sub(r"[^a-z'\s]", "", review.lower()).split()
    seq = [1]  # start-of-sequence token, as in the training data
    for word in tokens:
        idx = word_index.get(word)
        seq.append(2 if idx is None or idx + INDEX_FROM >= NUM_WORDS else idx + INDEX_FROM)
    return seq


def predict_sentiment(review):
    padded = pad_sequences([encode(review)], maxlen=MAX_LEN, truncating="post")
    p = model.predict(padded, verbose=0)[0][0]
    label, confidence = ("Positive", p) if p > 0.5 else ("Negative", 1 - p)
    print(f"{label} with confidence: {confidence:.0%}")


predict_sentiment("i didn't enjoy the movie at all. it was very boring and lame.")
predict_sentiment("great")
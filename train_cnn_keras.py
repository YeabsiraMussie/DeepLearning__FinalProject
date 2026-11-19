
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from medmnist import BloodMNIST, INFO
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# 1. Reproducibility
keras.utils.set_random_seed(42)

# 2. Load official BloodMNIST partitions
train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

# 3. Normalize images
X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

y_train = train.labels.flatten().astype(np.int64)
y_val = val.labels.flatten().astype(np.int64)

# IMPORTANT: CNN images are NOT flattened
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# 4. Build CNN architecture
model = keras.Sequential([
    keras.Input(shape=(28, 28, 3)),

    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(8, activation="softmax")
])

model.summary()

# 5. Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Custom validation macro-F1 callback
class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(
            self.X_val,
            batch_size=128,
            verbose=0
        )

        predictions = np.argmax(
            probabilities, axis=1
        )

        score = f1_score(
            self.y_val,
            predictions,
            average="macro",
            labels=np.arange(8),
            zero_division=0
        )

        logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}")

# 7. Early stopping and model checkpoint
callbacks = [
    MacroF1Callback(X_val, y_val),

    keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_cnn.keras",
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True
    )
]

# 8. Train CNN
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# 9. Find the best epoch
best_epoch = np.argmax(
    history.history["val_macro_f1"]
) + 1

print("\nBest epoch:", best_epoch)

# 10. Load best saved model
model = keras.models.load_model("best_cnn.keras")

# 11. Evaluate on validation data
probabilities = model.predict(
    X_val, verbose=0
)

y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_val, y_pred)

macro_f1 = f1_score(
    y_val, y_pred,
    average="macro",
    labels=np.arange(8),
    zero_division=0
)

print("\nBEST CNN RESULTS")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Macro-F1: {macro_f1:.4f}")

class_names = [
    INFO["bloodmnist"]["label"][str(i)]
    for i in range(8)
]

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    labels=np.arange(8),
    target_names=class_names,
    zero_division=0
))

# 12. Plot learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs_run = range(
    1, len(history.history["loss"]) + 1
)

axes[0].plot(
    epochs_run, history.history["loss"],
    label="Training"
)
axes[0].plot(
    epochs_run, history.history["val_loss"],
    label="Validation"
)
axes[0].set_title("Cross-Entropy Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(
    epochs_run, history.history["accuracy"],
    label="Training"
)
axes[1].plot(
    epochs_run, history.history["val_accuracy"],
    label="Validation"
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(
    epochs_run, history.history["val_macro_f1"],
    label="Validation Macro-F1"
)
axes[2].set_title("Macro-F1")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()

plt.savefig(
    "cnn_learning_curves.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# 13. Save experiment history
import pandas as pd

pd.DataFrame(history.history).to_csv(
    "cnn_training_history.csv",
    index=False
)

print("\nCNN experiment files saved.")

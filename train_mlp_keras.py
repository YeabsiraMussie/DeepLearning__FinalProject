
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

# 1. Set random seed
keras.utils.set_random_seed(42)

# 2. Load official dataset partitions
train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

# 3. Normalize and flatten
X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

X_train = X_train.reshape(-1, 2352)
X_val = X_val.reshape(-1, 2352)

y_train = train.labels.flatten().astype(np.int64)
y_val = val.labels.flatten().astype(np.int64)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# 4. Build MLP
model = keras.Sequential([
    keras.Input(shape=(2352,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(8, activation="softmax")
])

model.summary()

# 5. Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Calculate validation macro-F1 after each epoch
class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(
            self.X_val, batch_size=128, verbose=0
        )
        predictions = np.argmax(probabilities, axis=1)

        score = f1_score(
            self.y_val, predictions,
            average="macro",
            labels=np.arange(8),
            zero_division=0
        )

        logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}")

# 7. Callbacks
callbacks = [
    MacroF1Callback(X_val, y_val),

    keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_mlp.keras",
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True
    )
]

# 8. Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs= 40,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# 9. Load the best saved model
model = keras.models.load_model("best_mlp.keras")

# 10. Evaluate the best model
probabilities = model.predict(X_val, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val, y_pred,
    average="macro",
    labels=np.arange(8),
    zero_division=0
)

print("\nBEST MLP RESULTS")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Macro-F1: {macro_f1:.4f}")

class_names = [
    INFO["bloodmnist"]["label"][str(i)]
    for i in range(8)
]

print("\nClassification Report:")
print(classification_report(
    y_val, y_pred,
    labels=np.arange(8),
    target_names=class_names,
    zero_division=0
))

# 11. Plot learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history["loss"], label="Training")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set_title("Cross-Entropy Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Training")
axes[1].plot(history.history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(
    history.history["val_macro_f1"],
    label="Validation Macro-F1"
)
axes[2].set_title("Macro-F1")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()
plt.savefig("mlp_learning_curves.png", dpi=200)
plt.show()

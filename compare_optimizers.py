
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from medmnist import BloodMNIST
from sklearn.metrics import accuracy_score, f1_score

# 1. Set random seed
SEED = 42
keras.utils.set_random_seed(SEED)

# 2. Load official data (no test set)
train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

y_train = train.labels.flatten().astype(np.int64)
y_val = val.labels.flatten().astype(np.int64)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# 3. CNN architecture - same as Step 6
def build_cnn():
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

    return model

# 4. Create one shared initialization
keras.utils.set_random_seed(SEED)
initial_model = build_cnn()
initial_weights = initial_model.get_weights()

# 5. Calculate validation macro-F1
class MacroF1Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(
            X_val, batch_size=128, verbose=0
        )

        predictions = np.argmax(
            probabilities, axis=1
        )

        score = f1_score(
            y_val, predictions,
            labels=np.arange(8),
            average="macro",
            zero_division=0
        )

        logs["val_macro_f1"] = float(score)

# 6. Define the two optimizers
optimizers = {
    "Adam": lambda: keras.optimizers.Adam(
        learning_rate=0.001
    ),
    "SGD_Momentum": lambda: keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9
    )
}

results = {}
summary = []

# 7. Train both models
for name, make_optimizer in optimizers.items():

    print("\n" + "=" * 50)
    print("TRAINING:", name)
    print("=" * 50)

    # Reset random seed for comparable shuffling
    keras.utils.set_random_seed(SEED)

    # Create fresh model
    model = build_cnn()

    # Give both models identical starting weights
    model.set_weights(initial_weights)

    model.compile(
        optimizer=make_optimizer(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        MacroF1Callback(),

        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=8,
            restore_best_weights=True
        ),

        keras.callbacks.ModelCheckpoint(
            f"cnn_{name}.keras",
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True
        )
    ]

    start_time = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=128,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

    training_time = time.perf_counter() - start_time

    # 8. Load best validation checkpoint
    best_model = keras.models.load_model(
        f"cnn_{name}.keras"
    )

    probabilities = best_model.predict(
        X_val, batch_size=128, verbose=0
    )
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions,
        labels=np.arange(8),
        average="macro",
        zero_division=0
    )

    best_epoch = int(np.argmax(
        history.history["val_macro_f1"]
    ) + 1)

    print(f"\n{name} BEST RESULTS")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    # 9. Save training history
    results[name] = history.history

    pd.DataFrame(history.history).to_csv(
        f"cnn_{name}_history.csv",
        index=False
    )

    summary.append({
        "optimizer": name,
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
        "training_seconds": training_time
    })

# 10. Save comparison table
summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    "optimizer_comparison.csv",
    index=False
)

print("\nFINAL OPTIMIZER COMPARISON")
print(summary_df.to_string(index=False))

# 11. Plot comparison curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for name, history in results.items():

    epochs = np.arange(
        1, len(history["loss"]) + 1
    )

    axes[0].plot(
        epochs, history["val_loss"],
        label=name
    )

    axes[1].plot(
        epochs, history["val_accuracy"],
        label=name
    )

    axes[2].plot(
        epochs, history["val_macro_f1"],
        label=name
    )

axes[0].set_title("Validation Cross-Entropy Loss")
axes[1].set_title("Validation Accuracy")
axes[2].set_title("Validation Macro-F1")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Loss")
axes[1].set_ylabel("Accuracy")
axes[2].set_ylabel("Macro-F1")

plt.tight_layout()
plt.savefig(
    "optimizer_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nOptimizer comparison completed!")

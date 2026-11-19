
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from medmnist import BloodMNIST
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

# 1. Reproducibility and experiment settings
SEED = 42
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.001

keras.utils.set_random_seed(SEED)

# 2. Load training and validation data
train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

y_train = train.labels.flatten().astype(np.int64)
y_val = val.labels.flatten().astype(np.int64)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# 3. Create the CNN architecture
def build_cnn(regularization="None"):

    if regularization == "L2":
        reg = keras.regularizers.l2(0.0001)
    else:
        reg = None

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 3)),

        layers.Conv2D(
            32, (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=reg
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64, (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=reg
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=reg
        ),

        # Only active for the Dropout experiment
        *(
            [layers.Dropout(0.3)]
            if regularization == "Dropout"
            else []
        ),

        layers.Dense(
            8,
            activation="softmax",
            kernel_regularizer=reg
        )
    ])

    return model

# 4. Create identical initial trainable weights
keras.utils.set_random_seed(SEED)
initial_model = build_cnn("None")
initial_weights = initial_model.get_weights()

# 5. Callback for validation macro-F1
# Also records cross-entropy without the L2 penalty
class ValidationMetrics(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        preds = np.argmax(probs, axis=1)

        macro_f1 = f1_score(
            y_val,
            preds,
            labels=np.arange(8),
            average="macro",
            zero_division=0
        )

        cross_entropy = np.mean(
            keras.losses.sparse_categorical_crossentropy(
                y_val, probs
            ).numpy()
        )

        logs["val_macro_f1"] = float(macro_f1)
        logs["val_cross_entropy"] = float(cross_entropy)

# 6. Experiment configurations
experiments = ["None", "Dropout", "L2"]

results = {}
summary = []

# 7. Run each experiment
for name in experiments:

    print("\n" + "=" * 55)
    print("EXPERIMENT:", name)
    print("=" * 55)

    keras.utils.set_random_seed(SEED)

    model = build_cnn(name)
    model.set_weights(initial_weights)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ValidationMetrics(),

        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=8,
            restore_best_weights=True
        ),

        keras.callbacks.ModelCheckpoint(
            f"regularization_{name}.keras",
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True
        )
    ]

    start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

    elapsed = time.perf_counter() - start

    # 8. Evaluate best validation checkpoint
    best_model = keras.models.load_model(
        f"regularization_{name}.keras"
    )

    probs = best_model.predict(
        X_val, batch_size=BATCH_SIZE, verbose=0
    )

    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_val, preds)

    macro_f1 = f1_score(
        y_val, preds,
        labels=np.arange(8),
        average="macro",
        zero_division=0
    )

    best_epoch = int(
        np.argmax(history.history["val_macro_f1"]) + 1
    )

    print(f"\n{name} BEST RESULTS")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")

    # 9. Save metrics and histories
    results[name] = history.history

    pd.DataFrame(history.history).to_csv(
        f"regularization_{name}_history.csv",
        index=False
    )

    summary.append({
        "regularization": name,
        "best_epoch": best_epoch,
        "epochs_run": len(history.history["loss"]),
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
        "training_seconds": elapsed
    })

    # Save per-class metrics for the audit
    report = classification_report(
        y_val, preds,
        labels=np.arange(8),
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        f"regularization_{name}_classification.csv"
    )

# 10. Final comparison table
summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "regularization_comparison.csv",
    index=False
)

print("\nFINAL REGULARIZATION COMPARISON")
print(summary_df.to_string(index=False))

# 11. Plot all three experiments
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for name, history in results.items():

    epoch_numbers = np.arange(
        1, len(history["loss"]) + 1
    )

    axes[0].plot(
        epoch_numbers,
        history["val_cross_entropy"],
        label=name
    )

    axes[1].plot(
        epoch_numbers,
        history["val_accuracy"],
        label=name
    )

    axes[2].plot(
        epoch_numbers,
        history["val_macro_f1"],
        label=name
    )

axes[0].set_title("Validation Cross-Entropy")
axes[1].set_title("Validation Accuracy")
axes[2].set_title("Validation Macro-F1")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
    ax.legend()

axes[0].set_ylabel("Cross-Entropy")
axes[1].set_ylabel("Accuracy")
axes[2].set_ylabel("Macro-F1")

plt.tight_layout()

plt.savefig(
    "regularization_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nRegularization experiments completed!")

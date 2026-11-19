
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from medmnist import BloodMNIST
from sklearn.metrics import accuracy_score, f1_score

# ============================================
# 1. EXPERIMENT CONFIGURATION
# ============================================

SEEDS = [42, 123, 2026]
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.001

REGULARIZATIONS = ["None", "Dropout", "L2"]

OUTPUT_DIR = "multiseed_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 2. LOAD DATASET
# ============================================

train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

y_train = train.labels.flatten().astype(np.int64)
y_val = val.labels.flatten().astype(np.int64)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# Test set remains untouched.

# ============================================
# 3. DEFINE CNN
# ============================================

def build_cnn(regularization="None"):

    reg = (
        keras.regularizers.l2(0.0001)
        if regularization == "L2"
        else None
    )

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

# ============================================
# 4. VALIDATION MACRO-F1 CALLBACK
# ============================================

class MacroF1Callback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        probs = self.model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions = np.argmax(probs, axis=1)

        score = f1_score(
            y_val,
            predictions,
            labels=np.arange(8),
            average="macro",
            zero_division=0
        )

        logs["val_macro_f1"] = float(score)

# ============================================
# 5. RUN ALL EXPERIMENTS
# ============================================

results = []

for seed in SEEDS:

    print("\n" + "=" * 60)
    print(f"STARTING RANDOM SEED: {seed}")
    print("=" * 60)

    # Create one shared initialization per seed
    keras.utils.set_random_seed(seed)

    initial_model = build_cnn("None")
    initial_weights = initial_model.get_weights()

    for regularization in REGULARIZATIONS:

        print("\n" + "-" * 50)
        print(
            f"Seed: {seed} | "
            f"Regularization: {regularization}"
        )
        print("-" * 50)

        keras.utils.set_random_seed(seed)

        model = build_cnn(regularization)

        # Identical initial weights within this seed
        model.set_weights(initial_weights)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model_path = os.path.join(
            OUTPUT_DIR,
            f"cnn_{regularization}_seed{seed}.keras"
        )

        callbacks = [
            MacroF1Callback(),

            keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=PATIENCE,
                restore_best_weights=True
            ),

            keras.callbacks.ModelCheckpoint(
                model_path,
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

        # Load the best validation checkpoint
        best_model = keras.models.load_model(
            model_path
        )

        probabilities = best_model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions = np.argmax(
            probabilities, axis=1
        )

        accuracy = accuracy_score(
            y_val, predictions
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=np.arange(8),
            average="macro",
            zero_division=0
        )

        best_epoch = int(
            np.argmax(
                history.history["val_macro_f1"]
            ) + 1
        )

        epochs_run = len(history.history["loss"])

        print("\nBEST RESULTS")
        print("Seed:", seed)
        print("Regularization:", regularization)
        print("Best epoch:", best_epoch)
        print("Epochs trained:", epochs_run)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Training time: {elapsed:.2f} s")

        results.append({
            "seed": seed,
            "regularization": regularization,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
            "training_seconds": elapsed
        })

        # Save epoch-by-epoch history
        history_path = os.path.join(
            OUTPUT_DIR,
            f"history_{regularization}_seed{seed}.csv"
        )

        pd.DataFrame(
            history.history
        ).to_csv(history_path, index=False)

        keras.backend.clear_session()

    del initial_model

# ============================================
# 6. SAVE ALL INDIVIDUAL RESULTS
# ============================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR, "all_seed_results.csv"
    ),
    index=False
)

print("\nALL EXPERIMENT RESULTS")
print(results_df.to_string(index=False))

# ============================================
# 7. MEAN AND STANDARD DEVIATION
# ============================================

summary = results_df.groupby(
    "regularization"
).agg(
    mean_accuracy=("val_accuracy", "mean"),
    std_accuracy=("val_accuracy", "std"),
    mean_macro_f1=("val_macro_f1", "mean"),
    std_macro_f1=("val_macro_f1", "std"),
    mean_best_epoch=("best_epoch", "mean")
)

summary = summary.sort_values(
    "mean_macro_f1", ascending=False
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR, "multiseed_summary.csv"
    )
)

print("\n" + "=" * 60)
print("FINAL MULTI-SEED SUMMARY")
print("=" * 60)
print(summary.to_string())

# ============================================
# 8. VISUALIZE SEED VARIABILITY
# ============================================

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5)
)

for i, metric in enumerate([
    "val_accuracy", "val_macro_f1"
]):

    values = [
        results_df[
            results_df["regularization"] == name
        ][metric].values
        for name in REGULARIZATIONS
    ]

    axes[i].boxplot(
        values,
        tick_labels=REGULARIZATIONS
    )

    axes[i].set_title(
        "Validation Accuracy"
        if i == 0 else "Validation Macro-F1"
    )

    axes[i].set_ylabel("Score")
    axes[i].grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR, "multiseed_comparison.png"
    ),
    dpi=200
)

plt.show()

print("\nAll experiments completed!")

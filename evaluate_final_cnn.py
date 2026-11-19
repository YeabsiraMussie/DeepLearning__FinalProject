
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from medmnist import BloodMNIST, INFO

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss
)

# ============================================
# 1. SETTINGS
# ============================================

MODEL_PATH = os.path.join(
    "multiseed_results",
    "cnn_Dropout_seed42.keras"
)

OUTPUT_DIR = "final_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class_names = [
    INFO["bloodmnist"]["label"][str(i)]
    for i in range(8)
]

short_names = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature granul.",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet"
]

# ============================================
# 2. LOAD THE TRAINED MODEL
# ============================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = keras.models.load_model(MODEL_PATH)

print("Loaded final CNN:", MODEL_PATH)
model.summary()

# ============================================
# 3. LOAD OFFICIAL TEST PARTITION
# ============================================

test = BloodMNIST(
    split="test",
    download=True
)

X_test = test.imgs.astype(np.float32) / 255.0
y_test = test.labels.flatten().astype(np.int64)

print("\nTest shape:", X_test.shape)
print("Test labels:", y_test.shape)

assert X_test.shape == (3421, 28, 28, 3)

# ============================================
# 4. MAKE TEST PREDICTIONS
# ============================================

probabilities = model.predict(
    X_test,
    batch_size=128,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# ============================================
# 5. CALCULATE FINAL TEST METRICS
# ============================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    labels=np.arange(8),
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

cross_entropy = log_loss(
    y_test,
    probabilities,
    labels=np.arange(8)
)

print("\n" + "=" * 45)
print("FINAL BLOODMNIST TEST RESULTS")
print("=" * 45)

print(f"Test accuracy:       {accuracy:.4f}")
print(f"Test macro-F1:       {macro_f1:.4f}")
print(f"Test weighted-F1:    {weighted_f1:.4f}")
print(f"Test cross-entropy:  {cross_entropy:.4f}")

# ============================================
# 6. CLASSIFICATION REPORT
# ============================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(8),
    target_names=class_names,
    digits=4,
    zero_division=0
))

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(8),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(
        OUTPUT_DIR, "test_classification_report.csv"
    )
)

# ============================================
# 7. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(8)
)

fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=short_names
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    colorbar=False
)

ax.set_title("BloodMNIST Test Confusion Matrix")

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR, "test_confusion_matrix.png"
    ),
    dpi=200
)

# ============================================
# 8. NORMALIZED CONFUSION MATRIX
# ============================================

cm_normalized = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(8),
    normalize="true"
)

fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=short_names
).plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",
    xticks_rotation=45,
    colorbar=False
)

ax.set_title(
    "BloodMNIST Normalized Test Confusion Matrix"
)

plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "test_confusion_matrix_normalized.png"
    ),
    dpi=200
)

# ============================================
# 9. SAVE SUMMARY AND PREDICTIONS
# ============================================

summary = pd.DataFrame([{
    "model": "CNN_Adam_Dropout_seed42",
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
    "test_weighted_f1": weighted_f1,
    "test_cross_entropy": cross_entropy
}])

summary.to_csv(
    os.path.join(
        OUTPUT_DIR, "final_test_summary.csv"
    ),
    index=False
)

predictions_df = pd.DataFrame({
    "sample_index": np.arange(len(y_test)),
    "true_label": y_test,
    "predicted_label": y_pred,
    "confidence": np.max(probabilities, axis=1)
})

predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR, "test_predictions.csv"
    ),
    index=False
)

print("\nFinal evaluation files saved to:")
print(OUTPUT_DIR)

plt.show()

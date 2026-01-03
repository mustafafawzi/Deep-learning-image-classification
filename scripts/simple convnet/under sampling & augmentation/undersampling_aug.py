import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report



# Configuratie
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30

# Werkdirectory = map waar dit script staat
BASE_DIR = Path(__file__).resolve().parent

# Gebruik undersampled dataset root folder
DATASET_ROOT = BASE_DIR / "picasso_rubens_undersampled"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR   = DATASET_ROOT / "val"
TEST_DIR  = DATASET_ROOT / "test"

OUT_DIR = BASE_DIR / "out_simple_convnet_undersampled_aug_dropout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_model.keras"


# Helpers
def save_training_curves(history, out_path: Path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("accuracy", []), label="train acc")
    plt.plot(epochs, hist.get("val_accuracy", []), label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("loss", []), label="train loss")
    plt.plot(epochs, hist.get("val_loss", []), label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# class_names bewaren VOOR prefetch
class_names = train_ds.class_names
print("Class names (alphabetisch):", class_names)
print("→ label 0 =", class_names[0])
print("→ label 1 =", class_names[1])

# Performance: prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)


# Model: data augmentation + dropout
inputs = keras.Input(shape=IMG_SIZE + (3,))

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")

x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

# Conv Block 1
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.15)(x)

# Conv Block 2
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.20)(x)

# Conv Block 3
x = layers.Conv2D(128, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.25)(x)

# Conv Block 4
x = layers.Conv2D(256, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.30)(x)

# Extra conv
x = layers.Conv2D(256, 3, activation="relu")(x)

# Head
x = layers.Flatten()(x)
x = layers.Dropout(0.15)(x)        
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"]
)

model.summary()



# Train (EarlyStopping + Best Model)
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        str(BEST_MODEL_PATH),
        save_best_only=True,
        monitor="val_loss",
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Metrics summary
hist = history.history

best_val_loss_epoch = int(np.argmin(hist["val_loss"]))
best_val_loss = float(hist["val_loss"][best_val_loss_epoch])
best_val_acc_at_best_loss = float(hist["val_accuracy"][best_val_loss_epoch])
train_loss_at_best_loss = float(hist["loss"][best_val_loss_epoch])
train_acc_at_best_loss = float(hist["accuracy"][best_val_loss_epoch])

best_val_acc_epoch = int(np.argmax(hist["val_accuracy"]))
best_val_acc = float(hist["val_accuracy"][best_val_acc_epoch])
val_loss_at_best_acc = float(hist["val_loss"][best_val_acc_epoch])

summary_lines = [
    "=== BEST METRICS SUMMARY ===",
    f"Best val_loss epoch: {best_val_loss_epoch + 1}",
    f"  train_loss: {train_loss_at_best_loss:.4f}",
    f"  val_loss  : {best_val_loss:.4f}",
    f"  train_acc : {train_acc_at_best_loss:.4f}",
    f"  val_acc   : {best_val_acc_at_best_loss:.4f}",
    "",
    f"Best val_accuracy epoch: {best_val_acc_epoch + 1}",
    f"  val_accuracy: {best_val_acc:.4f}",
    f"  val_loss    : {val_loss_at_best_acc:.4f}",
    "",
]

summary_text = "\n".join(summary_lines)
print(summary_text)

best_metrics_path = OUT_DIR / "best_training_metrics.txt"
best_metrics_path.write_text(summary_text)

history_csv = OUT_DIR / "history.csv"
pd.DataFrame(hist).to_csv(history_csv, index=False)

curves_path = OUT_DIR / "training_curves.png"
save_training_curves(history, curves_path)


# Test evaluation + confusion matrix + report
test_loss, test_acc = model.evaluate(test_ds)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

y_true = np.concatenate([y for _, y in test_ds]).astype(int).ravel()

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix — Rubens vs Picasso (Testset)")
fig.tight_layout()

cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=200)
plt.close(fig)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=3
)

report_path = OUT_DIR / "classification_report.txt"
report_path.write_text(report)

metrics_path = OUT_DIR / "test_metrics.txt"
metrics_path.write_text(f"test_loss={test_loss}\ntest_accuracy={test_acc}\n")

print("Training klaar. Beste model opgeslagen naar:", BEST_MODEL_PATH)
print("Outputs opgeslagen in:", OUT_DIR)

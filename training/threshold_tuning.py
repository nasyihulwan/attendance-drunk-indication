import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("drunk_sober_mobilenet.h5")

# =========================
# LOAD TEST DATA
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "../dataset/test"

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = test_data.classes
y_prob = model.predict(test_data, verbose=1).reshape(-1)

# =========================
# THRESHOLD SWEEP
# =========================
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

print("\n=== THRESHOLD TUNING RESULTS ===\n")

for th in thresholds:
    y_pred = (y_prob > th).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"Threshold = {th}")
    print(cm)
    print(f"False Positive (sober -> drunk): {fp}")
    print(f"False Negative (drunk -> sober): {fn}")

    report = classification_report(
        y_true, y_pred,
        target_names=list(test_data.class_indices.keys()),
        digits=3
    )
    print(report)
    print("-" * 50)

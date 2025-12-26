import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# =========================
# PREDICTION
# =========================
y_true = test_data.classes  # ground truth
y_prob = model.predict(test_data, verbose=1)
y_pred = (y_prob > 0.5).astype(int).reshape(-1)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =========================
# REPORT
# =========================
class_names = list(test_data.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# VISUALIZE
# =========================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Threshold = 0.5)")
plt.tight_layout()
plt.show()

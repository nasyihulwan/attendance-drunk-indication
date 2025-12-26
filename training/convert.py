import tensorflow as tf
import os

print(f"TensorFlow version: {tf.__version__}")

# Path model lama
OLD_MODEL = "drunk_sober_mobilenet.h5"
NEW_MODEL = "drunk_sober_mobilenet_v2"
NEW_MODEL_H5 = "drunk_sober_mobilenet_compatible.h5"

if not os.path.exists(OLD_MODEL):
    print(f"❌ Model not found at {OLD_MODEL}")
    print("Please put your model file there first!")
    exit(1)

print(f"\n{'='*60}")
print("MODEL CONVERTER - TensorFlow 2.20 Compatible")
print(f"{'='*60}\n")

# ========================================
# Method 1: Convert . h5 to SavedModel
# ========================================
print("Method 1: Converting .h5 to SavedModel format...")
try:
    # Load old model
    print(f"  Loading model from {OLD_MODEL}...")
    model = tf.keras.models. load_model(OLD_MODEL, compile=False)
    print("  ✓ Model loaded")
    
    # Save as SavedModel (recommended format)
    print(f"  Saving as SavedModel to {NEW_MODEL}...")
    model.save(NEW_MODEL, save_format='tf')
    print(f"  ✓ SavedModel created at {NEW_MODEL}/")
    
except Exception as e:
    print(f"  ✗ Method 1 failed: {e}")

# ========================================
# Method 2: Rebuild and Save Compatible . h5
# ========================================
print("\nMethod 2: Rebuilding model architecture...")
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.models import Model
    
    # Rebuild architecture (sama seperti training script)
    print("  Building MobileNetV2 architecture...")
    base_model = MobileNetV2(
        weights=None,  # No pretrained weights
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation="sigmoid")(x)
    
    new_model = Model(inputs=base_model.input, outputs=output)
    print("  ✓ Architecture built")
    
    # Load weights from old model
    print(f"  Loading weights from {OLD_MODEL}...")
    old_model = tf.keras.models.load_model(OLD_MODEL, compile=False)
    
    # Transfer weights
    new_model.set_weights(old_model.get_weights())
    print("  ✓ Weights transferred")
    
    # Compile
    new_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("  ✓ Model compiled")
    
    # Save new . h5
    print(f"  Saving compatible .h5 to {NEW_MODEL_H5}...")
    new_model.save(NEW_MODEL_H5, save_format='h5')
    print(f"  ✓ Compatible . h5 created at {NEW_MODEL_H5}")
    
except Exception as e:
    print(f"  ✗ Method 2 failed: {e}")

# ========================================
# Verification
# ========================================
print(f"\n{'='*60}")
print("VERIFICATION")
print(f"{'='*60}\n")

# Test SavedModel
if os.path.exists(NEW_MODEL):
    try:
        print(f"Testing SavedModel...")
        test_model = tf.keras.models.load_model(NEW_MODEL)
        
        import numpy as np
        dummy_input = np.random.rand(1, 224, 224, 3)
        prediction = test_model.predict(dummy_input, verbose=0)
        
        print(f"  ✓ SavedModel works! Prediction:  {prediction[0][0]:. 4f}")
    except Exception as e:
        print(f"  ✗ SavedModel test failed: {e}")

# Test compatible .h5
if os.path.exists(NEW_MODEL_H5):
    try:
        print(f"\nTesting compatible .h5...")
        test_model = tf. keras.models.load_model(NEW_MODEL_H5)
        
        import numpy as np
        dummy_input = np.random.rand(1, 224, 224, 3)
        prediction = test_model.predict(dummy_input, verbose=0)
        
        print(f"  ✓ Compatible .h5 works! Prediction: {prediction[0][0]:.4f}")
    except Exception as e:
        print(f"  ✗ Compatible .h5 test failed: {e}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print("\nTo use in your app, update model path to one of:")
print(f"  1. {NEW_MODEL}  (SavedModel - recommended)")
print(f"  2. {NEW_MODEL_H5}  (Compatible .h5)")
print()
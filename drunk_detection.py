import cv2
import numpy as np
import os
import threading

class DrunkDetection:  
    def __init__(self, model_path, threshold=0.60):
        self.model_path = model_path
        self.threshold = threshold
        self. img_size = (224, 224)
        self.min_blur = 50
        self.min_brightness = 60
        self.model = None
        self.model_lock = threading.Lock()
        
        # Lazy load - tidak pre-load untuk avoid startup errors
        print("✅ Drunk detection initialized (model will load on first use)")
    
    def _load_model(self):
        """Load TensorFlow model dengan proper error handling"""
        if self.model is not None:
            return True
        
        with self.model_lock:
            if self.model is not None:
                return True
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model not found:  {self.model_path}")
                return False
            
            try:
                import tensorflow as tf
                
                print("⏳ Loading drunk detection model...")
                
                # Method 1: Load with compile=False
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False
                )
                
                # Warmup prediction
                dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
                
                print("✅ Drunk detection model loaded successfully")
                return True
                
            except Exception as e:
                print(f"⚠️ Error loading model with method 1: {e}")
                
                # Method 2: Rebuild MobileNetV2 architecture
                try:  
                    import tensorflow as tf
                    
                    print("🔄 Trying to rebuild MobileNetV2 architecture...")
                    
                    # Rebuild model architecture
                    base_model = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights=None  # Don't load pretrained weights
                    )
                    
                    x = base_model.output
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers. Dense(1, activation='sigmoid')(x)
                    
                    self.model = tf.keras. Model(inputs=base_model. input, outputs=x)
                    
                    # Try to load saved weights
                    try: 
                        self.model.load_weights(self.model_path)
                        print("✅ Model weights loaded successfully")
                    except Exception as weight_error:
                        print(f"⚠️ Could not load weights: {weight_error}")
                        print("⚠️ Using untrained model - predictions may be inaccurate")
                    
                    # Warmup
                    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    _ = self.model.predict(dummy_input, verbose=0)
                    
                    print("✅ Model ready (architecture rebuilt)")
                    return True
                    
                except Exception as e2:
                    print(f"❌ All loading methods failed: {e2}")
                    self.model = None
                    return False
    
    def is_low_quality(self, frame):
        """Check if frame quality is too low"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = gray.mean()
            return blur < self. min_blur or brightness < self. min_brightness
        except Exception as e:
            print(f"Error in quality check: {e}")
            return True
    
    def predict(self, frame):
        """Predict drunk/sober from frame"""
        # Load model on first use
        if self.model is None:
            if not self._load_model():
                return "ERROR", 0.0
        
        try:  
            # Preprocessing (simple normalization)
            resized = cv2.resize(frame, self. img_size)
            norm = resized / 255.0  # Normalize to [0, 1]
            inp = np.expand_dims(norm, axis=0)
            
            # Prediction
            prob_sober = float(self.model.predict(inp, verbose=0)[0][0])
            label = "SOBER" if prob_sober > self.threshold else "DRUNK"
            
            return label, prob_sober
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "ERROR", 0.0
    
    def predict_batch(self, frames):
        """Predict batch of frames for efficiency"""
        if self.model is None:
            if not self._load_model():
                return [("ERROR", 0.0) for _ in frames]
        
        try:
            # Preprocessing batch
            batch = []
            for frame in frames: 
                resized = cv2.resize(frame, self.img_size)
                norm = resized / 255.0
                batch.append(norm)
            
            batch_array = np.array(batch)
            
            # Batch prediction
            probs = self.model.predict(batch_array, verbose=0)
            
            results = []
            for prob_sober in probs: 
                prob_value = float(prob_sober[0])
                label = "SOBER" if prob_value > self. threshold else "DRUNK"
                results.append((label, prob_value))
            
            return results
        except Exception as e: 
            print(f"Error in batch prediction: {e}")
            return [("ERROR", 0.0) for _ in frames]
import cv2
import numpy as np
import os
import json
import threading
from queue import Queue, Empty

class FaceRecognition:  
    def __init__(self, known_faces_dir, database_file):
        self.known_faces_dir = known_faces_dir
        self.database_file = database_file
        self.face_database = {}
        
        # cascade classifier untuk face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # deepface lazy loading
        self.deepface = None
        self.deepface_lock = threading.Lock()
        
        # async identification queue
        self.identification_queue = Queue(maxsize=2)
        self.identification_results = {}
        self.results_lock = threading.Lock()
        
        os.makedirs(known_faces_dir, exist_ok=True)
        self.load_database()
        
        # pre-load deepface di background
        threading.Thread(target=self._preload_deepface, daemon=True).start()
    
    def _preload_deepface(self):
        """pre-load deepface model di background untuk menghindari lag"""
        print("🔄 preloading deepface model...")
        self._init_deepface()
    
    def _init_deepface(self):
        """lazy initialization of deepface dengan thread safety"""
        if self.deepface is None:
            with self.deepface_lock:
                if self.deepface is None:  # double-check locking
                    try:
                        from deepface import DeepFace
                        self.deepface = DeepFace
                        print("✅ deepface initialized")
                    except Exception as e:
                        print(f"❌ deepface initialization failed: {e}")
                        self.deepface = None
        return self.deepface is not None
    
    def load_database(self):
        """load face database dari json"""
        if os.path.exists(self.database_file):
            with open(self.database_file, "r") as f:
                self.face_database = json.load(f)
            print(f"✅ loaded {len(self.face_database)} people from database")
        else:
            self.face_database = {}
            self.save_database()
            print("📁 created new face database")
    
    def save_database(self):
        """save face database to json"""
        with open(self.database_file, "w") as f:
            json.dump(self.face_database, f, indent=2)
    
    def detect_face(self, frame):
        """detect face in frame - operasi ringan, bisa di main thread"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(80, 80)
            )
            
            if len(faces) > 0:
                # ambil face terbesar
                (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                
                # tambah padding
                pad = 25
                x2 = max(0, x - pad)
                y2 = max(0, y - pad)
                w2 = min(frame.shape[1] - x2, w + pad * 2)
                h2 = min(frame.shape[0] - y2, h + pad * 2)
                
                face_crop = frame[y2:y2+h2, x2:x2+w2]
                return face_crop, (x2, y2, w2, h2)
        except Exception as e:
            print(f"error in face detection: {e}")
        return None, None
    
    def identify_face_async(self, face_img, request_id):
        """identifikasi face secara async - tidak block main thread"""
        def _identify_worker():
            result = self._identify_face_sync(face_img)
            with self.results_lock:
                self.identification_results[request_id] = result
        
        # jalankan di background thread
        threading.Thread(target=_identify_worker, daemon=True).start()
    
    def get_identification_result(self, request_id):
        """ambil hasil identifikasi jika sudah selesai"""
        with self.results_lock:
            return self.identification_results.pop(request_id, None)
    
    def _identify_face_sync(self, face_img):
        """identifikasi face synchronous - dipanggil di background thread"""
        if not self.face_database:
            return "unknown", None, 0
        
        if not self._init_deepface():
            return "unknown", None, 0
        
        try:
            # save temp image
            temp_path = os.path.join(self.known_faces_dir, "_temp_identify.jpg")
            cv2.imwrite(temp_path, face_img)
            
            best_code = "unknown"
            best_name = None
            best_distance = 999
            
            # compare dengan semua face di database
            for code, data in self.face_database.items():
                for db_img_path in data["images"]:
                    if not os.path.exists(db_img_path):
                        continue
                    
                    try:
                        result = self.deepface.verify(
                            img1_path=temp_path,
                            img2_path=db_img_path,
                            model_name='VGG-Face',
                            detector_backend='opencv',
                            enforce_detection=False
                        )
                        
                        distance = result['distance']
                        
                        if distance < best_distance:
                            best_distance = distance
                            if result['verified']:
                                best_code = code
                                best_name = data["name"]
                    
                    except Exception as e: 
                        continue
            
            # cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # hitung confidence
            if best_code == "unknown":
                confidence = 0
            else:
                confidence = max(0, min(100, (0.6 - best_distance) / 0.6 * 100))
            
            return best_code, best_name, round(confidence, 2)
        
        except Exception as e: 
            print(f"error in face identification: {e}")
            return "unknown", None, 0
    
    def identify_face(self, face_img):
        """identifikasi face - wrapper untuk backward compatibility"""
        return self._identify_face_sync(face_img)
    
    def register_face(self, image, code, name):
        """register new face"""
        if not self._init_deepface():
            return False, "deepface not available"
        
        try: 
            # save temp image untuk verifikasi
            temp_path = os.path.join(self.known_faces_dir, "_temp_register.jpg")
            cv2.imwrite(temp_path, image)
            
            # verify face exists
            try:
                faces = self.deepface.extract_faces(
                    img_path=temp_path,
                    detector_backend='opencv',
                    enforce_detection=True
                )
                if len(faces) == 0:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return False, "no face detected"
                if len(faces) > 1:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return False, "multiple faces detected"
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False, f"face detection failed: {str(e)}"
            
            # cleanup temp
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # save image
            if code not in self.face_database:
                self.face_database[code] = {"name": name, "images": []}
            
            count = len(self.face_database[code]["images"])
            filename = f"{code}_{count + 1}.jpg"
            filepath = os.path.join(self.known_faces_dir, filename)
            cv2.imwrite(filepath, image)
            
            self.face_database[code]["images"].append(filepath)
            self.save_database()
            
            return True, f"face registered for {name} ({code})"
        
        except Exception as e:
            return False, str(e)
    
    def delete_face(self, code):
        """delete registered face"""
        if code not in self.face_database:
            return False, f"no data for code {code}"
        
        # delete images
        for filepath in self.face_database[code]["images"]:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        name = self.face_database[code]["name"]
        count = len(self.face_database[code]["images"])
        del self.face_database[code]
        self.save_database()
        
        return True, f"deleted {count} images for {name} ({code})"
    
    def list_faces(self):
        """list all registered faces"""
        faces_list = []
        for code, data in self.face_database.items():
            faces_list.append({
                'code': code,
                'name': data['name'],
                'images_count': len(data['images'])
            })
        
        return faces_list
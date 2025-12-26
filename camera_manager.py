import cv2
import numpy as np
import threading
import time
from collections import deque

class CameraManager:
    def __init__(self, buffer_size=5):
        self.camera = None
        self.camera_lock = threading.Lock()
        self.camera_available = False
        self.is_running = False
        
        # frame buffering untuk smooth streaming
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
    def init_camera(self):
        """inisialisasi kamera dengan fallback options"""
        print("🎥 initializing camera...")
        
        if self.camera is not None and self.camera.isOpened():
            print("✅ camera already initialized")
            return self.camera
        
        # konfigurasi kamera dengan prioritas
        camera_configs = [
            (0, cv2.CAP_V4L2),
            (0, cv2.CAP_ANY),
            (2, cv2.CAP_V4L2),
            (1, cv2.CAP_V4L2),
            (1, cv2.CAP_ANY),
        ]
        
        for idx, backend in camera_configs:
            try: 
                print(f"  trying camera: {idx} with backend: {backend}")
                cam = cv2.VideoCapture(idx, backend)
                
                if cam.isOpened():
                    ret, frame = cam.read()
                    if ret and frame is not None:
                        print(f"  ✅ camera opened: {idx}")
                        
                        # set optimal resolution untuk performance
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cam.set(cv2.CAP_PROP_FPS, 30)
                        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal buffer untuk low latency
                        
                        self.camera = cam
                        self.camera_available = True
                        return self.camera
                    else:
                        cam.release()
                else:
                    cam.release()
            except Exception as e:
                print(f"  failed: {e}")
                continue
        
        print("❌ no camera available")
        self.camera_available = False
        return None
    
    def read_frame(self):
        """baca frame dari kamera dengan locking minimal"""
        if self.camera is None or not self.camera.isOpened():
            return None
        
        with self.camera_lock:
            success, frame = self.camera.read()
            if success:
                return frame
        return None
    
    def get_latest_frame(self):
        """ambil frame terbaru dari buffer untuk streaming smooth"""
        with self.buffer_lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()
        return None
    
    def add_to_buffer(self, frame):
        """tambah frame ke buffer"""
        with self.buffer_lock:
            self.frame_buffer.append(frame.copy())
    
    def release(self):
        """release kamera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.camera_available = False
            print("✅ camera released")
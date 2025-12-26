import os
import json
import cv2
import numpy as np
from datetime import datetime
from collections import Counter

class AttendanceManager:
    def __init__(self, attendance_dir, window_size=8):
        self.attendance_dir = attendance_dir
        self. window_size = window_size
        self.attendance_today = {}
        
        os.makedirs(attendance_dir, exist_ok=True)
    
    def check_attendance_today(self, person_code, session_type):
        """Check if person already recorded attendance today for specific session"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{person_code}_{session_type}_{today}"
        
        # Always check from file, don't use cache
        # This ensures we get latest data after delete
        summary_path = os.path.join(self.attendance_dir, "attendance_summary.jsonl")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        record_date = record['timestamp'][: 10]
                        
                        if (record_date == today and 
                            record['person_code'] == person_code and 
                            record['attendance_type'] == session_type):
                            return True
        
        return False
    
    def save_attendance(self, recording_data, face_detector, drunk_detector):
        """Save attendance record"""
        session_id = recording_data['session_id']
        session_dir = os.path.join(self.attendance_dir, session_id)
        frames_dir = os.path.join(session_dir, "frames")
        
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save frames
        saved_frame_names = []
        for idx, frame in enumerate(recording_data['frames'], 1):
            frame_name = f"frame_{idx:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_frame_names.append(frame_name)
        
        # Save face
        face_image_name = "face_detected.jpg"
        for frame in recording_data['frames']:
            face_crop, _ = face_detector.detect_face(frame)
            if face_crop is not None:
                face_path = os.path.join(session_dir, face_image_name)
                cv2.imwrite(face_path, face_crop)
                break
        
        # Calculate windowed decisions
        windows = []
        for i in range(0, len(recording_data['labels']), self.window_size):
            window = recording_data['labels'][i: i + self.window_size]
            if len(window) == 0:
                continue
            majority = Counter(window).most_common(1)[0][0]
            windows.append(majority)
        
        # Calculate statistics
        avg_prob = float(np.mean(recording_data['probs'])) if recording_data['probs'] else 0.0
        median_prob = float(np.median(recording_data['probs'])) if recording_data['probs'] else 0.0
        
        # Final decision
        if len(windows) >= 2:
            final_decision = Counter(windows).most_common(1)[0][0]
        else:
            final_decision = "INCONCLUSIVE"
        
        # Create log
        log = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "person_code": recording_data['person_code'],
            "person_name": recording_data['person_name'],
            "attendance_type": recording_data['attendance_type'],
            "duration_sec": recording_data. get('duration', 7),
            "frames_used": len(recording_data['frames']),
            "threshold":  drunk_detector.threshold,
            "average_prob_sober": round(avg_prob, 3),
            "median_prob_sober": round(median_prob, 3),
            "window_decisions": windows,
            "final_decision": final_decision,
            "quality_gate": {
                "min_blur": drunk_detector.min_blur,
                "min_brightness": drunk_detector.min_brightness
            },
            "frames_dir": "frames/",
            "saved_frames": saved_frame_names,
            "face_image":  face_image_name
        }
        
        # Save log
        log_path = os. path.join(session_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        
        # Save to summary
        summary_path = os. path.join(self.attendance_dir, "attendance_summary.jsonl")
        with open(summary_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        
        print(f"✅ Attendance saved:  {session_id}")
        
        return log
    
    def get_history(self, filter_date=None):
        """Get attendance history"""
        summary_path = os.path.join(self.attendance_dir, "attendance_summary.jsonl")
        
        if not os.path.exists(summary_path):
            return []
        
        history = []
        with open(summary_path, "r") as f:
            for line in f: 
                if line.strip():
                    record = json.loads(line)
                    
                    if filter_date: 
                        record_date = record['timestamp'][: 10]
                        if record_date != filter_date:
                            continue
                    
                    history.append(record)
        
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history
    
    def delete_record(self, session_id):
        """Delete attendance record"""
        import shutil
        
        # Delete session folder
        session_dir = os. path.join(self.attendance_dir, session_id)
        if os.path.exists(session_dir):
            shutil. rmtree(session_dir)
        
        # Remove from summary
        summary_path = os.path.join(self. attendance_dir, "attendance_summary.jsonl")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                lines = f.readlines()
            
            with open(summary_path, "w") as f:
                for line in lines:
                    if line.strip():
                        record = json.loads(line)
                        if record['session_id'] != session_id:
                            f.write(line)
        
        # Clear cache (remove the cache functionality since we check file directly now)
        self.attendance_today.clear()
        
        return True
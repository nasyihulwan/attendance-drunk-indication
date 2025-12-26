from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from datetime import datetime
import time
import threading
from collections import deque, Counter
import os
from queue import Queue, Empty

# import modules
from camera_manager import CameraManager
from face_recognition import FaceRecognition
from drunk_detection import DrunkDetection
from attendance_manager import AttendanceManager

app = Flask(__name__)

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "training", "drunk_sober_mobilenet.h5")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
DATABASE_FILE = os.path.join(BASE_DIR, "face_database.json")

CAPTURE_DURATION = 7
WINDOW_SIZE = 8
FACE_DETECTION_INTERVAL = 2.0  # deteksi face setiap 2 detik untuk mengurangi beban

# =========================
# INITIALIZE MODULES
# =========================
print("="*50)
print("🚀 initializing modules...")
print("="*50)

camera_manager = CameraManager(buffer_size=3)
print("✅ camera manager initialized")

face_recognition = FaceRecognition(KNOWN_FACES_DIR, DATABASE_FILE)
print("✅ face recognition initialized")

drunk_detector = DrunkDetection(MODEL_PATH)

attendance_manager = AttendanceManager(ATTENDANCE_DIR, WINDOW_SIZE)
print("✅ attendance manager initialized")

print("="*50)
print("✅ all modules initialized successfully")
print("="*50)

# =========================
# GLOBAL VARIABLES
# =========================
# camera thread control
camera_thread_started = False
camera_thread = None
stop_camera_thread = False
pause_camera_processing = False

# frame state
frame_count = 0
session_start = None
current_frame = None

# session state
current_session_type = None
auto_mode_active = False

# face detection state (dengan async processing)
last_face_detection_time = 0
face_identification_in_progress = False
identification_request_id = None
detected_person_code = None
detected_person_name = None
detected_person_confidence = 0

# recording state
is_recording = False
recording_data = {
    'frames': [],
    'probs': [],
    'labels': [],
    'start_time': None,
    'session_id': None,
    'person_code': None,
    'person_name': None
}

# messaging
last_message = None

# processing queues untuk async operations
face_detection_queue = Queue(maxsize=2)
drunk_prediction_queue = Queue(maxsize=5)

# mode flags untuk membedakan attendance vs register
is_register_mode = False  # flag untuk mode register (plain camera)

# =========================
# ASYNC FACE DETECTION WORKER
# =========================
def face_detection_worker():
    """background worker untuk face detection & identification"""
    global detected_person_code, detected_person_name, detected_person_confidence
    global face_identification_in_progress, identification_request_id
    
    print("🔄 face detection worker started")
    
    while not stop_camera_thread:
        try:
            # ambil frame dari queue
            frame_data = face_detection_queue.get(timeout=0.5)
            if frame_data is None:
                continue
            
            frame = frame_data['frame']
            request_id = frame_data['request_id']
            
            # step 1: detect face (cepat)
            face_crop, face_rect = face_recognition.detect_face(frame)
            
            if face_crop is None:
                detected_person_code = None
                detected_person_name = None
                detected_person_confidence = 0
                face_identification_in_progress = False
                continue
            
            # step 2: identify person (lambat, tapi di background thread)
            code, name, confidence = face_recognition.identify_face(face_crop)
            
            if code == "unknown" or confidence < 50:
                detected_person_code = None
                detected_person_name = None
                detected_person_confidence = 0
            else:
                detected_person_code = code
                detected_person_name = name
                detected_person_confidence = confidence
                print(f"✅ detected: {name} ({code}) - {confidence}%")
            
            face_identification_in_progress = False
            
        except Empty:
            continue
        except Exception as e:
            print(f"error in face detection worker: {e}")
            face_identification_in_progress = False
            time.sleep(0.1)
    
    print("✅ face detection worker stopped")

# =========================
# ATTENDANCE WORKFLOW - SIMPLIFIED
# =========================
def auto_attendance_workflow():
    """
    Simplified workflow untuk menghindari race condition
    """
    global last_face_detection_time, face_identification_in_progress
    global identification_request_id, last_message
    
    current_time = time.time()
    
    # Rate limiting
    if current_time - last_face_detection_time < FACE_DETECTION_INTERVAL: 
        return
    
    # Skip if already processing
    if face_identification_in_progress:
        return
    
    # Check if we have a frame
    if current_frame is None:
        return
    
    # Send frame for async identification
    if not face_detection_queue.full():
        identification_request_id = f"req_{int(current_time * 1000)}"
        face_detection_queue.put({
            'frame': current_frame. copy(),
            'request_id': identification_request_id
        })
        face_identification_in_progress = True
        last_face_detection_time = current_time
    
    # Check identification result
    if detected_person_code is not None and detected_person_name is not None:
        code = detected_person_code
        name = detected_person_name
        
        # Check if already recorded
        already_recorded = attendance_manager.check_attendance_today(code, current_session_type)
        
        if already_recorded:
            last_message = {
                'text': f'⚠️ {name} ({code}) already recorded {current_session_type. upper()} today',
                'type': 'warning'
            }
            print(f"⏭️ {name} already recorded today")
            last_face_detection_time = current_time + 5
            return
        
        # Start drunk detection
        print(f"🎬 starting drunk detection for {name} ({code})")
        start_drunk_detection_recording(code, name)

def start_drunk_detection_recording(person_code, person_name):
    """Start recording - RESET arrays properly"""
    global is_recording, recording_data, last_message
    
    if is_recording:
        return
    
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    
    # CRITICAL: Initialize with EMPTY arrays
    recording_data = {
        'frames': [],  # Empty list
        'probs': [],   # Empty list
        'labels':  [],  # Empty list
        'start_time': time.time(),
        'session_id': session_id,
        'person_code': person_code,
        'person_name': person_name,
        'attendance_type': current_session_type,
        'duration':  CAPTURE_DURATION
    }
    
    is_recording = True
    
    last_message = {
        'text': f'🎬 recording started for {person_name} ({person_code})',
        'type': 'info'
    }
    
    print(f"🎬 recording:  {person_name} ({person_code}) for {current_session_type.upper()}")
    print(f"📝 recording_data initialized: {len(recording_data['frames'])} frames")
    
    # Auto-stop after duration
    def auto_stop_and_save():
        time.sleep(CAPTURE_DURATION)
        save_attendance_log()
    
    threading.Thread(target=auto_stop_and_save, daemon=True).start()

def save_attendance_log():
    """Save attendance - with debug logging"""
    global is_recording, recording_data, detected_person_code
    global detected_person_name, detected_person_confidence, last_message
    
    if not is_recording:
        print("⚠️ save_attendance_log called but not recording")
        return
    
    is_recording = False
    
    try:
        # Debug:  Check what we captured
        frames_count = len(recording_data['frames'])
        probs_count = len(recording_data['probs'])
        labels_count = len(recording_data['labels'])
        
        print(f"📊 saving attendance:")
        print(f"   frames:   {frames_count}")
        print(f"   probs:    {probs_count}")
        print(f"   labels:  {labels_count}")
        
        if frames_count == 0:
            print("❌ no frames captured!")
            last_message = {
                'text': '❌ no frames captured during recording',
                'type': 'error'
            }
            return
        
        # Save attendance
        log = attendance_manager.save_attendance(
            recording_data,
            face_recognition,
            drunk_detector
        )
        
        person_name = recording_data['person_name']
        person_code = recording_data['person_code']
        session_type = recording_data['attendance_type']. upper()
        decision = log['final_decision']
        
        last_message = {
            'text': f'✅ {person_name} ({person_code}) - {session_type} recorded | status: {decision}',
            'type': 'success'
        }
        
        print(f"✅ attendance saved: {person_name} ({person_code}) - {session_type}")
        
        # Reset detection
        detected_person_code = None
        detected_person_name = None
        detected_person_confidence = 0
        
    except Exception as e: 
        print(f"❌ error saving attendance: {e}")
        import traceback
        traceback.print_exc()
        last_message = {
            'text': f'❌ error saving attendance:  {str(e)}',
            'type': 'error'
        }

def save_attendance_log():
    """save attendance setelah recording selesai"""
    global is_recording, recording_data, detected_person_code
    global detected_person_name, detected_person_confidence, last_message
    
    if not is_recording:
        return
    
    is_recording = False
    
    try:
        log = attendance_manager.save_attendance(
            recording_data,
            face_recognition,
            drunk_detector
        )
        
        person_name = recording_data['person_name']
        person_code = recording_data['person_code']
        session_type = recording_data['attendance_type'].upper()
        decision = log['final_decision']
        
        last_message = {
            'text': f'✅ {person_name} ({person_code}) - {session_type} recorded | status: {decision}',
            'type': 'success'
        }
        
        print(f"✅ attendance saved: {person_name} ({person_code}) - {session_type}")
        
        # reset detection state
        detected_person_code = None
        detected_person_name = None
        detected_person_confidence = 0
        
    except Exception as e:
        print(f"❌ error saving attendance: {e}")
        last_message = {
            'text': f'❌ error saving attendance: {str(e)}',
            'type': 'error'
        }

# =========================
# CAMERA PROCESSING THREAD
# =========================
def process_camera():
    """
    Background thread untuk process camera frames
    FIXED: Pastikan frames masuk ke recording dengan benar
    """
    global current_frame, frame_count, session_start, stop_camera_thread
    global auto_mode_active, is_recording, recording_data, pause_camera_processing
    
    print("🎬 camera processing thread started")
    
    session_start = datetime.now()
    cam = None
    
    while not stop_camera_thread: 
        try:  
            # Check if paused
            if pause_camera_processing:
                if cam is not None and cam.isOpened():
                    cam.release()
                    cam = None
                    print("🔹 camera released (paused)")
                
                time.sleep(0.5)
                continue
            
            # Initialize camera if needed
            if cam is None:  
                cam = camera_manager.init_camera()
                
                if cam is None:
                    print("⚠️ camera not available")
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "NO CAMERA DETECTED", (100, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    current_frame = placeholder
                    camera_manager.add_to_buffer(placeholder)
                    time.sleep(1)
                    continue
                
                print("✅ camera initialized and started")
            
            # Read frame from camera
            frame = camera_manager.read_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # CRITICAL: Update current_frame BEFORE using it
            current_frame = frame. copy()
            
            # Add to buffer for smooth streaming
            camera_manager.add_to_buffer(frame)
            
            # DRUNK DETECTION RECORDING - Process SYNCHRONOUSLY
            # This is the FIX - process recording in main camera thread
            if is_recording: 
                # Check quality
                if not drunk_detector.is_low_quality(frame):
                    # Predict drunk/sober
                    label, prob_sober = drunk_detector.predict(frame)
                    
                    if label != "ERROR":
                        # CRITICAL:  Append frame, prob, and label together
                        recording_data['frames'].append(frame. copy())
                        recording_data['probs'].append(prob_sober)
                        recording_data['labels'].append(label)
                        
                        # Debug log
                        if len(recording_data['frames']) % 30 == 0:  # Every 1 second
                            print(f"📹 recording: {len(recording_data['frames'])} frames captured")
            
            # Auto attendance workflow (async - lightweight)
            if auto_mode_active and current_session_type is not None and not is_recording:
                auto_attendance_workflow()
            
            # Maintain 30fps
            time.sleep(0.033)
            
        except Exception as e:
            print(f"❌ error in camera loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    print("🛑 stopping camera thread...")
    if cam is not None:  
        cam.release()
    camera_manager.release()
    print("✅ camera released")

def start_camera_thread():
    """start camera processing thread"""
    global camera_thread, camera_thread_started, stop_camera_thread
    
    if not camera_thread_started:
        print("🚀 starting camera thread...")
        stop_camera_thread = False
        camera_thread = threading.Thread(target=process_camera, daemon=True)
        camera_thread.start()
        
        # start face detection worker
        face_worker = threading.Thread(target=face_detection_worker, daemon=True)
        face_worker.start()
        
        camera_thread_started = True
        time.sleep(1.5)

# =========================
# VIDEO STREAMING
# =========================
def generate_frames():
    """
    generate frames untuk streaming
    menggunakan buffer untuk smooth playback
    """
    global current_frame, is_recording, recording_data
    global detected_person_code, detected_person_name, current_session_type
    
    # tunggu camera ready
    while current_frame is None: 
        time.sleep(0.1)
    
    while True:
        # ambil frame dari buffer (lebih smooth)
        frame = camera_manager.get_latest_frame()
        
        if frame is None or pause_camera_processing:
            time.sleep(0.033)
            continue
        
        # overlay: face detection box
        face_crop, face_rect = face_recognition.detect_face(frame)
        if face_rect is not None:
            (x, y, w, h) = face_rect
            
            if detected_person_code is not None and detected_person_name is not None:
                # identified person - green box
                color = (0, 255, 0)
                label = f"{detected_person_name} ({detected_person_code})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x+6, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            else:
                # unknown face - purple box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        # overlay: recording indicator
        if is_recording:
            elapsed = time.time() - recording_data['start_time']
            remaining = max(0, CAPTURE_DURATION - elapsed)
            
            cv2.putText(frame, "RECORDING - DRUNK DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"time: {remaining:.1f}s", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        else:
            # overlay: session info
            if current_session_type is not None: 
                session_text = f"session: {current_session_type.upper()}"
                session_color = (0, 255, 0) if current_session_type == 'in' else (0, 0, 255)
                cv2.putText(frame, session_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, session_color, 2)
                
                mode_text = "AUTO MODE - stand in front of camera"
                cv2.putText(frame, mode_text, (10, frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # overlay: frame counter
        cv2.putText(frame, f"frame: {frame_count}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # smooth 30fps streaming
        time.sleep(0.033)

def generate_frames_register():
    """
    generate frames untuk halaman register
    lebih simple, tanpa overlay attendance
    """
    global pause_camera_processing
    
    # start camera jika belum started
    if not camera_thread_started:
        start_camera_thread()
    
    # resume camera untuk register
    pause_camera_processing = False
    
    # tunggu camera ready
    while current_frame is None:
        time.sleep(0.1)
    
    while True:
        # ambil frame dari buffer
        frame = camera_manager.get_latest_frame()
        
        if frame is None:
            time.sleep(0.033)
            continue
        
        frame_copy = frame.copy()
        
        # overlay: face detection box untuk register
        face_crop, face_rect = face_recognition.detect_face(frame_copy)
        if face_rect is not None:
            (x, y, w, h) = face_rect
            # green box untuk indicate face detected
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_copy, "FACE DETECTED", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # instruction text jika tidak ada face
            cv2.putText(frame_copy, "Position your face in frame", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # encode frame
        ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # smooth 30fps streaming
        time.sleep(0.033)

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('attendance.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_register')
def video_feed_register():
    """video feed khusus untuk halaman register (tanpa overlay)"""
    return Response(generate_frames_register(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
                   
@app.route('/api/stop_register_camera', methods=['POST'])
def stop_register_camera():
    """Stop register camera session"""
    global pause_camera_processing
    
    # Pause camera processing
    pause_camera_processing = True

    print("📹 Register camera stopped")
    
    return jsonify({
        'success': True,
        'message': 'Register camera stopped'
    })


# =========================
# SESSION & AUTO MODE ROUTES
# =========================
@app.route('/api/set_session', methods=['POST'])
def set_session():
    """set current session type (in/out) dan activate auto mode"""
    global current_session_type, auto_mode_active, last_message, pause_camera_processing
    
    data = request.json
    session_type = data.get('session_type')
    
    if session_type not in ['in', 'out']:
        return jsonify({'error': 'invalid session type'}), 400
    
    current_session_type = session_type
    auto_mode_active = True
    pause_camera_processing = False  # resume camera
    
    last_message = None
    
    # start camera thread jika belum started
    if not camera_thread_started:
        start_camera_thread()
    
    print(f"✅ session started: {session_type.upper()} - camera resumed")
    
    return jsonify({
        'success': True,
        'session_type': session_type,
        'auto_mode': auto_mode_active
    })

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring_api():
    """stop auto mode monitoring dan pause camera"""
    global auto_mode_active, current_session_type, detected_person_code
    global detected_person_name, detected_person_confidence, is_recording
    global pause_camera_processing
    
    # pause camera processing
    pause_camera_processing = True
    
    # deactivate auto mode
    auto_mode_active = False
    current_session_type = None
    
    # reset detection state
    detected_person_code = None
    detected_person_name = None
    detected_person_confidence = 0
    
    # jika sedang recording, stop
    if is_recording:
        is_recording = False
    
    print("ℹ️ monitoring stopped - camera paused and will be released")
    
    return jsonify({
        'success': True,
        'message': 'monitoring stopped, camera released'
    })

@app.route('/api/auto_mode_status')
def auto_mode_status():
    """get current auto mode status"""
    global last_message
    
    status = {
        'auto_mode': auto_mode_active,
        'session_type': current_session_type,
        'is_recording': is_recording,
        'detected_person': None,
        'last_message': last_message
    }
    
    if detected_person_code is not None and detected_person_name is not None:
        status['detected_person'] = {
            'code': detected_person_code,
            'name': detected_person_name,
            'confidence': detected_person_confidence
        }
    
    if is_recording:
        elapsed = time.time() - recording_data['start_time']
        status['remaining'] = max(0, CAPTURE_DURATION - elapsed)
        status['frames_captured'] = len(recording_data['frames'])
    
    # clear message setelah sending
    if last_message:
        temp_message = last_message
        last_message = None
        status['last_message'] = temp_message
    
    return jsonify(status)

# =========================
# FACE REGISTRATION ROUTES
# =========================
@app.route('/api/train', methods=['POST'])
def train_face():
    """register face baru ke database"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image'}), 400
        
        name = request.form.get('name')
        code = request.form.get('code')
        
        if not name or not code: 
            return jsonify({'error': 'name and code required'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        success, message = face_recognition.register_face(image, code, name)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture_frame', methods=['GET'])
def capture_frame():
    """capture frame saat ini untuk registration"""
    global current_frame, pause_camera_processing
    
    # pastikan camera running
    if not camera_thread_started:
        start_camera_thread()
    
    # resume camera jika paused
    pause_camera_processing = False
    
    # tunggu frame ready
    timeout = 5
    start_time = time.time()
    while current_frame is None:
        if time.time() - start_time > timeout:
            return jsonify({'error': 'camera timeout'}), 500
        time.sleep(0.1)
    
    try:
        frame = current_frame.copy()
        
        # detect face
        face_crop, face_rect = face_recognition.detect_face(frame)
        
        if face_crop is None:
            return jsonify({'error': 'no face detected'}), 400
        
        # encode face image
        ret, buffer = cv2.imencode('.jpg', face_crop)
        if not ret:
            return jsonify({'error': 'failed to encode image'}), 500
        
        # return as base64
        import base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/faces')
def list_faces():
    faces = face_recognition.list_faces()
    return jsonify({'total': len(faces), 'faces': faces})

@app.route('/api/delete_face/<code>', methods=['DELETE'])
def delete_face(code):
    success, message = face_recognition.delete_face(code)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 404

# =========================
# ATTENDANCE HISTORY ROUTES
# =========================
@app.route('/get_attendance_history')
def get_attendance_history():
    filter_date = request.args.get('date', None)
    history = attendance_manager.get_history(filter_date)
    return jsonify(history)

@app.route('/delete_attendance/<session_id>', methods=['DELETE'])
def delete_attendance(session_id):
    success = attendance_manager.delete_record(session_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 500

# =========================
# RUN
# =========================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 automatic attendance system")
    print("="*50)
    print(f"📁 base: {BASE_DIR}")
    print(f"🤖 model: {MODEL_PATH}")
    print(f"📋 attendance: {ATTENDANCE_DIR}")
    print(f"👤 known faces: {KNOWN_FACES_DIR}")
    print("\n📋 workflow:")
    print("  1️⃣ face recognition - detect & identify")
    print("  2️⃣ drunk detection - 7 second recording")
    print("  3️⃣ save log - complete attendance record")
    print("\n⚡ optimizations:")
    print("  • async face identification")
    print("  • pre-loaded models")
    print("  • frame buffering")
    print("  • reduced detection interval")
    print("="*50 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 shutting down...")
        stop_camera_thread = True
        time.sleep(1)
        print("✅ cleanup complete")
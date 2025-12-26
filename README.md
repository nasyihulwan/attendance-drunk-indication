# 📋 Automatic Attendance System with Drunk Detection

An intelligent attendance system combining **Face Recognition** and **Drunk Detection** using Deep Learning for automatic attendance recording with safety monitoring.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ⚠️ Disclaimer

**For educational purposes only.** Drunk detection provides **indicative results** - **NOT** for legal evidence. Use responsibly and ethically.

---

## 🎯 Features

- ✅ Real-time Face Recognition with DeepFace (VGG-Face)
- ✅ AI Drunk Detection using MobileNet
- ✅ Automatic Attendance Logging (Clock IN/OUT)
- ✅ Multi-image Registration (Webcam + Upload)
- ✅ Drag & Drop File Upload
- ✅ Attendance History with Date Filtering
- ✅ Windowed Decision Making (8-frame majority voting)

---

## 📊 Dataset Sources

- **Sober:** https://universe.roboflow.com/new-workspace-8swzs/sober
- **Drunk:** https://universe.roboflow.com/prerak/drunk-detection-r4oat

_Thanks to Roboflow community for public datasets._

---

## 🛠️ Tech Stack

**Backend:** Python, Flask, OpenCV, TensorFlow, DeepFace, NumPy  
**Frontend:** HTML5, CSS3, Vanilla JavaScript  
**AI Models:** VGG-Face (face recognition), MobileNet (drunk detection)

---

## 📋 Prerequisites

- Python 3.8+
- Webcam
- 4GB RAM minimum
- 2GB free disk space

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/attendance-drunk-detection.git
cd attendance-drunk-detection

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**

```txt
Flask==2.3.0
opencv-python==4.8.0
tensorflow==2.13.0
deepface==0.0.79
numpy==1.24.3
Pillow==10.0.0
```

**Project Structure:**

```
attendance-drunk-indication/
├── app.py
├── camera_manager.py
├── face_recognition.py
├── drunk_detection.py
├── attendance_manager.py
├── templates/
│   ├── attendance.html
│   └── register.html
├── training/
│   └── drunk_sober_mobilenet.h5
├── known_faces/
├── attendance/
```

---

## 🎮 Usage

### Start Application

```bash
python app.py
```

Open browser: `http://localhost:5000`

### Register Person

1. Go to "Register New Face"
2. **Capture** from webcam OR **Upload** images (multiple supported)
3. Enter Person Code (e.g., P001) and Name
4. Click "Register All Images"

### Record Attendance

1. Select session (🟢 Clock IN / 🔴 Clock OUT)
2. Click "START MONITORING"
3. Stand in front of camera
4. System auto-detects → 7-sec drunk detection → saves attendance

---

## 🔧 Configuration

**Camera Settings** (`camera_manager.py`):

```python
cam. set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**Detection Parameters** (`app.py`):

```python
CAPTURE_DURATION = 7  # Recording duration (seconds)
WINDOW_SIZE = 8       # Windowed decision frames
```

**Drunk Detection** (`drunk_detection.py`):

```python
self.threshold = 0.60      # Sober threshold
self.min_blur = 50         # Quality:  blur
self.min_brightness = 60   # Quality: brightness
```

---

## 📊 How It Works

**Face Recognition:**

```
Camera → Face Detection → DeepFace Embedding → Similarity Check → Identification
```

**Drunk Detection:**

```
7s Recording → Quality Filter → MobileNet Prediction → Windowed Voting → Decision
```

**Windowed Decision:** Divides frames into 8-frame windows → majority vote per window → final decision from window majorities (reduces false positives).

---

## 🐛 Troubleshooting

**Camera not detected:**

```bash
# Linux
sudo usermod -a -G video $USER

# Test
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**DeepFace model download (first run):**

```python
from deepface import DeepFace
DeepFace.build_model("VGG-Face")
```

**Port conflict:**

```python
# Change in app.py
app.run(host='0.0.0.0', port=5001)
```

---

## 📄 License

Permission is granted to use, modify, and distribute this software freely.

---

## 🙏 Acknowledgments

- Indonesia University of Education
- Roboflow Community (datasets)
- DeepFace, TensorFlow, OpenCV

---

## 📞 Contact

**Muhammad Nasyih Ulwan**  
Indonesia University of Education

**GitHub:** https://github.com/nasyihulwan
**Email:** nasyihulwan@upi.edu

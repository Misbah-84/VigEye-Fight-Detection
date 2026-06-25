# VigEye: AI-Powered Intelligent Surveillance System

VigEye is a multi-tasking, deep-learning-based intelligent surveillance framework designed to automate threat monitoring in real-time CCTV feeds. By integrating temporal behavior analysis with high-precision object detection, VigEye simultaneously scans for physical conflicts (fights), brandished weapons, and environmental hazards (smoke/fire) to trigger instant alerts.

## 🌟 Key Features

*   **Real-Time Fight Detection:** Employs temporal frame analysis to distinguish aggressive physical altercations from benign, everyday human interactions.
*   **Weapon Localization:** Utilizes state-of-the-art object detection to locate firearms, knives, and other hazardous instruments down to small pixel resolutions.
*   **Early Smoke & Fire Detection:** Continuously monitors environmental pixel variations to catch smoke plumes and fire hazards before they escalate.
*   **Edge Optimization:** Lightweight inference pipeline built to support high-FPS streaming on edge hardware and centralized servers.
*   **Alert Automation:** Structured metadata output ready for integration with real-time dashboards or webhooks.

---

## 🏗️ System Architecture
[ CCTV Video Input / RTSP Stream ]
│
▼
[ Frame Preprocessing Layer ]
│
┌────────┼────────┐
▼        ▼        ▼
[Fight]   [Weapon] [Smoke]
(LSTM/3D)  (YOLO)   (CNN)
│        │        │
└────────┼────────┘
▼
[ Alert & Logic Aggregator ] ──> [ Live UI Overlay / Telegram API / Webhook ]


---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher
*   NVIDIA GPU + CUDA Toolkit (Highly Recommended for real-time FPS)

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Misbah-84/VigEye-Fight-Detection.git](https://github.com/Misbah-84/VigEye-Fight-Detection.git)


cd VigEye-Fight-Detection
pip install -r requirements.txt
2. Model Weights
Place your trained deep learning weights (.pt, .onnx, or .h5 files) inside the weights/ directory:

weights/fight_detection_model.pt

weights/weapon_detection_model.pt

weights/smoke_detection_model.pt

💻 Usage
Run the main inference pipeline on a local video file, a live webcam feed, or an IP Camera RTSP stream.

Run on a Video File
Bash
python main.py --source data/test_video.mp4 --conf 0.5
Run on Live Webcam Feed
Bash
python main.py --source 0 --show-preview

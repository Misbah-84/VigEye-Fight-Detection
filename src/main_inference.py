import torch
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# NOTE: Place your video file in the 'data/raw/' folder.
# Change this path to test the file. Use 0 for webcam testing.
SOURCE_PATH = 'data/raw/long_test.mp4' 
CONFIDENCE_THRESHOLD = 0.10  # Set low for robust detection; adjust to 0.50 later.
# ---------------------

def run_tracker(source_path, conf_threshold):
    """Initializes YOLO, runs tracking, and displays results."""
    
    # --- 1. SYSTEM AND DEVICE SETUP ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- System Check: Using {device.upper()} ---")

    # Load YOLOv8 Nano model for fast inference
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model. Check installation. Details: {e}")
        return

    # --- 2. VIDEO INPUT ---
    cap = cv2.VideoCapture(source_path, cv2.CAP_DSHOW) if source_path == 0 else cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source at '{source_path}'.")
        return

    print(f"Tracking started. Press 'q' to quit.")
    
    # --- 3. THE REAL-TIME PROCESSING LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # Standardize frame size for model consistency and speed
            frame = cv2.resize(frame, (640, 480))
            
            # Run YOLO Tracking (classes=0 means 'person')
            results = model.track(
                frame, 
                persist=True, 
                device=device, 
                classes=0, 
                conf=conf_threshold, 
                verbose=False
            )
            
            # --- 4. VISUALIZATION AND EXIT ---
            # .plot() draws boxes and unique tracking IDs on the frame
            annotated_frame = results[0].plot() 
            
            cv2.imshow("VigEye: Person Tracking", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # --- 5. CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracker(SOURCE_PATH, CONFIDENCE_THRESHOLD)
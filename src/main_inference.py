import torch
import cv2
from ultralytics import YOLO

#this is the path of video which i saved in my laptop just for testing
SOURCE_PATH = 'data/raw/long_test.mp4' 
CONFIDENCE_THRESHOLD = 0.10  


def run_tracker(source_path, conf_threshold):
    """Initializes YOLO, runs tracking, and displays results."""
    
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- System Check: Using {device.upper()} ---")

   
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model. Check installation. Details: {e}")
        return

  
    cap = cv2.VideoCapture(source_path, cv2.CAP_DSHOW) if source_path == 0 else cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source at '{source_path}'.")
        return

    print(f"Tracking started. Press 'q' to quit.")
    
   
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
           
            frame = cv2.resize(frame, (640, 480))
            
           
            results = model.track(
                frame, 
                persist=True, 
                device=device, 
                classes=0, 
                conf=conf_threshold, 
                verbose=False
            )
           
            annotated_frame = results[0].plot() 
            
            cv2.imshow("VigEye: Person Tracking", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

 
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracker(SOURCE_PATH, CONFIDENCE_THRESHOLD)
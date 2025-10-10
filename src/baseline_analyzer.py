import cv2
import numpy as np
from ultralytics import YOLO
import os


DATA_DIR = 'data/raw/RWF-2000_Clips'
YOLO_MODEL = YOLO('yolov8n.pt') 
DETECTION_THRESHOLD = 0.50  
VELOCITY_THRESHOLD = 5.0    
PROXIMITY_THRESHOLD = 80    
# ---------------------

def calculate_distance(box1, box2):
    """Calculates Euclidean distance between the center of two bounding boxes."""
   
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    return np.linalg.norm(center1 - center2)

def run_heuristic_baseline(video_path):
    """Runs a simple rule-based fight detector on a single video."""
    cap = cv2.VideoCapture(video_path)
    
   
    prev_centers = {} 
    
    
    fight_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
       
        results = YOLO_MODEL.track(frame, persist=True, classes=0, conf=DETECTION_THRESHOLD, verbose=False)
        tracked_objects = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, track_id, conf, class_id]

        current_centers = {}
        
       
        person_detections = [(d[0:4], int(d[4])) for d in tracked_objects if int(d[6]) == 0] # Filter for 'person' (class ID 0)
        
       
        is_fight_frame = False
        
        for i in range(len(person_detections)):
            (box_i, id_i) = person_detections[i]
            
            # Calculate current center for velocity check
            center_i = np.array([(box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2])
            current_centers[id_i] = center_i
            
           
            for j in range(i + 1, len(person_detections)):
                (box_j, id_j) = person_detections[j]
                
                # Check 3a: PROXIMITY
                distance = calculate_distance(box_i, box_j)
                if distance < PROXIMITY_THRESHOLD:
                    
                   
                    if id_i in prev_centers and id_j in prev_centers:
                        vel_i = np.linalg.norm(center_i - prev_centers[id_i])
                        vel_j = np.linalg.norm(center_j - prev_centers[id_j])
                        avg_vel = (vel_i + vel_j) / 2
                        
                        if avg_vel > VELOCITY_THRESHOLD:
                           
                            is_fight_frame = True
                            break 
            
            if is_fight_frame:
                fight_counter += 1
                break
        
        
        prev_centers = current_centers
        
        
        if fight_counter >= 5:
            cap.release()
            return True 

    cap.release()
    return False


if __name__ == '__main__':
    
  
    test_results = {}
    
   
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(('.mp4', '.avi')):
            video_path = os.path.join(DATA_DIR, file_name)
            
            
            
           
            prediction = run_heuristic_baseline(video_path)
            
           
            test_results[file_name] = prediction
            print(f"Video: {file_name} -> Heuristic Prediction: {prediction}")
    
    
    print("\nHeuristic test complete. Now calculate F1 Score in a separate notebook/script.")
import cv2
import numpy as np
import os
import glob

# NOTE: Use forward slashes (/) or double backslashes (\\) for Windows paths.
DATA_DIR = 'D:/vigeye/data/raw/RWF-2000_Clips'
OUTPUT_DIR = 'data/processed/flow_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev=gray1, 
        next=gray2, 
        flow=None, 
        pyr_scale=0.5, 
        levels=3, 
        winsize=15, 
        iterations=3, 
        poly_n=5, 
        poly_sigma=1.2, 
        flags=0
    )
    return flow

# --- Find video files using recursive search (glob) ---
def process_all_videos(data_dir, output_dir):
    video_files = []
    

    video_files.extend(glob.glob(os.path.join(data_dir, '**/*.mp4'), recursive=True))
    video_files.extend(glob.glob(os.path.join(data_dir, '**/*.avi'), recursive=True))

    if not video_files:
        print(f"ERROR: No video files found. Checked subdirectories for .mp4 and .avi.")
        return

    for video_path in video_files:
        filename = os.path.basename(video_path).rsplit('.', 1)[0]
        print(f"Processing: {filename}")
        
        cap = cv2.VideoCapture(video_path)
        
        ret, prev_frame = cap.read()
        if not ret: continue
        
        all_flow_frames = []
        
        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret: break
            
            # Resize frames to a consistent size (160x120 is common for flow input)
            current_frame_resized = cv2.resize(current_frame, (160, 120))
            prev_frame_resized = cv2.resize(prev_frame, (160, 120))
            
            flow_vectors = calculate_optical_flow(prev_frame_resized, current_frame_resized)
            
            all_flow_frames.append(flow_vectors)
            
            prev_frame = current_frame
        
        cap.release()

        if all_flow_frames:
            flow_array = np.stack(all_flow_frames)
            output_path = os.path.join(output_dir, f"{filename}_flow.npy")
            np.save(output_path, flow_array)
            print(f"    -> Saved array of shape {flow_array.shape} to {output_path}")

if __name__ == '__main__':
    process_all_videos(DATA_DIR, OUTPUT_DIR)
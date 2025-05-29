import cv2
import urllib.request
import numpy as np
import threading
import mediapipe as mp
import math
import time
import os
import pandas as pd
from datetime import datetime

# URLs of the two phones (modify with your IP addresses)
URL1 = "http://192.168.1.140:4747/video"
URL2 = "http://192.168.1.223:4747/video"

# Global variables for frames
frame1 = None
frame2 = None
lock1 = threading.Lock()
lock2 = threading.Lock()

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, 
                   model_complexity=1,
                   smooth_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)

# Create folder for storing angle data
data_folder = "angle_data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Initialize DataFrame for storing angle data
columns = [
    "timestamp",
    "cam1_left_knee_angle", "cam1_right_knee_angle",
    "cam1_left_ankle_angle", "cam1_right_ankle_angle", 
    "cam1_left_hip_angle", "cam1_right_hip_angle",
    "cam2_left_knee_angle", "cam2_right_knee_angle",
    "cam2_left_ankle_angle", "cam2_right_ankle_angle",
    "cam2_left_hip_angle", "cam2_right_hip_angle"
]
angle_data_df = pd.DataFrame(columns=columns)

# Create Excel filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"angle_data_{timestamp}.xlsx"
excel_path = os.path.join(data_folder, excel_filename)

def read_stream(url, frame_var, lock):
    global frame1, frame2
    stream = urllib.request.urlopen(url)
    bytes_data = b''

    while True:
        try:
            bytes_data += stream.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    with lock:
                        if frame_var == "frame1":
                            frame1 = frame
                        elif frame_var == "frame2":
                            frame2 = frame
        except Exception as e:
            print(f"Error in stream {url}: {e}")
            break

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    """
    if None in [a, b, c]:
        return None
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure within valid range
    
    # Calculate angle in degrees
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return angle

def get_landmark_coords(landmarks, landmark_id):
    """
    Extract x,y coordinates of a specific landmark
    Returns None if the landmark is not visible or not detected
    """
    if not landmarks or not landmarks.landmark:
        return None
    
    landmark = landmarks.landmark[landmark_id]
    if landmark.visibility < 0.5:  # Check if landmark is visible enough
        return None
    
    return [landmark.x, landmark.y]

def process_frame(frame, camera_id):
    """
    Process a frame to detect pose and calculate angles
    Returns processed frame and angles dictionary
    """
    if frame is None:
        return None, {}
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)
    
    # Convert back to BGR for displaying
    processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    angles = {}
    
    # Draw the pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            processed_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        # Get landmark coordinates
        landmarks = results.pose_landmarks
        
        # Ankle angles (angle between knee, ankle, and foot)
        left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
        left_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)
        left_foot = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value)
        
        right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        right_foot = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value)
        
        # Calculate knee angles (hip-knee-ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate ankle angles (knee-ankle-foot)
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot)
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot)
        
        # Calculate hip angles (shoulder-hip-knee)
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Store angles
        angles = {
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "left_ankle_angle": left_ankle_angle,
            "right_ankle_angle": right_ankle_angle,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle
        }
        
        # Display angles on frame
        h, w, _ = processed_frame.shape
        y_offset = 30
        font_scale = 0.6
        text_color = (255, 255, 255)
        
        for i, (angle_name, angle_value) in enumerate(angles.items()):
            if angle_value is not None:
                text = f"{angle_name}: {angle_value:.1f}°"
                cv2.putText(processed_frame, text, (10, y_offset + i * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    
    return processed_frame, angles

def save_to_excel(df, file_path):
    """
    Save the DataFrame to an Excel file
    """
    try:
        df.to_excel(file_path, index=False)
        print(f"Data saved to Excel file: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        try:
            # Try with a simpler CSV format as fallback
            csv_path = file_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"Data saved to CSV file (fallback): {csv_path}")
            return True
        except Exception as e2:
            print(f"Error saving to CSV: {e2}")
            return False

# Start the two threads
thread1 = threading.Thread(target=read_stream, args=(URL1, "frame1", lock1))
thread2 = threading.Thread(target=read_stream, args=(URL2, "frame2", lock2))
thread1.daemon = True
thread2.daemon = True
thread1.start()
thread2.start()

# Main display loop
last_save_time = time.time()
save_interval = 0.1  # Save data every 100ms
excel_save_interval = 5.0  # Save to Excel every 5 seconds
last_excel_save_time = time.time()

print(f"Starting angle detection. Data will be saved to {excel_path}")
print("Press 'q' to quit or 's' to save data immediately")

try:
    while True:
        with lock1:
            f1 = frame1.copy() if frame1 is not None else None
        with lock2:
            f2 = frame2.copy() if frame2 is not None else None

        # Process frames
        processed_frame1, angles1 = process_frame(f1, "cam1")
        processed_frame2, angles2 = process_frame(f2, "cam2")

        # Save angles data at regular intervals
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            if angles1 or angles2:  # Only save if we have some angle data
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Prepare row data
                row_data = {
                    "timestamp": timestamp_str,
                    "cam1_left_knee_angle": angles1.get("left_knee_angle", None),
                    "cam1_right_knee_angle": angles1.get("right_knee_angle", None),
                    "cam1_left_ankle_angle": angles1.get("left_ankle_angle", None),
                    "cam1_right_ankle_angle": angles1.get("right_ankle_angle", None),
                    "cam1_left_hip_angle": angles1.get("left_hip_angle", None),
                    "cam1_right_hip_angle": angles1.get("right_hip_angle", None),
                    "cam2_left_knee_angle": angles2.get("left_knee_angle", None),
                    "cam2_right_knee_angle": angles2.get("right_knee_angle", None),
                    "cam2_left_ankle_angle": angles2.get("left_ankle_angle", None),
                    "cam2_right_ankle_angle": angles2.get("right_ankle_angle", None),
                    "cam2_left_hip_angle": angles2.get("left_hip_angle", None),
                    "cam2_right_hip_angle": angles2.get("right_hip_angle", None)
                }
                
                # Add data to DataFrame
                angle_data_df = pd.concat([angle_data_df, pd.DataFrame([row_data])], ignore_index=True)
                last_save_time = current_time
        
        # Save to Excel file periodically
        if current_time - last_excel_save_time >= excel_save_interval:
            if len(angle_data_df) > 0:
                save_to_excel(angle_data_df, excel_path)
                last_excel_save_time = current_time

        # Display the frames
        if processed_frame1 is not None and processed_frame2 is not None:
            # Resize if necessary
            if processed_frame1.shape != processed_frame2.shape:
                processed_frame2 = cv2.resize(processed_frame2, 
                                           (processed_frame1.shape[1], processed_frame1.shape[0]))

            # Add labels to frames
            h, w, _ = processed_frame1.shape
            cv2.putText(processed_frame1, "Camera 1", (10, h - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame2, "Camera 2", (10, h - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            combined = cv2.hconcat([processed_frame1, processed_frame2])
            
            # Show recording indicator and data status
            indicator_text = f"Recording | {len(angle_data_df)} frames captured | Saving every {excel_save_interval}s"
            cv2.putText(combined, indicator_text, (10, 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Pose Detection - Press 'q' to quit, 's' to save now", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual save
            print("Manual save triggered...")
            if len(angle_data_df) > 0:
                save_to_excel(angle_data_df, excel_path)

finally:
    # Final save before closing
    if len(angle_data_df) > 0:
        print("Saving final data...")
        save_to_excel(angle_data_df, excel_path)
    
    # Clean up
    cv2.destroyAllWindows()
    pose.close()
    print("Application closed")
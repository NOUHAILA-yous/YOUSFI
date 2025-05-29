import cv2
import mediapipe as mp
import time
import urllib.request
import numpy as np
import threading

# URLs for two DroidCam devices
DROIDCAM_URLS = [
    "http://192.168.1.100:4747/video",  # Phonecamera
    "http://192.168.1.106:4747/video",  # Tablet camera
]

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Shared variables for threading
frames = [None, None]
fps_values = [0, 0]

def stream_thread(index, url):
    global frames, fps_values
    try:
        stream = urllib.request.urlopen(url)
        bytes_data = b''
        prev_time = 0

        while True:
            bytes_data += stream.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # FPS calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time
                fps_values[index] = int(fps)

                # Original frame copy
                original_frame = frame.copy()

                # Pose detection
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

                # Add FPS info
                cv2.putText(original_frame, f"FPS: {fps_values[index]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps_values[index]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Combine and store
                if original_frame.shape != frame.shape:
                    frame = cv2.resize(frame, (original_frame.shape[1], original_frame.shape[0]))

                combined = cv2.hconcat([original_frame, frame])
                cv2.putText(combined, "Original", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(combined, "Pose", (original_frame.shape[1] + 10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                frames[index] = combined

    except Exception as e:
        print(f"[Camera {index + 1}] Error: {e}")

# Start threads for each camera
threads = []
for i, url in enumerate(DROIDCAM_URLS):
    t = threading.Thread(target=stream_thread, args=(i, url))
    t.start()
    threads.append(t)

# Main display loop
try:
    while True:
        for i in range(len(frames)):
            if frames[i] is not None:
                resized = cv2.resize(frames[i], (0, 0), fx=0.5, fy=0.5)
                cv2.imshow(f"Camera {i+1} - Original & Pose", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Arrêt manuel...")

finally:
    pose.close()
    cv2.destroyAllWindows()

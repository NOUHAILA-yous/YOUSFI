import cv2
import mediapipe as mp
import math
import os

# Construction du chemin vers le dossier 'pfa' sur le bureau
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "pfa")
os.makedirs(desktop_path, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def draw_segment(frame, landmarks, points_idx, line_color, point_color, thickness=2):
    h, w = frame.shape[:2]
    points = []
    for idx in points_idx:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, point_color, cv2.FILLED)
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        cv2.line(frame, start, end, line_color, thickness)

cap = cv2.VideoCapture(0)

capture_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Segmentation PIEDS (3 points)
        pieds_gauche = [mp_pose.PoseLandmark.LEFT_HEEL.value,
                        mp_pose.PoseLandmark.LEFT_ANKLE.value,
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        pieds_droit = [mp_pose.PoseLandmark.RIGHT_HEEL.value,
                       mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                       mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        draw_segment(frame, lm, pieds_gauche, (255,255,0), (0,255,255))
        draw_segment(frame, lm, pieds_droit, (255,255,0), (0,255,255))

        # Segmentation CHEVILLES (3 points)
        chevilles_gauche = [mp_pose.PoseLandmark.LEFT_ANKLE.value,
                            mp_pose.PoseLandmark.LEFT_HEEL.value,
                            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        chevilles_droite = [mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                            mp_pose.PoseLandmark.RIGHT_HEEL.value,
                            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        draw_segment(frame, lm, chevilles_gauche, (255,0,0), (0,0,255))
        draw_segment(frame, lm, chevilles_droite, (255,0,0), (0,0,255))

        # Segmentation HANCHES (3 points)
        hanches_gauche = [mp_pose.PoseLandmark.LEFT_HIP.value,
                          mp_pose.PoseLandmark.LEFT_KNEE.value,
                          mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hanches_droite = [mp_pose.PoseLandmark.RIGHT_HIP.value,
                          mp_pose.PoseLandmark.RIGHT_KNEE.value,
                          mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        draw_segment(frame, lm, hanches_gauche, (0,0,255), (0,255,0))
        draw_segment(frame, lm, hanches_droite, (0,0,255), (0,255,0))

    cv2.putText(frame, "Appuyez sur 'c' pour capture, 'q' pour quitter", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Segmentation et Capture Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        filename = os.path.join(desktop_path, f"capture_{capture_count}.png")
        cv2.imwrite(filename, frame)
        print(f"Image capturée et sauvegardée sous : {filename}")
        capture_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
















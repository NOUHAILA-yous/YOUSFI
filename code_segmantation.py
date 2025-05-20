import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Ta webcam Camo Studio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        def draw_line(p1, p2, color=(0, 255, 0), thickness=4):
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

        # Segments jambe gauche
        draw_line(lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE], (255, 0, 0), 6)
        draw_line(lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.LEFT_ANKLE], (255, 0, 0), 6)
        draw_line(lm[mp_pose.PoseLandmark.LEFT_ANKLE], lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX], (255, 0, 0), 6)

        # Segments jambe droite
        draw_line(lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], (0, 0, 255), 6)
        draw_line(lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE], (0, 0, 255), 6)
        draw_line(lm[mp_pose.PoseLandmark.RIGHT_ANKLE], lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX], (0, 0, 255), 6)

        # Exemple bras gauche
        draw_line(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.LEFT_ELBOW], (0, 255, 0), 6)
        draw_line(lm[mp_pose.PoseLandmark.LEFT_ELBOW], lm[mp_pose.PoseLandmark.LEFT_WRIST], (0, 255, 0), 6)

        # Exemple bras droit
        draw_line(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_ELBOW], (0, 255, 0), 6)
        draw_line(lm[mp_pose.PoseLandmark.RIGHT_ELBOW], lm[mp_pose.PoseLandmark.RIGHT_WRIST], (0, 255, 0), 6)

    cv2.imshow("Segments - Pas de points", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









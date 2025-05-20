import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialisation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# DataFrame
df = pd.DataFrame(columns=["Frame", "Left_Knee_Angle", "Right_Knee_Angle", "Status"])
frame_count = 0

# Fonction pour angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# Boucle
while True:
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Coordonnées en pixels
        def get_px(landmark): return int(landmark.x * w), int(landmark.y * h)

        # Points clés
        l_hip, l_knee, l_ankle = get_px(lm[mp_pose.PoseLandmark.LEFT_HIP]), get_px(lm[mp_pose.PoseLandmark.LEFT_KNEE]), get_px(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
        r_hip, r_knee, r_ankle = get_px(lm[mp_pose.PoseLandmark.RIGHT_HIP]), get_px(lm[mp_pose.PoseLandmark.RIGHT_KNEE]), get_px(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])

        # Angles
        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        status = "Normale" if 160 <= left_knee_angle <= 180 and 160 <= right_knee_angle <= 180 else "Anormale"

        # Lignes
        cv2.line(img, l_hip, l_knee, (0, 255, 0), 4)
        cv2.line(img, l_knee, l_ankle, (0, 255, 0), 4)
        cv2.line(img, r_hip, r_knee, (0, 0, 255), 4)
        cv2.line(img, r_knee, r_ankle, (0, 0, 255), 4)

        # Points (couleurs différentes)
        cv2.circle(img, l_hip, 8, (255, 0, 255), -1)
        cv2.circle(img, l_knee, 8, (255, 165, 0), -1)
        cv2.circle(img, l_ankle, 8, (0, 255, 255), -1)

        cv2.circle(img, r_hip, 8, (255, 0, 255), -1)
        cv2.circle(img, r_knee, 8, (255, 165, 0), -1)
        cv2.circle(img, r_ankle, 8, (0, 255, 255), -1)

        # Affichage des angles
        cv2.putText(img, f"{int(left_knee_angle)}°", (l_knee[0] + 10, l_knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"{int(right_knee_angle)}°", (r_knee[0] + 10, r_knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Ajout dans dataframe
        df.loc[len(df)] = [frame_count, round(left_knee_angle, 2), round(right_knee_angle, 2), status]
        frame_count += 1

    cv2.imshow("Analyse de la marche", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()












import cv2
import mediapipe as mp
import os
import time
from datetime import datetime

# === Initialisation Mediapipe ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Préparer le dossier de sortie sur le bureau ===
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop, "pfa")
os.makedirs(output_folder, exist_ok=True)

# === Capture vidéo ===
cap = cv2.VideoCapture(0)
width, height = int(cap.get(3)), int(cap.get(4))

# Nom de la vidéo
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_folder, f"marche_{timestamp}.mp4")

# Initialisation de l'enregistrement vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

def marche_status(knee, ankle):
    if abs(knee.y - ankle.y) < 0.1:
        return "Normale", (0, 255, 0)
    else:
        return "Anormale", (0, 0, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        def get_point(idx):
            p = lm[idx]
            return int(p.x * w), int(p.y * h), p

        # Points articulaires
        points = {
            "Hanche G": get_point(mp_pose.PoseLandmark.LEFT_HIP),
            "Genou G": get_point(mp_pose.PoseLandmark.LEFT_KNEE),
            "Cheville G": get_point(mp_pose.PoseLandmark.LEFT_ANKLE),
            "Hanche D": get_point(mp_pose.PoseLandmark.RIGHT_HIP),
            "Genou D": get_point(mp_pose.PoseLandmark.RIGHT_KNEE),
            "Cheville D": get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)
        }

        # Dessin des lignes
        for side in ["G", "D"]:
            cv2.line(frame, points[f"Hanche {side}"][:2], points[f"Genou {side}"][:2], (255, 0, 0), 2)
            cv2.line(frame, points[f"Genou {side}"][:2], points[f"Cheville {side}"][:2], (255, 0, 0), 2)

        # Dessin des cercles
        for key in points:
            cv2.circle(frame, points[key][:2], 6, (0, 255, 255), -1)

        # Détermination de la marche
        statut, color = marche_status(points["Genou G"][2], points["Cheville G"][2])
        cv2.putText(frame, f"Marche : {statut}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Affichage
    cv2.imshow("Analyse de la Marche", frame)
    out.write(frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()

















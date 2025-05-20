import cv2
import mediapipe as mp
import math

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

def angle_between_points(p1, p2, p3):
    a = (p1.x - p2.x, p1.y - p2.y)
    b = (p3.x - p2.x, p3.y - p2.y)
    dot_prod = a[0]*b[0] + a[1]*b[1]
    mag_a = math.sqrt(a[0]**2 + a[1]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2)
    angle_rad = math.acos(dot_prod / (mag_a * mag_b + 1e-8))
    return math.degrees(angle_rad)

cap = cv2.VideoCapture(0)

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

        # Calcul angles genoux pour déterminer la marche
        angle_genou_gauche = angle_between_points(
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value],
            lm[mp_pose.PoseLandmark.LEFT_KNEE.value],
            lm[mp_pose.PoseLandmark.LEFT_HIP.value])
        angle_genou_droit = angle_between_points(
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            lm[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            lm[mp_pose.PoseLandmark.RIGHT_HIP.value])

        # Calcul levée pied (différence y entre genou et cheville)
        levee_pied_gauche = abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y - lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
        levee_pied_droit = abs(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
        levee_pied = (levee_pied_gauche + levee_pied_droit) / 2

        # Déterminer marche normale/anormale (critères simples)
        if 150 <= angle_genou_gauche <= 180 and 150 <= angle_genou_droit <= 180 and levee_pied > 0.1:
            statut = "Marche Normale"
            color = (0,255,0)
        else:
            statut = "Marche Anormale"
            color = (0,0,255)

        # Affichage du statut
        cv2.putText(frame, f"Marche: {statut}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Segmentation Pieds/Chevilles/Hanches + Analyse Marche", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













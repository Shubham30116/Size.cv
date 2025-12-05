import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

known_width_cm = 21.0
known_width_px = 400.0
pixels_per_cm = known_width_px / known_width_cm


def get_tshirt_size(shoulder_cm):
    if shoulder_cm < 6:
        return "S"
    elif 6 <= shoulder_cm < 7:
        return "M"
    elif 7 <= shoulder_cm < 8:
        return "L"
    else:
        return "XL"
    


with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame")
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Process only if landmarks detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            left_shoulder_x = int(left_shoulder.x * w)
            left_shoulder_y = int(left_shoulder.y * h)
            right_shoulder_x = int(right_shoulder.x * w)
            right_shoulder_y = int(right_shoulder.y * h)
            if left_shoulder.visibility < 0.7 or right_shoulder.visibility < 0.7:
             color=(0,255,0)
            else:
             color=(0,255,0)

             cv2.circle(image, (left_shoulder_x, left_shoulder_y), 8, (0, 255, 0), -1)
             cv2.circle(image, (right_shoulder_x, right_shoulder_y), 8, (0, 255, 0), -1)

            # Correct distance formula
            shoulder_width_px = np.sqrt((right_shoulder_x - left_shoulder_x)**2 +
                                        (right_shoulder_y - left_shoulder_y)**2)

            shoulder_width_cm = shoulder_width_px / pixels_per_cm
            tshirt_size = get_tshirt_size(shoulder_width_cm)

            cv2.putText(image, f"Shoulder width: {shoulder_width_cm:.1f} cm", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(image, f"Estimated Size: {tshirt_size}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.imshow("T-shirt Size Estimator", image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("done")
            break

cap.release()
cv2.destroyAllWindows()

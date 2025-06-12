import cv2
import dlib
import numpy as np

# Load dlib’s face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\hasin\Downloads\Artificial-Intelligence-based-Online-Exam-Proctoring-System-20250601T124841Z-1-001\Artificial-Intelligence-based-Online-Exam-Proctoring-System\shape_predictor_model\shape_predictor_68_face_landmarks.dat")

def detect_pupil(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        pupil_contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(pupil_contour)
        return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
    return None, None
def process_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_direction = "Looking Center"

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
        left_eye_rect = cv2.boundingRect(left_eye_points)
        right_eye_rect = cv2.boundingRect(right_eye_points)
        
        left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
        right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
        
        left_pupil, left_bbox = detect_pupil(left_eye)
        right_pupil, right_bbox = detect_pupil(right_eye)
        
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        
        if left_pupil and left_bbox:
            cv2.circle(frame, (left_eye_rect[0] + left_pupil[0], left_eye_rect[1] + left_pupil[1]), 5, (0, 0, 255), -1)
        if right_pupil and right_bbox:
            cv2.circle(frame, (right_eye_rect[0] + right_pupil[0], right_eye_rect[1] + right_pupil[1]), 5, (0, 0, 255), -1)

        if left_pupil and right_pupil:
            lx, ly = left_pupil
            rx, ry = right_pupil
            
            eye_width = left_eye_rect[2]
            eye_height = left_eye_rect[3]

            norm_lx = lx / eye_width
            norm_rx = rx / eye_width
            norm_ly = ly / eye_height
            norm_ry = ry / eye_height

            print(f"Normalized left pupil: ({norm_lx:.2f}, {norm_ly:.2f}), right pupil: ({norm_rx:.2f}, {norm_ry:.2f})")

            if norm_lx < 0.35 and norm_rx < 0.35:
                gaze_direction = "Looking Left"
            elif norm_lx > 0.65 and norm_rx > 0.65:
                gaze_direction = "Looking Right"
            elif norm_ly < 0.35 and norm_ry < 0.35:
                gaze_direction = "Looking Up"
            elif norm_ly > 0.65 and norm_ry > 0.65:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"
        else:
            print("Pupil(s) not detected properly")

    return frame, gaze_direction


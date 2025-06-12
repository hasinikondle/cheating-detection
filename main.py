import cv2
import time
import winsound
import os
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from head_pose_estimation import head_pose_detection
from datetime import datetime
import threading
from tab_switch_detection import monitor_tab_switch
from eye_movement import process_eye_movement
from audio_detection import audio_detection

global data_record
data_record = []

audio_logs = []
tab_switch_logs = []

running = True

frequency = 2500
duration = 1000

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam.open()

def save_suspicious_frame(frame, reason):
    folder = 'suspicious_frames'
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{reason}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)

def faceCount_detection(faceCount, frame):
    if faceCount > 1:
        #time.sleep(5)
        remark = "Multiple faces have been detected."
        save_suspicious_frame(frame, "multiple_faces")
        winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        save_suspicious_frame(frame, "no_face")
        #time.sleep(5)
        winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark

def proctoringAlgo():
    blinkCount = 0

    while running:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 🚫 End exam on too many tab switches
        if len(tab_switch_logs) > 10:
            print("Tab switched more than 5 times. Ending exam.")
            save_suspicious_frame(frame, "tab_switch_limit_exceeded")
            winsound.Beep(frequency, duration)
            break

        record = []
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        print("Current time is:", current_time)
        record.append(current_time)

        faceCount, faces = detectFace(frame)
        remark = faceCount_detection(faceCount, frame)
        print(remark)
        record.append(remark)

        for rect in faces:
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if faceCount == 1:
            blinkStatus = isBlinking(faces, frame)
            print(blinkStatus[2])
            if blinkStatus[2] == "Blink":
                blinkCount += 1
                record.append(blinkStatus[2] + " count: " + str(blinkCount))
            else:
                record.append(blinkStatus[2])

            frame, gaze_direction = process_eye_movement(frame)
            print("Gaze Direction:", gaze_direction)
            record.append(gaze_direction)
            if "Right" in gaze_direction or "Left" in gaze_direction:
                save_suspicious_frame(frame, "gaze_" + gaze_direction.lower())

            mouth_status = mouthTrack(faces, frame)
            print(mouth_status)
            record.append(mouth_status)
            if "Speaking" in mouth_status:
                save_suspicious_frame(frame, "speaking")

            object_labels, object_log = detectObject(frame)
            print(object_labels)
            record.append(object_labels)

            if object_log:
                print(object_log)
                record.append(object_log)
                save_suspicious_frame(frame, "object_detected")
                winsound.Beep(frequency, duration)
               # time.sleep(4)
                data_record.append(record)
                continue

            head_pose = head_pose_detection(faces, frame)
            print("Head Pose:", head_pose)
            record.append(head_pose)

            if isinstance(head_pose, str) and ("down" in head_pose.lower() or "side" in head_pose.lower()):
                save_suspicious_frame(frame, "head_pose_" + head_pose.lower())

        else:
            data_record.append(record)
            continue

        data_record.append(record)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()
    cv2.destroyAllWindows()

def run_audio_detection():
    audio_detection(log_list=audio_logs)

tab_switch_thread = threading.Thread(target=monitor_tab_switch, args=(tab_switch_logs,), daemon=True)
tab_switch_thread.start()

def main_app():
    activityVal = "\n".join(map(str, data_record))
    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))
        file.write("\n\nTab Switch Logs:\n")
        for log in tab_switch_logs:
            file.write(log + "\n")
        if len(tab_switch_logs) > 5:
            file.write("\n[ALERT] Exam ended due to excessive tab switching.\n")

    if audio_logs:
        with open('audio_detection_log.txt', 'w') as f:
            for log in audio_logs:
                f.write(log + "\n")

if __name__ == "__main__":
    audio_thread = threading.Thread(target=run_audio_detection)
    audio_thread.start()

    try:
        for _ in proctoringAlgo():
            pass
    except KeyboardInterrupt:
        print("Interrupted by user")

    running = False
    audio_thread.join()
    main_app()

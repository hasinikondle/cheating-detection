import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = r"C:\Users\hasin\Downloads\Artificial-Intelligence-based-Online-Exam-Proctoring-System-20250601T124841Z-1-001\Artificial-Intelligence-based-Online-Exam-Proctoring-System\object_detection_model\weights\best_yolov12.pt"
model = YOLO(model_path)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def detectObject(frame, confidence_threshold=0.5):
    """
    Detect objects using YOLOv8. Returns:
    - labels_this_frame: List of (label, confidence)
    - event_log: A string like 'Mobile phone detected' if mobile is found, else None
    """
    results = model(frame, verbose=False)
    labels_this_frame = []
    event_log = None

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            if conf < confidence_threshold:
                continue

            label = model.names.get(cls_id, f"Class {cls_id}")
            labels_this_frame.append((label, conf))

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Check for mobile phone detection (you can change the label as per your model)
            if label.lower() in ["mobile", "mobile phone", "cell phone"]:
                event_log = "Mobile phone detected"

    return labels_this_frame, event_log

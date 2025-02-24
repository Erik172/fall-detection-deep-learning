import torch
import cv2
import time
import numpy as np
import supervision as sv
import yaml
from ultralytics import YOLO
from src.inference.model_loader import load_model

# Cargar configuración desde un archivo YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_TYPE = config["model_type"]  # LSTM o CNN1D
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo de detección de caídas
model = load_model(MODEL_TYPE, DEVICE)

# Inicializar YOLO para detección de poses
model_yolo = YOLO("yolo11m-pose.pt", verbose=False)
cap = cv2.VideoCapture("data/videos/test/falls/walking-trip.mp4")

# Parámetros
sequence_length = 30
num_keypoints = 17 * 2
keypoints_buffer = []

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break

    results = model_yolo(frame)[0]
    annotated_frame = frame.copy()

    if len(results.keypoints.xy) > 0:
        keypoints = results.keypoints.xy[0].numpy().flatten()
        keypoints = np.pad(keypoints, (0, num_keypoints - keypoints.shape[0]), mode='constant')
    else:
        keypoints = np.zeros(num_keypoints, dtype=np.float32)

    keypoints_buffer.append(keypoints)

    if len(keypoints_buffer) > sequence_length:
        keypoints_buffer.pop(0)

    if len(keypoints_buffer) == sequence_length:
        input_sequence = torch.tensor([keypoints_buffer], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            output = model(input_sequence)
            prediction = torch.argmax(output, dim=1).item()

        label_text = "FALL DETECTED!" if prediction == 1 else "No Fall"
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
        cv2.putText(annotated_frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 1)

    processing_time = time.time() - start_time
    fps_real = 1 / processing_time if processing_time > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {fps_real:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 1)

    cv2.imshow("Fall Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

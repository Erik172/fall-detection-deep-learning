import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

def video_detect_falls(video_path, yolo_model_path, gru_model, fall_threshold=0.95, scale_percent=100, sequence_length=20, show_pose=False, record=False):
    from utils.body import BODY_CONNECTIONS_DRAW, BODY_PARTS_NAMES
    
    cap = cv2.VideoCapture(video_path)
    model_yolo = YOLO(yolo_model_path, verbose=False)
    byte_tracker = sv.ByteTrack()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    text_scale = 2 if scale_percent < 80 else 0.5
    text_thickness = 4 if scale_percent < 80 else 1
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)
    trace_annotator = sv.TraceAnnotator(thickness=2)
    
    keypoints_buffer = []

    # Configurar el grabador de video si record=True
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
        output_path = 'output.mp4'  # Nombre del archivo de salida
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video terminado")
            break
        
        results = model_yolo(frame)[0]
        annotated_frame = frame.copy()
        
        if len(results.keypoints.xy) > 0:
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)
            labels = [f'#{tracker_id} {results.names[class_id]} {confidence:.2f}' 
                      for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id)]
            
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            for person_idx in range(len(results.keypoints.xy)):
                if results.keypoints.xy[person_idx].size(0) == 0:
                    continue
                keypoints = results.keypoints.xy[person_idx].cpu().numpy().flatten()
                keypoints_buffer.append(keypoints)
                
                if len(keypoints_buffer) > sequence_length:
                    keypoints_buffer.pop(0)
                
                if len(keypoints_buffer) == sequence_length:
                    keypoints_sequence = np.array(keypoints_buffer, dtype=np.float32)
                    keypoints_sequence = torch.tensor(keypoints_sequence).unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction = gru_model(keypoints_sequence)
                        fall_probability = prediction.item()
                        is_fall = fall_probability > fall_threshold
                    
                    cv2.putText(annotated_frame, f'Fall: {is_fall} ({fall_probability:.2f})', 
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)
                    
                    if is_fall:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, 'Fall detected', (x1 + 10, y1 + 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
                
                if show_pose:
                    keypoints = results.keypoints.xy[person_idx]
                    body = {part: keypoints[i] for i, part in enumerate(BODY_PARTS_NAMES)}
                                
                    for group, (connections, color) in BODY_CONNECTIONS_DRAW.items():
                        for part_a, part_b in connections:
                            x1, y1 = map(int, body[part_a])
                            x2, y2 = map(int, body[part_b])
                            
                            if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                                continue
                            
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                        for part_a, _ in connections:
                            x, y = map(int, body[part_a])
                            if x == 0 or y == 0:
                                continue
                            cv2.circle(annotated_frame, (x, y), 4, color, -2)
        
        processing_time = time.time() - start_time
        fps_real = 1 / processing_time if processing_time > 0 else 0
        
        cv2.putText(annotated_frame, f'{width}x{height}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Real: {fps_real:.2f} FPS', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Personas: {len(results.keypoints.xy)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
        
        annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
        cv2.imshow('frame', annotated_frame)

        # Guardar el frame en el video si record=True
        if record:
            out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if record:
        out.release()  # Liberar el video guardado
    cv2.destroyAllWindows()

import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

def video_detect_falls(video_path, yolo_model_path, gru_model, fall_threshold=0.95, scale_percent=100, sequence_length=20, show_pose=False, record=False):
    # Import necessary modules
    from src.utils.body import BODY_CONNECTIONS_DRAW, BODY_PARTS_NAMES
    from src.models.fall_detection_gru import FallDetectionGRU
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    # Load YOLO model for object detection
    model_yolo = YOLO(yolo_model_path, verbose=False)
    # Initialize ByteTrack for object tracking
    byte_tracker = sv.ByteTrack()
    
    # Initialize FallDetectionGRU model
    model = FallDetectionGRU(
        input_size=34,  # Input size based on keypoints (17 keypoints * 2 coordinates)
        hidden_size=64,  # Number of features in the hidden state
        num_layers=2,  # Number of recurrent layers
        output_size=1,  # Output size (probability of fall)
        dropout_prob=.6  # Dropout probability
    )   
    # Load pre-trained GRU model weights
    model.load_state_dict(torch.load(gru_model))
    # Set the model to evaluation mode
    model.eval()
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Calculate new dimensions based on scale_percent
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    # Adjust text scale and thickness based on video scale
    text_scale = 2 if scale_percent < 80 else 0.5
    text_thickness = 4 if scale_percent < 80 else 1
    
    # Initialize annotators for drawing on frames
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)
    trace_annotator = sv.TraceAnnotator(thickness=2)
    
    # Buffer to store keypoints sequences for fall detection
    keypoints_buffer = []

    # Configure video recorder if record=True
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        output_path = 'output.mp4'  # Output file name
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Main loop to process video frames
    while True:
        start_time = time.time()  # Record start time for FPS calculation
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("Video ended")  # Video stream ended
            break
        
        # Perform object detection using YOLO model
        results = model_yolo(frame)[0]
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Check if any keypoints are detected
        if len(results.keypoints.xy) > 0:
            # Convert YOLO results to Supervision Detections object
            detections = sv.Detections.from_ultralytics(results)
            # Update detections with ByteTrack
            detections = byte_tracker.update_with_detections(detections)
            # Create labels for detected objects
            labels = [f'#{tracker_id} {results.names[class_id]} {confidence:.2f}' 
                      for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id)]
            
            # Annotate frame with traces, bounding boxes, and labels
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            # Process each detected person
            for person_idx in range(len(results.keypoints.xy)):
                # Skip if no keypoints for the current person
                if results.keypoints.xy[person_idx].size(0) == 0:
                    continue
                # Flatten keypoints and add to buffer
                keypoints = results.keypoints.xy[person_idx].cpu().numpy().flatten()
                keypoints_buffer.append(keypoints)
                
                # Maintain buffer size
                if len(keypoints_buffer) > sequence_length:
                    keypoints_buffer.pop(0)
                
                # If buffer is full, perform fall detection
                if len(keypoints_buffer) == sequence_length:
                    keypoints_sequence = np.array(keypoints_buffer, dtype=np.float32)
                    keypoints_sequence = torch.tensor(keypoints_sequence).unsqueeze(0)  # Add batch dimension
                    
                    # Perform inference with GRU model
                    with torch.no_grad():
                        prediction = model(keypoints_sequence)
                        fall_probability = prediction.item()
                        is_fall = fall_probability > fall_threshold
                    
                    # Display fall detection results on the frame
                    cv2.putText(annotated_frame, f'Fall: {is_fall} ({fall_probability:.2f})', 
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)
                    
                    # If a fall is detected, draw a red rectangle and text
                    if is_fall:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red rectangle
                            cv2.putText(annotated_frame, 'Fall detected', (x1 + 10, y1 + 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
                
                # If show_pose is True, draw keypoints and connections
                if show_pose:
                    keypoints = results.keypoints.xy[person_idx]
                    body = {part: keypoints[i] for i, part in enumerate(BODY_PARTS_NAMES)}
                                
                    # Draw body connections
                    for group, (connections, color) in BODY_CONNECTIONS_DRAW.items():
                        for part_a, part_b in connections:
                            x1, y1 = map(int, body[part_a])
                            x2, y2 = map(int, body[part_b])
                            
                            # Skip if keypoints are not detected (coordinates are 0)
                            if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                                continue
                            
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                        # Draw keypoints
                        for part_a, _ in connections:
                            x, y = map(int, body[part_a])
                            if x == 0 or y == 0: # Skip if keypoint not detected
                                continue
                            cv2.circle(annotated_frame, (x, y), 4, color, -2)
        
        # Calculate processing time and real FPS
        processing_time = time.time() - start_time
        fps_real = 1 / processing_time if processing_time > 0 else 0
        
        # Display video resolution, real FPS, and number of detected persons
        cv2.putText(annotated_frame, f'{width}x{height}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Real: {fps_real:.2f} FPS', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
        
        # Save the frame to the video if record=True
        if record:
            out.write(annotated_frame)
        
        # Resize frame for display
        annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
        # Display the annotated frame
        cv2.imshow('frame', annotated_frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    if record:
        out.release()  # Release the saved video
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

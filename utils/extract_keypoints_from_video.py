import os
import cv2
import numpy as np
from ultralytics import YOLO

def extract_keypoints_from_video(video_path: str, model, sequence_length: int = 10, save: bool = False, output_path: str = 'keypoints.npy'):
    num_keypoints = 17 * 2  # Número de keypoints (17 puntos * 2 coordenadas x, y)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'El archivo de video {video_path} no existe')
    
    cap = cv2.VideoCapture(video_path)
    keypoints_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video terminado
        
        results = model(frame)[0]
        
        if len(results.keypoints.xy) > 0:
            keypoints = results.keypoints.xy[0].numpy().flatten()
            if keypoints.shape[0] != num_keypoints:
                keypoints = np.pad(keypoints, (0, num_keypoints - keypoints.shape[0]))
        else:
            # Si no se detectan keypoints, usar ceros
            keypoints = np.zeros(num_keypoints, dtype=np.float32)
            
        keypoints_buffer.append(keypoints)
    
    cap.release()
    
    # Manejar casos donde el video es más corto o más largo que sequence_length
    if len(keypoints_buffer) < sequence_length:
        # Si el video es más corto, repetir el último frame hasta completar sequence_length
        last_frame = keypoints_buffer[-1]
        while len(keypoints_buffer) < sequence_length:
            keypoints_buffer.append(last_frame)
    elif len(keypoints_buffer) > sequence_length:
        # Si el video es más largo, tomar solo los últimos sequence_length frames
        keypoints_buffer = keypoints_buffer[-sequence_length:]
    
    keypoints_buffer = np.array(keypoints_buffer, dtype=np.float32)
    
    if save:
        np.save(output_path, keypoints_buffer)
        print(f'Guardado en {output_path}')
    
    return keypoints_buffer
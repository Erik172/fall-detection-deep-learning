import os
import cv2
import numpy as np
from ultralytics import YOLO

def extract_keypoints_from_video(video_path: str, model, sequence_length: int = 10, save: bool = False, output_path: str = 'keypoints.npy'):
    """Extracts keypoints from a video using a YOLO model.

    Args:
        video_path (str): Path to the input video file.
        model: The YOLO model instance to use for keypoint detection.
        sequence_length (int, optional): The desired length of the keypoint sequence.
                                         Defaults to 10.
        save (bool, optional): Whether to save the extracted keypoints to a .npy file.
                               Defaults to False.
        output_path (str, optional): Path to save the keypoints if save is True.
                                     Defaults to 'keypoints.npy'.

    Raises:
        FileNotFoundError: If the video_path does not exist.

    Returns:
        numpy.ndarray: An array of keypoint sequences.
    """
    num_keypoints = 17 * 2  # Number of keypoints (17 points * 2 coordinates x, y)
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file {video_path} not found') # Translated error message
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    # Buffer to store keypoints from frames
    keypoints_buffer = []
    
    # Process video frame by frame
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Video ended or error reading frame
        
        # Perform keypoint detection using the YOLO model
        results = model(frame)[0]
        
        # Check if keypoints are detected in the current frame
        if len(results.keypoints.xy) > 0:
            # Get keypoints for the first detected person and flatten them
            keypoints = results.keypoints.xy[0].cpu().numpy().flatten() # Added .cpu() for robustness
            # Pad with zeros if the number of detected keypoints is less than expected
            if keypoints.shape[0] != num_keypoints:
                keypoints = np.pad(keypoints, (0, num_keypoints - keypoints.shape[0]))
        else:
            # If no keypoints are detected, use an array of zeros
            keypoints = np.zeros(num_keypoints, dtype=np.float32)
            
        keypoints_buffer.append(keypoints)
    
    # Release the video capture object
    cap.release()
    
    # Handle cases where the video is shorter or longer than sequence_length
    if len(keypoints_buffer) == 0: # Handle empty videos
        # If the video was empty or no keypoints were ever detected, fill with zeros
        for _ in range(sequence_length):
            keypoints_buffer.append(np.zeros(num_keypoints, dtype=np.float32))
    elif len(keypoints_buffer) < sequence_length:
        # If the video is shorter, repeat the last frame's keypoints to complete the sequence
        last_keypoints = keypoints_buffer[-1]
        while len(keypoints_buffer) < sequence_length:
            keypoints_buffer.append(last_keypoints)
    elif len(keypoints_buffer) > sequence_length:
        # If the video is longer, take only the last sequence_length frames
        keypoints_buffer = keypoints_buffer[-sequence_length:]
    
    # Convert the buffer to a NumPy array
    keypoints_buffer = np.array(keypoints_buffer, dtype=np.float32)
    
    # Save the keypoints to a file if specified
    if save:
        np.save(output_path, keypoints_buffer)
        print(f'Keypoints saved to {output_path}') # Translated print message
    
    return keypoints_buffer
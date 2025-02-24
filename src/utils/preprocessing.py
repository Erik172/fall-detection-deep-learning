import numpy as np

def normalize_keypoints(keypoints):
    """ Normaliza los puntos clave para que tengan valores en un rango estable """
    min_val = np.min(keypoints, axis=0)
    max_val = np.max(keypoints, axis=0)
    return (keypoints - min_val) / (max_val - min_val + 1e-6)

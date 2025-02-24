import torch
from torch.utils.data import Dataset
import numpy as np

class FallDetectionDataset(Dataset):
    def __init__(self, data_path, fps=30):
        self.keypoints = np.load(f"{data_path}/keypoints_sequences_{fps}fps.npy")
        self.labels = np.load(f"{data_path}/labels_{fps}fps.npy")

        self.keypoints = torch.tensor(self.keypoints, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.keypoints[idx], self.labels[idx]

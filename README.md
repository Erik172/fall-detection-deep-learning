# Fall Detection using Pose Estimation and GRU

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/me/stats/post/2941db4c95a3)

![Banana Peel](media/banana-peel-inference.gif)

## Description

This project implements a fall detection system using pose estimation techniques and recurrent neural networks (GRU). The system is capable of analyzing video sequences to identify falls in real-time.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for model training and evaluation.
- `src/`: Contains the project's source code.
  - `models/`: Implementations of fall detection models.
    - `fall_detection_lstm.py`: LSTM model for fall detection.
    - `fall_detection_gru.py`: GRU model for fall detection.
  - `utils/`: Utilities and helper functions.
    - `video_detect_falls.py`: Functions for fall detection in videos.
    - `body.py`: Definitions of connections and body parts for pose estimation.
- `data/`: Directory to store training and test data.
- `media/`: Contains images and videos used in the README and other documents.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fall-detection.git
    cd fall-detection
    ```

2. Create a virtual environment and install the dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage

### Fall Detection in Video

To detect falls in a video, use the following script:

```python
from src.utils.video_detect_falls import video_detect_falls

video_detect_falls(
    video_path='data/videos/falls/yoga-fail-fall.mp4', # Change this to the path of the video you want to test
    yolo_model_path='yolo11x-pose.pt',
    gru_model='models/gru_model.pth',
    fall_threshold=.95,
    scale_percent=100,
    sequence_length=20,
    show_pose=True,
    record=True,
)
```
BODY_PARTS_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

BODY_CONNECTIONS_DRAW = {
    "piernas": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")], (255, 0, 0)),
    "brazos": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")], (0, 255, 0)),
    "cabeza": ([("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")], (0, 0, 255)),
    "torso": ([("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip")], (255, 255, 0))
}

BODY_GROUPS = {
    "piernas": ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"],
    "brazos": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
    "cabeza": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
}

def draw_skeleton(ax, keypoints, title):
    for joint1, joint2 in BODY_CONNECTIONS:
        x_values = [keypoints[joint1, 0], keypoints[joint2, 0]]
        y_values = [keypoints[joint1, 1], keypoints[joint2, 1]]
        ax.plot(x_values, y_values, 'bo-', markersize=3, alpha=0.5)  # Dibujar conexiones
    ax.scatter(keypoints[:, 0], keypoints[:, 1], color='red', s=50, label='Joint position')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
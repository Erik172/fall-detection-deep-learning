import torch
from src.models.lstm import FallDetectorLSTM
from src.models.cnn1d import FallDetectorCNN1D

def load_model(model_type, device):
    """Carga el modelo seleccionado según el tipo especificado."""
    model_paths = {
        "LSTM": "models/lstm_fall_detector.pth",
        "CNN1D": "models/fall_detector_cnn.pth"
    }

    if model_type == "LSTM":
        model = FallDetectorLSTM(input_size=34, hidden_size=128, num_layers=2, num_classes=2).to(device)
    elif model_type == "CNN1D":
        model = FallDetectorCNN1D(input_size=34, num_classes=2).to(device)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    model.load_state_dict(torch.load(model_paths[model_type], map_location=device))
    model.eval()
    return model

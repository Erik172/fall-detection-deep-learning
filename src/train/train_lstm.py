import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.lstm import FallDetectorLSTM
from src.utils.dataset import FallDetectionDataset

def train_fall_detector(data_path, model_save_path, batch_size=32, sequence_length=30, num_keypoints=34, hidden_size=128, num_layers=2, num_classes=2, num_epochs=60, learning_rate=0.001, fps=30):
    # Cargar datos
    train_dataset = FallDetectionDataset(data_path, fps=fps)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Crear modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectorLSTM(num_keypoints, hidden_size, num_layers, num_classes, use_gru=False).to(device)

    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    accuracy_history = []

    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            loss_history.append(loss.item())
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Guardar modelo
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo guardado en {model_save_path}")
    
    return loss_history

# Ejemplo de uso
# train_fall_detector("data/processed", "models/fall_detector_lstm.pth")

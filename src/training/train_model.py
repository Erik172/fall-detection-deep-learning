import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=0.0001, 
                weight_decay=1e-4, patience=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Entrena un modelo de detección de caídas usando LSTM o GRU.

    Args:
        model (torch.nn.Module): Modelo a entrenar.
        train_loader (DataLoader): DataLoader para los datos de entrenamiento.
        val_loader (DataLoader): DataLoader para los datos de validación.
        num_epochs (int): Número de épocas de entrenamiento.
        learning_rate (float): Tasa de aprendizaje.
        weight_decay (float): Parámetro de regularización L2.
        patience (int): Número de épocas sin mejora antes de early stopping.
        device (str): Dispositivo a usar ("cuda" o "cpu").

    Returns:
        model: Modelo entrenado.
        history: Diccionario con métricas de entrenamiento y validación.
    """
    model.to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Listas para almacenar métricas
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)

        # Evaluación en validación
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        # Guardar el mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"⏹️ Early stopping en la época {epoch + 1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Restaurar el mejor modelo
    if best_model:
        model.load_state_dict(best_model)
        print("✔️ Modelo restaurado al mejor estado guardado.")

    return model, history

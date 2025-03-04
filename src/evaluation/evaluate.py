import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evalúa el modelo en un conjunto de prueba y calcula métricas de rendimiento.

    Args:
        model (torch.nn.Module): Modelo entrenado.
        test_loader (DataLoader): DataLoader con los datos de prueba.
        device (str): Dispositivo a usar ("cuda" o "cpu").

    Returns:
        dict: Métricas de evaluación (accuracy, precision, recall, f1-score, test_loss).
    """
    model.to(device)
    model.eval()
    
    criterion = nn.BCELoss()
    test_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    print(f"🔍 Evaluación del modelo:")
    print(f"📉 Test Loss: {avg_test_loss:.4f}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔄 Recall: {recall:.4f}")
    print(f"⭐ F1 Score: {f1:.4f}")

    return {"test_loss": avg_test_loss, "accuracy": accuracy, 
            "precision": precision, "recall": recall, "f1_score": f1}
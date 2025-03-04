import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Grafica la pérdida y precisión del entrenamiento y validación.

    Args:
        history (dict): Diccionario con las métricas de entrenamiento y validación.
            Debe contener las claves: 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    plt.figure(figsize=(10, 4))

    # Gráfica de pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Gráfica de precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

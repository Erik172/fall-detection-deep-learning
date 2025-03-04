import torch
import torch.nn as nn

class FallDetectionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(FallDetectionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa GRU (en vez de LSTM)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # Capas fully connected adicionales
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout_prob)
        
        # Funciones de activación
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización de pesos para mejorar la convergencia."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.01)
    
    def forward(self, x):
        # Inicializar estado oculto (sin c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pasar a través de la GRU
        out, _ = self.gru(x, h0)
        
        # Tomar la salida del último paso de tiempo
        out = out[:, -1, :]
        
        # Batch Normalization
        out = self.bn(out)
        
        # Capas fully connected con activaciones y dropout
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        # Aplicar sigmoide para obtener probabilidades
        out = self.sigmoid(out)
        return out

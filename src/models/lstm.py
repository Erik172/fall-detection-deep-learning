import torch
import torch.nn as nn

class FallDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, use_gru=False):
        super(FallDetectorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru

        self.fc1 = nn.Linear(input_size, hidden_size)  # Proyección inicial

        # LSTM o GRU
        if use_gru:
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.fc2 = nn.Linear(hidden_size, num_classes)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Transformación inicial
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        if self.use_gru:
            out, _ = self.rnn(x, h0)
        else:
            out, _ = self.rnn(x, (h0, c0))

        out = self.fc2(out[:, -1, :])  # Última salida de la secuencia
        return out

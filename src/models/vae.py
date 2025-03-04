from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # Cambiado ReLU a LeakyReLU
            nn.BatchNorm1d(hidden_dim),  # Normalización de lotes
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # Otra capa de LeakyReLU
            nn.BatchNorm1d(hidden_dim),  # Normalización de lotes
            nn.Linear(hidden_dim, latent_dim * 2)  # Salida: media y log-varianza
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # Cambiado ReLU a LeakyReLU
            nn.BatchNorm1d(hidden_dim),  # Normalización de lotes
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # Otra capa de LeakyReLU
            nn.BatchNorm1d(hidden_dim),  # Normalización de lotes
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Para datos en el rango [0, 1]
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick para muestrear de la distribución latente."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Paso a través del encoder
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)  # Dividir en media y log-varianza
        
        # Muestrear del espacio latente
        z = self.reparameterize(mu, logvar)
        
        # Reconstruir los datos
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
    # Pérdida de reconstrucción (MSE o BCE)
        # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # BCE como función de pérdida de reconstrucción
        BCE = nn.functional.cross_entropy(recon_x, x, reduction='sum')  # BCE como función de pérdida de reconstrucción

        # Divergencia KL
        # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # donde sigma es la desviación estándar
        # y mu es la media
        # logvar es el logaritmo de la varianza
        # D_KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Pérdida total
        return BCE + KL
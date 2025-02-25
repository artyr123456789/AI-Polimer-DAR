import torch
import torch.nn as nn

class InverseEncoder(nn.Module):
    def __init__(self, input_dims):
        super(InverseEncoder, self).__init__()
        
        # Input layers for target properties
        self.density_layer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()  # Add non-linearity
        )
        self.strength_layer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()  # Add non-linearity
        )
        self.thermal_layer = nn.Sequential(
            nn.Linear(3, 32),  # [conductivity, expansion, Tg]
            nn.ReLU()  # Add non-linearity
        )
        
        # Merge features
        self.fc = nn.Sequential(
            nn.Linear(16 + 16 + 32, 128),
            nn.ReLU()  # Add non-linearity
        )
        
        # Hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # Add normalization
            nn.ReLU(),  # Add non-linearity
            nn.Dropout(0.4),  # Add dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Add normalization
            nn.ReLU(),  # Add non-linearity
            nn.Dropout(0.4)  # Add dropout
        )
        
        # Latent space
        self.mu = nn.Linear(128, 64)
        self.logvar = nn.Linear(128, 64)

    def forward(self, x):
        # Process density
        density = self.density_layer(x['density'])
        
        # Process tensile strength
        strength = self.strength_layer(x['tensile_strength'])
        
        # Process thermal properties
        thermal = self.thermal_layer(x['thermal_properties'])
        
        # Combine all features
        combined = torch.cat([density, strength, thermal], dim=1)
        
        # Pass through fully connected layer
        hidden = self.hidden(self.fc(combined))
        
        # Compute mu and logvar for VAE
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        
        return mu, logvar
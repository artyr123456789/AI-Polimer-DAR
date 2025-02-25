import torch
import torch.nn as nn

class ForwardDecoder(nn.Module):
    def __init__(self, output_dims):
        super(ForwardDecoder, self).__init__()
        
        # Map latent space to hidden representation
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU()  # Add non-linearity
        )
        
        # Hidden layers with normalization and activation
        self.hidden = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),  # Normalization
            nn.LeakyReLU(),     # Non-linear activation
            nn.Linear(128, 256),
            nn.LayerNorm(256),  # Normalization
            nn.LeakyReLU()      # Non-linear activation
        )
        
        # Output layers for each property type
        self.density_out = nn.Linear(256, 1)  # Density (no activation, can be any value)
        self.tensile_out = nn.Linear(256, 1)  # Tensile strength (no activation, can be any value)
        self.thermal_out = nn.Sequential(     # Thermal properties [conductivity, expansion, Tg]
            nn.Linear(256, 3),
            nn.ReLU()  # Ensure thermal properties are non-negative
        )

    def forward(self, z):
        # Map latent vector to hidden representation
        hidden = self.hidden(self.latent_to_hidden(z))
        
        # Generate outputs for each property
        return {
            'density': self.density_out(hidden),  # Density
            'tensile_strength': self.tensile_out(hidden),  # Tensile strength
            'thermal_properties': self.thermal_out(hidden)  # Thermal properties
        }
import torch
import torch.nn as nn

class InverseDecoder(nn.Module):
    def __init__(self, output_dims):
        super(InverseDecoder, self).__init__()
        
        # Map latent space to hidden representation
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()  # Add non-linearity
        )
        
        # Hidden layers with normalization and activation
        self.hidden = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),  # Normalization
            nn.LeakyReLU(),     # Non-linear activation
            nn.Dropout(0.3),    # Dropout for regularization
            nn.Linear(256, 512),
            nn.LayerNorm(512),  # Normalization
            nn.LeakyReLU()      # Non-linear activation
        )
        
        # Output heads
        self.material_logits = nn.Linear(512, output_dims['materials'])  # Material IDs (logits)
        self.proportions_out = nn.Sequential(
            nn.Linear(512, output_dims['proportions']),
            nn.Sigmoid()  # Normalize proportions to [0, 1]
        )
        self.temperature_out = nn.Sequential(
            nn.Linear(512, output_dims['time_temp_steps']),
            nn.Tanh()  # Limit temperature values to [-1, 1] (can be scaled later)
        )

    def forward(self, z):
        # Map latent vector to hidden representation
        hidden = self.hidden(self.latent_to_hidden(z))
        
        # Generate outputs for each property
        return {
            'materials': self.material_logits(hidden),  # Material logits
            'proportions': self.proportions_out(hidden),  # Proportions in [0, 1]
            'temperature_profile': self.temperature_out(hidden)  # Scaled temperature profile
        }
        # Added to ForwardEncoder/InverseDecoder  
    self.time_processor = nn.LSTM(input_size=64, hidden_size=32, num_layers=2)  
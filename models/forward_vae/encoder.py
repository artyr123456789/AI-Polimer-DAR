import torch
import torch.nn as nn

class ForwardEncoder(nn.Module):
    def __init__(self, input_dims):
        super(ForwardEncoder, self).__init__()
        
        # Input layers grouped by variable type
        self.proportion_layer = nn.Sequential(
            nn.Linear(input_dims['proportions'], 64),
            nn.ReLU()
        )
        self.temp_layer = nn.Sequential(
            nn.Linear(input_dims['temperature'], 32),
            nn.ReLU()
        )
        self.equipment_layer = nn.Embedding(input_dims['equipment'], 16)
        
        # If equipment is a list of indices, we need to aggregate embeddings
        self.equipment_aggregator = nn.Sequential(
            nn.Linear(16, 16),  # Assuming equipment embeddings are aggregated by mean or sum
            nn.ReLU()
        )
        
        # Concatenate all features
        self.fc = nn.Sequential(
            nn.Linear(64 + 32 + 16, 256),
            nn.ReLU()
        )
        
        # Hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent space
        self.mu = nn.Linear(64, 32)
        self.logvar = nn.Linear(64, 32)

    def forward(self, x):
        # Process proportions
        prop = self.proportion_layer(x['proportions'])
        
        # Process temperature
        temp = self.temp_layer(x['temperature'])
        
        # Process equipment
        equip = self.equipment_layer(x['equipment'])  # Shape: [batch_size, num_equipment, embedding_dim]
        
        # Aggregate equipment embeddings (e.g., mean or sum)
        if len(equip.shape) == 3:  # If equipment is a list of indices
            equip = equip.mean(dim=1)  # Average embeddings across equipment
        equip = self.equipment_aggregator(equip)
        
        # Combine all features
        combined = torch.cat([prop, temp, equip], dim=1)
        
        # Pass through fully connected layer
        combined = self.fc(combined)
        
        # Pass through hidden layers
        hidden = self.hidden(combined)
        
        # Compute mu and logvar for VAE
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        
        return mu, logvar
        # Added to ForwardEncoder/InverseDecoder  
    self.time_processor = nn.LSTM(input_size=64, hidden_size=32, num_layers=2)  
 
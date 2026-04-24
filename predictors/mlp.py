
import torch
import torch.nn as nn

class MLPInteractionPredictor(nn.Module):
    
    def __init__(self, input_dim, sequence_length, hidden_dims=[512, 256, 128], dropout=0.3):
        super(MLPInteractionPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dims = hidden_dims
        
        # Flatten the sequence dimension
        flattened_input_dim = input_dim * sequence_length
        
        # Build the MLP layers
        layers = []
        prev_dim = flattened_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim) with input_dim the number of features at each time step
            
        Returns:
            Output tensor of shape (batch_size, 1) with logits
        """
        # Flatten the sequence dimension
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch_size, sequence_length * input_dim)
        
        # Pass through MLP
        logits = self.mlp(x)
        
        return logits
import torch
import torch.nn as nn

class LSTMInteractionPredictor(nn.Module):
    
    def __init__(self, input_dim, sequence_length, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=False):
        super(LSTMInteractionPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate the output dimension after LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through the LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim) 
               with input_dim the number of features at each time step
            
        Returns:
            Output tensor of shape (batch_size, 1) with logits
        """
        # LSTM forward pass
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Use the last hidden state from the last layer for classification
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden_forward = hidden[-2]  # Last forward hidden state
            hidden_backward = hidden[-1]  # Last backward hidden state
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            final_hidden = hidden[-1]  # Last layer's hidden state
        
        # final_hidden shape: (batch_size, hidden_dim * num_directions)
        
        # Pass through classifier
        logits = self.classifier(final_hidden)
        
        return logits


import torch
import torch.nn as nn
import numpy as np

class DecayMLP(nn.Module):
    """
    Learned Decay MLP: 
    Inputs:
    - Delta Time (seconds, normalized)
    - Semantic Score (from vector store)
    - Source Authority (Optional)
    """
    def __init__(self, input_dim=2, hidden_dim=8):
        super(DecayMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output a decay scalar [0, 1]
        )
        
        # Initialize with reasonable defaults
        # We want the model to act somewhat like exponential decay initially
        with torch.no_grad():
            # Roughly favoring higher scores and lower delta_time
            self.net[0].weight.fill_(0.01)
            self.net[2].weight.fill_(0.1)

    def forward(self, delta_time: float, vector_score: float):
        # Normalize delta_time roughly (e.g., scale by days in a year)
        # Assuming delta_time is in seconds
        dt_scaled = delta_time / (86400 * 365) # days -> years 
        
        x = torch.tensor([[dt_scaled, vector_score]], dtype=torch.float32)
        return self.net(x).item()

class LearnedDecayEngine:
    def __init__(self):
        self.model = DecayMLP()

    def get_score(self, delta_time, vector_score):
        return self.model(delta_time, vector_score)

# Global instances for our middleware
LEARNED_ENGINE = LearnedDecayEngine()

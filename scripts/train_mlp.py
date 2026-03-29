import torch
import torch.optim as optim
import sys
import os
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.decay.learned_model import LEARNED_ENGINE

def train_pass(delta_time, vector_score, target_score):
    """
    Very simple online training step for the MLP.
    This simulates how the benchmark labels improve the model.
    """
    optimizer = optim.Adam(LEARNED_ENGINE.model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Scale dt by year to keep neural weights stable
    dt_scaled = delta_time / (86400 * 365)
    
    X = torch.tensor([[dt_scaled, vector_score]], dtype=torch.float32)
    y = torch.tensor([[target_score]], dtype=torch.float32)
    
    # Update loop
    for _ in range(10):
        optimizer.zero_grad()
        pred = LEARNED_ENGINE.model.net(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    print(f"Trained on dt={delta_time:.0f}s score={vector_score:.2f} -> target={target_score:.2f}. Loss: {loss.item():.6f}")

if __name__ == "__main__":
    # Example: A very fresh, very relevant chunk should have decay score ~1.0
    train_pass(delta_time=1000, vector_score=0.9, target_score=1.0)
    # A very old but very relevant chunk might be a landmark, let's teach it to stay high
    train_pass(delta_time=86400 * 365 * 5, vector_score=0.9, target_score=0.8)
    
    # Save the model
    torch.save(LEARNED_ENGINE.model.state_dict(), "decay_mlp.pth")
    print("Model saved to decay_mlp.pth")

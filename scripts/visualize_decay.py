import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.decay.exponential import ExponentialDecay
from engine.decay.piecewise import PiecewiseDecay

def visualize():
    now = datetime.now(timezone.utc)
    days = np.linspace(0, 400, 400)
    timestamps = [now - timedelta(days=float(d)) for d in days]
    
    # 1. Exponential Decay (News)
    exp_fast = ExponentialDecay(params={"lambda": 1e-5}) # Very fast
    exp_med = ExponentialDecay(params={"lambda": 1e-6})  # Medium
    exp_slow = ExponentialDecay(params={"lambda": 1e-7}) # Landmark
    
    # 2. Piecewise Decay (Docs)
    piecewise = PiecewiseDecay(params={})
    
    scores_fast = [exp_fast.compute(ts, now) for ts in timestamps]
    scores_med = [exp_med.compute(ts, now) for ts in timestamps]
    scores_slow = [exp_slow.compute(ts, now) for ts in timestamps]
    scores_piece = [piecewise.compute(ts, now) for ts in timestamps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(days, scores_fast, label='Exponential (Fast/News)', linestyle='--')
    plt.plot(days, scores_med, label='Exponential (Med/Default)', linestyle='-.')
    plt.plot(days, scores_slow, label='Exponential (Slow/Landmark)')
    plt.plot(days, scores_piece, label='Piecewise (Docs/Stable)', linewidth=2)
    
    plt.title('HalfLife Score Decay Families')
    plt.xlabel('Days Since Creation')
    plt.ylabel('Temporal Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), 'decay_curves.png')
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    try:
        visualize()
    except ImportError:
        print("Matplotlib or Numpy not installed. Skipping visualization.")

import pytest
from datetime import datetime, timedelta, timezone
from engine.decay.exponential import ExponentialDecay
from engine.decay.piecewise import PiecewiseDecay

def test_exponential_decay():
    now = datetime.now(timezone.utc)
    ts_fresh = now - timedelta(seconds=0)
    ts_old = now - timedelta(days=7)
    
    # Lambda = 1e-6 (8-day half-life approx)
    decay = ExponentialDecay(params={"lambda": 1e-6})
    
    score_fresh = decay.compute(ts_fresh, now)
    score_old = decay.compute(ts_old, now)
    
    assert score_fresh == 1.0
    assert 0.0 < score_old < 1.0
    assert score_fresh > score_old

def test_piecewise_decay():
    now = datetime.now(timezone.utc)
    ts_week = now - timedelta(days=2)   # 1.0 score expected
    ts_year = now - timedelta(days=10) # 0.7 score expected
    ts_old = now - timedelta(days=400) # 0.3 score expected
    
    decay = PiecewiseDecay(params={})
    
    score_week = decay.compute(ts_week, now)
    score_year = decay.compute(ts_year, now)
    score_old = decay.compute(ts_old, now)
    
    assert score_week == 1.0
    assert score_year == 0.7
    assert score_old == 0.3

def test_negative_time_drift():
    """Ensure decay handles cases where timestamp is in the future."""
    now = datetime.now(timezone.utc)
    ts_future = now + timedelta(seconds=100)
    
    decay = ExponentialDecay(params={"lambda": 1e-6})
    score = decay.compute(ts_future, now)
    
    # Negative time delta should be capped at 0.0 (max relevance)
    assert score == 1.0

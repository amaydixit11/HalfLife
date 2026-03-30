# 📝 Google Colab Testing Guide

HalfLife is designed to be production-ready (Docker), but it also works in "Zero-Infrastructure" environments like Google Colab.

## 🚀 1. Automated Setup
Run this in a Colab cell to install the necessary services without Docker:

```python
# 1. Install & Start Redis
!apt-get install -y redis-server > /dev/null
!service redis-server start

# 2. Download & Launch Qdrant Binary (x86_64)
!wget https://github.com/qdrant/qdrant/releases/download/v1.9.0/qdrant-x86_64-unknown-linux-gnu.tar.gz -q
!tar -xf qdrant-x86_64-unknown-linux-gnu.tar.gz
!nohup ./qdrant > qdrant.log 2>&1 &

# 3. Install Dependencies
!pip install -q rich sentence-transformers qdrant-client redis typer
```

## 🎭 2. Running the Demo
Once the setup is complete, you can run the same adversarial scenarios directly in Colab:

```python
# Assuming you've cloned/uploaded the repo to /content/HalfLife
%cd /content/HalfLife
!python scripts/demo.py
```

## 🔬 3. Local-Only Mode (No Binaries Required)
If you just want to test the Python SDK logic without running servers, you can initialize the client in "Local Storage" mode (requires zero binaries):

```python
from qdrant_client import QdrantClient
from halflife import HalfLifeIngestor

# This creates a local Qdrant instance in the /content folder
ingestor = HalfLifeIngestor(qdrant_url="local_db")
```

---

### ⚠️ Note on Redis in Colab
While the binary setup works, some advanced Redis features (like EventBus invalidation) might require manual config if you aren't using the default `6379` port. The snippet above uses the default, so it should be plug-and-play.

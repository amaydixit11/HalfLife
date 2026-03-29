FROM python:3.10-slim

WORKDIR /app

# No system updates or build-essential needed for binary wheels.
# Using --only-binary :all: in requirements.txt.

# Install python dependencies — fast binary fetch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

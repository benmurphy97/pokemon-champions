FROM python:3.11-slim

WORKDIR /app

# PyTorch CPU — separate layer so it stays cached across requirements changes
RUN pip install --no-cache-dir --timeout 300 torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

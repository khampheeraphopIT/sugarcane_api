FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV + PyTorch (added libgl1 for opencv support)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create weights directory if it doesn't exist
RUN mkdir -p ml/weights

# Render uses $PORT, default to 10000
ENV PORT=10000
EXPOSE ${PORT}

# Run with 1 worker to save RAM on Free Tier
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1

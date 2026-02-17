# FROM python:3.10-slim

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# # Expose API port
# EXPOSE 8000

# # Run FastAPI app
# CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000 5000

# Default command
CMD ["python", "main.py"]
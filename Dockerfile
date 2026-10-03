# IBVAP Production CPU-First Surveillance Node
FROM python:3.10-slim

# System dependencies for OpenCV, EasyOCR, and SQLite3
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY app/ ./app/
COPY configs/ ./configs/
COPY templates/ ./templates/
COPY download_models.py .
COPY check_setup.py .

# Create persistent storage directories
RUN mkdir -p forensic_logs known_faces models/face models/plate models/yolov8 data

# Attempt automated model pre-downloads (non-fatal if network is restricted during build)
RUN python download_models.py || true
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

ENV PYTHONUNBUFFERED=1
ENV DB_PATH=data/defense_audit.db
EXPOSE 8000

# Health check verifies the live telemetry API
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/system_telemetry || exit 1

CMD ["python", "run.py"]
FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY uav_project/backend/requirements.txt ./uav_project/backend/
RUN pip install --no-cache-dir -r uav_project/backend/requirements.txt

# Copy full application codebase
COPY . .

WORKDIR /app/uav_project/backend

EXPOSE 8000

# Run FastAPI with uvicorn listening on all interfaces
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]


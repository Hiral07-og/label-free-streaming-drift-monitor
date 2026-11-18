# Multi-stage Dockerfile for Label-Free Streaming Drift Monitor
# Unified Project Container (ML Engine + Streaming Backend + Web Dashboard)

FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim as runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy unified application source code
COPY main.py /app/main.py
COPY ml_engine /app/ml_engine
COPY backend /app/backend

# Expose FastAPI, Web Dashboard, and Prometheus metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch unified server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]

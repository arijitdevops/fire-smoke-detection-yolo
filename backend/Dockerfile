# Backend image: FastAPI + Ultralytics on CPU.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UPLOAD_DIR=/data/uploads \
    OUTPUT_DIR=/data/outputs \
    JOB_DB_PATH=/data/jobs.sqlite3 \
    MODEL_PATH=/models/best.pt

# libglib2 is required by opencv-python-headless; ffmpeg gives OpenCV a broad
# set of demuxers for user-supplied video.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" "torchvision>=0.17,<1.0" \
    && pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /data/uploads /data/outputs /models \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser /data /models /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

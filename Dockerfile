FROM python:3.11-slim

# System deps: OpenCV headless, ffmpeg (PyAV), OpenMP (XGBoost/sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "api.endpoint:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.11-slim

WORKDIR /app

# System libs needed by librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only first (avoids pulling the ~2 GB CUDA build)
RUN pip install --no-cache-dir \
    torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# All other Python dependencies
RUN pip install --no-cache-dir \
    openai-whisper \
    transformers \
    sentencepiece \
    sacremoses \
    edge-tts \
    librosa \
    soundfile \
    "fastapi>=0.104.0" \
    "uvicorn[standard]" \
    websockets \
    tqdm \
    numpy

# Pre-download models so they're baked into the image
# (no cold-start model download at runtime)
RUN python -c "import whisper; whisper.load_model('small')"
RUN python -c "\
from transformers import MarianMTModel, MarianTokenizer; \
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-hi-en'); \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-hi-en')"

# Copy only the files the webapp needs
COPY data_pipeline/ ./data_pipeline/
COPY webapp/        ./webapp/

EXPOSE 7860
ENV PORT=7860

CMD ["python", "webapp/app.py"]

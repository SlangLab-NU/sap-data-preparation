FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Pin numpy to 1.x (CRITICAL)
RUN pip install --no-cache-dir "numpy<2.0,>=1.21.0"

# Install PyTorch 2.1.2
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install transformers 4.x (compatible with torch 2.1.2)
RUN pip install --no-cache-dir \
    "transformers>=4.36.0,<5.0.0" \
    "huggingface-hub<0.20.0"

# Install StyleTTS2's exact requirements
RUN pip install --no-cache-dir \
    "accelerate>=0.25.0,<0.26.0" \
    "einops>=0.7.0,<0.8.0" \
    "einops-exts>=0.0.4,<0.0.5" \
    "filelock>=3.12.4,<3.13" \
    "networkx>=2.8.8,<3.0.0" \
    "librosa>=0.10.1,<0.11.0" \
    "soundfile>=0.12.1,<0.13.0" \
    "langchain>=0.1.0,<0.2.0" \
    "munch>=4.0.0,<5.0.0" \
    phonemizer \
    "gruut[de,es,fr]" \
    cached-path \
    nltk \
    pydub \
    pyyaml \
    click \
    scipy

# Install styletts2 with --no-deps (dependencies already installed)
RUN pip install --no-cache-dir --no-deps styletts2

# Install analysis packages
RUN pip install --no-cache-dir \
    pandas \
    tqdm \
    matplotlib

# Install Lhotse
RUN pip install --no-cache-dir \
    "lhotse>=1.20.0" \
    "lilcom>=0.2" \
    "cytoolz>=0.10.1"

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

WORKDIR /workspace
ENV PYTHONPATH=/workspace:$PYTHONPATH
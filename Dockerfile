FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04


# ----- Install base packages -----
RUN apt-get update && apt-get install -y \
    python3.12 python3-pip python3.12-venv git ffmpeg nano curl \
    && rm -rf /var/lib/apt/lists/*


# ----- Create Python venv -----
RUN python3 -m venv /venv

ENV PATH="/venv/bin:$PATH"


# ----- Set up Hugging Face cache directory -----
RUN mkdir /huggingface

ENV HF_HOME=/huggingface

# ----- Set up workspace -----
WORKDIR /workspace


# ----- Install Python deps -----
COPY requirements.txt /workspace/

RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# ----- Install Ollama -----

# This replicates the official Ollama installation steps
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz \
    | tar -xvz -C /usr/local

# Create runtime directory for ollama
RUN mkdir -p /usr/share/ollama
ENV OLLAMA_MODELS=/usr/share/ollama/models

# Enable GPU for Ollama
ENV OLLAMA_USE_GPU=1
ENV OLLAMA_GPU_DRIVER=nvidia

# Force the server to use nvidia runtime libs
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


# ----- Copy files -----
COPY src /workspace/src

ENV PYTHONPATH=/workspace/src


# ----- Add shortcut -----

RUN chmod +x /workspace/src/skipnote_ui/slidebasedvideo/cli.py

RUN ln -s /workspace/src/skipnote_ui/slidebasedvideo/cli.py /usr/local/bin/slidebasedvideo_cli

# ----- Start Ollama server by default -----
EXPOSE 11434

CMD ["ollama", "serve"]
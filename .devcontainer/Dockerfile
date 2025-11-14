FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 python3-pip python3.12-venv git ffmpeg nano \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /huggingface

ENV HF_HOME=/huggingface

WORKDIR /workspace

RUN python3 -m venv /venv

ENV PATH="/venv/bin:$PATH"

COPY requirements.txt /workspace/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY --from=ollama/ollama /bin/ollama /usr/local/bin/ollama

CMD ["/bin/bash"]

# skipnote

Work in progress...


## Install

You only require Docker installed.

If you want better performance, NVIDIA GPU is needed.

If you have a NVIDIA GPU, you must setup Docker to exploit it. Follow [these](#install-nvidia-toolkit-for-docker).

Then install Skipnote, creating an image `skipnote` based on current `src`:

```
docker build --tag skipnote .
```

Finally, follow [this](#run) to run!

### Install NVIDIA Toolkit for Docker 

#### Opensuse Tumbleweed

```
# 1️⃣ Install toolkit NVIDIA
sudo zypper install nvidia-container-toolkit

# 2️⃣ Configure Docker to use NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker

# 3️⃣ Restart Docker
sudo systemctl restart docker

# 4️⃣ Test
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see something like that:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 555.xx.xx    Driver Version: 555.xx.xx    CUDA Version: 12.4     |
+-----------------------------------------------------------------------------+
```

In this case you can use in Dockerfile:

```
# dev
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

# prod
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04
```

Otherwise, if you have CUDA 13:

```
# dev
FROM nvidia/cuda:13.0.0-cudnn9-devel-ubuntu22.04

# prod
FROM nvidia/cuda:13.0.0-cudnn9-runtime-ubuntu22.04
```


#### Set default runtime

If you see `runc` you are stil using regular Docker runtime.

```
docker info | grep -i "Default Runtime"
```

In order to set `nvidia` as default runtime (avoiding `--runtime=nvidia` in `docker run`):

```
sudo nano /etc/docker/daemon.json
```

and add `"default-runtime": "nvidia"`:

```
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

Then reboot or restart Docker:

```
sudo systemctl restart docker
```




## Run

### Slide-based Video

```
docker run -v <huggingface-cache>:/huggingface -v <ollama-directory>:/usr/share/ollama -v <input-video>:/input -v <output-directory-path>:/output \
--rm --gpus=all --runtime=nvidia --security-opt label=disable --name skipnote \
skipnote slidebasedvideo_cli --video-path /input --output-path /output --export-markdown \
--language <lang> --ollama-model <model> --transcriber-model <model> --transcriber-compute-type <type> --transcriber-beam-size <size>
```

> [!NOTE]
> Remove `--runtime=nvidia` if you do **not** have setup NVIDIA GPU toolkit.

> [!NOTE]
> You could also add `--post-processing` to `slidebasedvideo_cli` 

For example, on **Linux**:

```
docker run -v $HOME/.cache/huggingface:/huggingface -v /usr/share/ollama/.ollama:/usr/share/ollama -v <input-video>:/input -v <output-directory-path>:/output \
--rm --gpus=all --runtime=nvidia --security-opt label=disable --name skipnote \
skipnote slidebasedvideo_cli --video-path /input --output-path /output --export-markdown \
--language en --ollama-model gemma3:12b --transcriber-model large-v3-turbo --transcriber-compute-type int8_float16 --transcriber-beam-size 4
```

For example, on **Windows**:

```
docker run -v C:\Users\<YOU>\.cache\huggingface:/huggingface -v C:\Users\<YOU>\ollama:/usr/share/ollama/.ollama -v <input-video>:/input -v <output-directory-path>:/output \
--rm --gpus=all --runtime=nvidia --security-opt label=disable --name skipnote \
skipnote slidebasedvideo_cli --video-path /input --output-path /output --export-markdown \
--language en --ollama-model gemma3:12b --transcriber-model large-v3-turbo --transcriber-compute-type int8_float16 --transcriber-beam-size 4
```

## Wellknown Bug

`easyocr` + `faster-whisper` on GPU causes a silent segfault in C++ CUDA library. Use `easyocr` with `gpu=False`.
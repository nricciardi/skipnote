# skipnote

Work in progress...


## Install

### Docker

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




## Wellknown Bug

`easyocr` + `faster-whisper` on GPU causes a silent segfault in C++ CUDA library. Use `easyocr` with `gpu=False`.



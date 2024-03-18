FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},compute,display

SHELL [ "/bin/bash", "--login", "-c" ]

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

#install libs first
RUN apt-get update -q && \
    apt-get install -q -y \
    wget \
    python3.8-dev \
    python3-pip \
    python3.8-tk \
    git \
    ninja-build \
    ffmpeg libsm6 libxext6 libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# intall pytorch
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV PATH=${PATH}:/usr/local/cuda:/usr/local/cuda/bin
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip uninstall numpy scikit-image pandas matplotlib shapely setuptools urllib3 -y
RUN pip install numpy==1.20.0 scikit-image==0.19.3 pandas==1.4.4 matplotlib==3.6 shapely==1.8.5.post1 setuptools==59.5.0
RUN pip install scikit-learn pyquaternion cachetools descartes future tensorboard
RUN pip install IPython

# Install MMCV-series
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"

RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install mmdetection3d
RUN git clone --branch v0.17.1 --single-branch https://github.com/open-mmlab/mmdetection3d.git
WORKDIR /mmdetection3d
RUN pip install scipy==1.7.3 scikit-image==0.20.0
RUN pip install --no-cache-dir -v -e .

# Copy to python dist-package foldder
RUN cp -r /mmdetection3d/mmdet3d /usr/local/lib/python3.8/dist-packages

# Install UniAD from source
WORKDIR /
RUN git clone https://github.com/OpenDriveLab/UniAD.git
WORKDIR /UniAD
RUN pip install -r requirements.txt
# We need this old version of torchmetrics to run UniAD...
RUN pip install torchmetrics==0.8.2

# Link python to python3
RUN ln /usr/bin/python3 /usr/bin/python

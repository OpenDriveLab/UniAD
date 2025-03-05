# Installation
> Modified from bevformer and mmdetection3d.

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n uniad2.0 python=3.9 -y
conda activate uniad2.0
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**c. GCC: Make sure gcc>=5 in conda env.**
```shell
# If gcc is not installed:
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
# Eg: export PATH=/mnt/gcc-5.4/bin:$PATH
```

**d. CUDA: Before installing MMCV family, you need to set up the CUDA_HOME (for compiling some operators on the gpu).**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.8/
```


**e. Install mmcv-series packages.**
```shell
git clone https://github.com/open-mmlab/mmcv.git & cd mmcv
git checkout v1.6.0
export MMCV_WITH_OPS=1 MMCV_CUDA_ARGS=-std=c++17
pip install -v -e .
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6
```


**h. Install UniAD.**
```shell
cd ~
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
pip install -r requirements.txt
```


**i. Prepare pretrained weights.**
```shell
mkdir ckpts && cd ckpts

# Pretrained weights of bevformer
# Also the initial state of training stage1 model
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth

# Pretrained weights of stage1 model (perception part of UniAD)
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth

# Pretrained weights of stage2 model (fully functional UniAD)
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
```

---
-> Next Page: [Prepare The Dataset](./DATA_PREP.md)
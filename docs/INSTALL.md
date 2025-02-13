# Installation
> Modified from bevformer and mmdetection3d.

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n uniad python=3.9 -y
conda activate uniad
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
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
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install mmdet3d==1.0.0rc6
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
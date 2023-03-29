# Installation
Modified from bevformer and mmdetection3d.

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n uniad python=3.8 -y
conda activate uniad
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
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
# Eg: export CUDA_HOME=/mnt/cuda-11.1/
```


**e. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
# If it's not working, try:
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**f. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**g. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

<!-- **h. Install timm.**
```shell
pip install timm
``` -->

**h. Clone UniAD.**
```shell
git clone https://github.com/OpenDriveLab/UniAD.git
```

**i. Install the rest of the packages.**
```shell
cd UniAD
pip install -r requirements.txt --no-deps
# --no-deps is to avoid conflicts
# After installation, please make sure your numpy version is 1.20.0 by running `pip list | grep numpy`
```

**j. Prepare pretrained weights.**
```shell
cd UniAD
mkdir ckpts
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth
```
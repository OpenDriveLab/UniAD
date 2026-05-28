# UniAD 环境搭建指南（WSL2 + CUDA 11.8）

本文档记录在 Windows WSL2 环境下从零搭建 UniAD 2.0 开发环境的完整步骤，包含所有踩坑记录和解决方案。

---

## 系统要求

- Windows 10/11，已启用 WSL2
- NVIDIA GPU（本文使用的显卡支持 CUDA 11.8）
- 磁盘空间：约 30GB（代码 + 依赖 + 数据）

---

## 1. 安装 Miniconda 并创建虚拟环境

```bash
# 下载 Miniconda（Linux 版本）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载 shell
source ~/.bashrc

# 创建 Python 3.9 环境
conda create -n uniad2.0 python=3.9 -y
conda activate uniad2.0
```

---

## 2. 安装 PyTorch（CUDA 11.8）

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

验证安装：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 期望输出：2.0.1+cu118  True
```

---

## 3. 安装 OpenMMLab 依赖

### 3.1 安装 mmcv-full 1.6.1（需要编译，较耗时）

```bash
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

> **注意**：如果上面的预编译版本找不到，需要从源码编译：
> ```bash
> pip install openmim
> mim install mmcv-full==1.6.1
> ```
> 编译约需 10-30 分钟，取决于 CPU 性能。

### 3.2 安装 mmdet、mmsegmentation、mmdet3d

```bash
pip install mmdet==2.26.0
pip install mmsegmentation==0.29.1

# mmdet3d 需要从源码安装
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6
pip install -v -e .
cd ..
```

> ⚠️ **踩坑**：如果 mmdet3d 安装时报 Permission denied（cv2/load_config_py2.py 等文件权限错误），原因是之前用 root 安装过，执行：
> ```bash
> sudo chown -R $USER:$USER ~/miniconda3/envs/uniad2.0/
> ```

---

## 4. 克隆 UniAD 并安装依赖

```bash
git clone https://github.com/YOUR_USERNAME/UniAD.git  # 使用你 fork 的仓库
cd UniAD

# 安装依赖（注意 numpy 版本限制）
pip install -r requirements.txt
```

> ⚠️ **踩坑 1**：requirements.txt 中 `numpy==1.22.4` 会导致 matplotlib 报错。修复：
> ```bash
> pip install "numpy>=1.23,<2.0"
> ```

> ⚠️ **踩坑 2**：numba 0.53.0 与 numpy 1.26+ 不兼容。修复：
> ```bash
> pip install "numba>=0.56"
> ```

---

## 5. 准备 nuScenes 数据集

### 5.1 下载数据

从 [nuScenes 官网](https://www.nuscenes.org/nuscenes) 下载以下文件（需注册）：

- `v1.0-mini.tar.gz`（完整 mini 数据集，~4GB）
- `nuScenes-map-expansion-v1.3.zip`（地图扩展包）
- `can_bus.zip`（CAN bus 数据）

### 5.2 组织目录结构

```
UniAD/data/nuscenes/
├── v1.0-mini/        → 解压后的 v1.0-mini 目录（包含 samples/ sweeps/ maps/ v1.0-mini/）
├── samples/          → 软链接到 v1.0-mini/samples
├── sweeps/           → 软链接到 v1.0-mini/sweeps
├── maps/             → 软链接到 v1.0-mini/maps
│   └── expansion/    → 从 map-expansion 包中复制（4 个 JSON 文件）
├── v1.0-mini/        → 软链接到 v1.0-mini/v1.0-mini（元数据）
├── v1.0-trainval/    → 软链接到 v1.0-mini/v1.0-mini（workaround，见注释）
└── can_bus/          → 从 can_bus.zip 解压
```

创建软链接的命令：

```bash
cd ~/UniAD/data/nuscenes

# 假设 v1.0-mini 解压在当前目录
ln -s v1.0-mini/samples samples
ln -s v1.0-mini/sweeps sweeps
ln -s v1.0-mini/maps maps
ln -s v1.0-mini/v1.0-mini v1.0-mini_meta   # 仅示意

# 复制地图扩展文件
cp -r /path/to/nuScenes-map-expansion/maps/expansion maps/expansion

# 解压 CAN bus
unzip /path/to/can_bus.zip -d can_bus
```

> ⚠️ **踩坑**：val PKL 文件内部记录的 nuScenes 版本是 `v1.0-trainval`，但 mini 数据集没有这个目录。需要创建一个指向 mini 元数据的软链接：
> ```bash
> ln -s v1.0-mini/v1.0-mini v1.0-trainval
> ```

### 5.3 生成数据信息文件

```bash
cd ~/UniAD
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/infos \
    --extra-tag nuscenes_infos_temporal \
    --version v1.0-mini \
    --canbus ./data/nuscenes
```

生成：
- `data/infos/nuscenes_infos_temporal_train.pkl`（323 samples）
- `data/infos/nuscenes_infos_temporal_val.pkl`（81 samples）

---

## 6. Bug 修复（必须执行）

在运行评估前，需要修复几个代码兼容性问题。详见 [Bug 修复记录](bug_fixes.md)。

关键修复文件：
- `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py`（Shapely 2.x 兼容性）
- `nuscenes/eval/tracking/mot.py`（motmetrics 版本兼容性）

---

## 7. 验证安装

```bash
python -c "
import torch
import mmcv
import mmdet
import mmdet3d
print(f'torch: {torch.__version__}')
print(f'mmcv: {mmcv.__version__}')
print(f'mmdet: {mmdet.__version__}')
print(f'mmdet3d: {mmdet3d.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

期望输出：
```
torch: 2.0.1+cu118
mmcv: 1.6.1
mmdet: 2.26.0
mmdet3d: 1.0.0rc6
CUDA available: True
```

---

## 8. 运行评估

```bash
conda activate uniad2.0
cd ~/UniAD

# Stage 1（约 3-5 分钟）
./tools/uniad_dist_eval.sh \
    ./projects/configs/stage1_track_map/base_track_map.py \
    ./ckpts/uniad_base_track_map.pth 1

# Stage 2（约 5-10 分钟）
./tools/uniad_dist_eval.sh \
    ./projects/configs/stage2_e2e/base_e2e.py \
    ./ckpts/uniad_base_e2e.pth 1
```

---

## 依赖版本汇总

| 包 | 版本 | 备注 |
|----|------|------|
| python | 3.9 | |
| torch | 2.0.1+cu118 | |
| torchvision | 0.15.2+cu118 | |
| mmcv-full | 1.6.1 | 需编译或用预编译包 |
| mmdet | 2.26.0 | |
| mmsegmentation | 0.29.1 | |
| mmdet3d | 1.0.0rc6 | 从源码安装 |
| numpy | 1.23~1.26 | 不能用 1.22，不能用 2.x |
| numba | ≥0.56 | 0.53 与 numpy 1.26+ 不兼容 |
| shapely | 2.x | 需修复 map_api.py |
| nuscenes-devkit | 1.1.10 | 需修复 mot.py |

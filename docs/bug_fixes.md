# Bug 修复记录

本文档记录在复现 UniAD 2.0 过程中遇到的所有兼容性问题及修复方案。这些问题主要由新版本依赖库（Shapely 2.x、新版 motmetrics、新版 matplotlib、新版 numpy/numba）与 UniAD 原始代码的不兼容导致。

---

## Bug 1：mmdet3d 安装 Permission Denied

**文件**：`~/miniconda3/envs/uniad2.0/lib/python3.9/site-packages/cv2/load_config_py2.py`

**错误信息**：
```
PermissionError: [Errno 13] Permission denied: '.../cv2/load_config_py2.py'
```

**原因**：之前用 `sudo` 或 root 用户安装了某些包，导致 conda 环境目录部分文件归 root 所有。

**修复**：
```bash
sudo chown -R $USER:$USER ~/miniconda3/envs/uniad2.0/
```

---

## Bug 2：matplotlib seaborn-whitegrid 样式不存在

**文件**：`projects/mmdet3d_plugin/datasets/eval_utils/map_api.py`，约第 34 行

**错误信息**：
```
OSError: 'seaborn-whitegrid' is not a valid package style, path of style file, ...
```

**原因**：新版 matplotlib 将 seaborn 样式重命名为 `seaborn-v0_8-*`。

**修复**：
```python
# 修改前
plt.style.use('seaborn-whitegrid')

# 修改后
plt.style.use('seaborn-v0_8-whitegrid')
```

---

## Bug 3：数据库版本不存在 v1.0-trainval

**错误信息**：
```
Exception: Database version not found: data/nuscenes/v1.0-trainval
```

**原因**：val PKL 文件的元数据中记录的 nuScenes 版本字符串是 `v1.0-trainval`，但我们只有 mini 数据集，目录名是 `v1.0-mini`。

**修复**：创建一个软链接让代码找到对应目录：
```bash
cd ~/UniAD/data/nuscenes
ln -s v1.0-mini/v1.0-mini v1.0-trainval
```

---

## Bug 4：maps/expansion/ 目录不存在

**错误信息**：
```
FileNotFoundError: boston-seaport.json not found in maps/expansion/
```

**原因**：maps/expansion 软链接指向了自身（循环链接），或地图扩展包未正确部署。

**修复**：
1. 从 nuScenes 官网下载 `nuScenes-map-expansion-v1.3.zip`
2. 解压后将 `maps/expansion/` 目录（含 4 个 JSON 文件）复制到：
   ```bash
   cp -r nuScenes-map-expansion/maps/expansion ~/UniAD/data/nuscenes/maps/expansion
   ```
3. 确认 4 个文件存在：
   - `boston-seaport.json`
   - `singapore-hollandvillage.json`
   - `singapore-onenorth.json`
   - `singapore-queenstown.json`

---

## Bug 5：numpy 版本过低（matplotlib 报错）

**错误信息**：
```
ImportError: Matplotlib requires numpy>=1.23; you have 1.22.4
```

**原因**：`requirements.txt` 中硬性指定了 `numpy==1.22.4`，与新版 matplotlib 不兼容。

**修复**：
```bash
pip install "numpy>=1.23,<2.0"
```

> 不能升级到 numpy 2.x，因为 mmcv-full 1.6.1 与 numpy 2.x 不兼容。

---

## Bug 6：numba 与 numpy 版本不兼容

**错误信息**：
```
SystemError: <class 'numba.core.compiler.Compiler'> returned a result with an error...
```

**原因**：numba 0.53.0 不支持 numpy 1.26+。

**修复**：
```bash
pip install "numba>=0.56"
```

---

## Bug 7：Shapely 2.x MultiPolygon 不可迭代

**文件**：`projects/mmdet3d_plugin/datasets/eval_utils/map_api.py`，约第 2097 行

**错误信息**：
```
TypeError: 'MultiPolygon' object is not iterable
```

**原因**：Shapely 2.x 移除了 `MultiPolygon` 和 `MultiLineString` 的直接迭代支持，必须通过 `.geoms` 属性访问子几何体。

**修复 1（MultiPolygon）**：
```python
# 修改前（约第 2097 行附近）
exteriors = [int_coords(poly.exterior.coords) for poly in polygons]

# 修改后
from shapely.geometry import MultiPolygon as MP
polygons = list(polygons.geoms) if isinstance(polygons, MP) else polygons
exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
```

**修复 2（MultiLineString）**：
```python
# 修改前（约第 2114 行附近）
for line in lines:
    ...

# 修改后
for line in lines.geoms:
    ...
```

---

## Bug 8：val PKL 与 mini 数据集 token 不匹配

**错误信息**：
```
KeyError: 'cab8d...' (某 sample token 在数据集中找不到)
```

**原因**：从网上下载的 val PKL 是完整 trainval 数据集的 info 文件，token 与 mini 数据集不一致。

**修复**：用 mini 数据集重新生成 PKL 文件（需要先部署好 CAN bus 数据）：
```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/infos \
    --extra-tag nuscenes_infos_temporal \
    --version v1.0-mini \
    --canbus ./data/nuscenes
```

---

## Bug 9：CAN bus 目录不存在

**错误信息**：
```
FileNotFoundError: CAN bus directory not found
```

**原因**：生成数据 PKL 时需要 CAN bus 扩展数据，但未下载。

**修复**：
1. 从 nuScenes 官网下载 `can_bus.zip`
2. 解压到正确位置：
   ```bash
   unzip can_bus.zip -d ~/UniAD/data/nuscenes/can_bus
   ```

---

## Bug 10：motmetrics `_events` 类型错误

**文件**：`nuscenes/eval/tracking/mot.py`（nuscenes-devkit 安装路径）

**错误信息**：
```
TypeError: list indices must be integers or slices, not str
```

**原因**：新版 motmetrics（0.60.0+）将内部 `_events` 从 `dict`（键为 `Type/OId/HId/D`）改为 `list of lists`（每项为 `[Type, OId, HId, D]`），`_indices` 从 `dict` 改为 `list of tuples`。nuscenes-devkit 的 mot.py 仍用旧 API。

**修复**（修改 `mot.py` 中处理 `_events` 和 `_indices` 的代码段）：

```python
# 处理 _events
events = frame_events._events
if isinstance(events, dict):
    # 旧版 motmetrics
    types = events['Type']
    oids = events['OId']
    hids = events['HId']
    ds = events['D']
else:
    # 新版 motmetrics（list of lists）
    types = [e[0] for e in events]
    oids = [e[1] for e in events]
    hids = [e[2] for e in events]
    ds = [e[3] for e in events]

# 处理 _indices
indices = frame_events._indices
if isinstance(indices, dict):
    idx = pd.MultiIndex.from_arrays(
        [indices[field] for field in _INDEX_FIELDS],
        names=_INDEX_FIELDS)
else:
    # 新版 motmetrics（list of tuples）
    idx = pd.MultiIndex.from_tuples(indices, names=_INDEX_FIELDS)
```

> 修改文件路径：`~/miniconda3/envs/uniad2.0/lib/python3.9/site-packages/nuscenes/eval/tracking/mot.py`

---

## 修复文件汇总

| 文件 | 修复内容 | Bug 编号 |
|------|----------|---------|
| `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` | seaborn 样式名 | #2 |
| `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` | MultiPolygon.geoms | #7 |
| `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` | MultiLineString.geoms | #7 |
| `nuscenes/eval/tracking/mot.py` | _events 列表格式 | #10 |
| `nuscenes/eval/tracking/mot.py` | _indices 列表格式 | #10 |
| `data/nuscenes/v1.0-trainval` (symlink) | 版本目录 workaround | #3 |

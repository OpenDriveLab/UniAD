<div align="center">   
  
# Planning-oriented Autonomous Driving
</div>

<!-- <p align="center">
 <a href="https://opendrivelab.github.io/UniAD/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-Open-yellowgreen.svg" target="_blank" />
  </a>
  <a href="https://github.com/OpenDriveLab/UniAD/blob/master/LICENSE">
    <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" target="_blank" />
  </a>
  <a href="https://github.com/OpenDriveLab/UniAD/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
    <img alt="Good first issue" src="https://img.shields.io/github/issues/OpenDriveLab/UniAD/good%20first%20issue" target="_blank" />
  </a>
</p> -->

<h3 align="center">
  <a href="https://opendrivelab.github.io/UniAD/">Project Page</a> |
  <a href="https://arxiv.org/abs/2212.10156">arXiv</a> |
  <a href="https://opendrivelab.com/">OpenDriveLab</a>
  
</h3>

https://user-images.githubusercontent.com/48089846/202974395-15fe83ac-eebb-4f38-8172-b8ca8c65127e.mp4

<br><br>

![teaser](sources/pipeline.png)

## Table of Contents:
1. [Highlights](#high)
2. [News](#news)
3. [Getting Started](#start)
   - [Installation](docs/INSTALL.md)
   - [Prepare Dataset](docs/DATA_PREP.md)
   - [Evaluation Example](docs/TRAIN_EVAL.md)
   - [GPU Requirements](docs/TRAIN_EVAL.md)
   - [Train/Eval](docs/TRAIN_EVAL.md)
4. [Results and Models](#models)
5. [TODO List](#todos)
6. [License](#license)
7. [Citation](#citation)

## Highlights <a name="high"></a>

- :oncoming_automobile: **Planning-oriented philosophy**: UniAD is a Unified Autonomous Driving algorithm framework following a planning-oriented philosophy. Instead of standalone modular design and multi-task learning, we cast a series of tasks, including perception, prediction and planning tasks hierarchically.
- :trophy: **SOTA performance**: All tasks within UniAD achieve SOTA performance, especially prediction and planning (motion: 0.71m minADE, occ: 63.4% IoU, planning: 0.31% avg.Col)

## News <a name="news"></a>

- **`Paper Title Change`**: To avoid confusion with the "goal-point" navigation in Robotics, we change the title from "Goal-oriented" to "Planning-oriented" suggested by Reviewers. Thank you!
- [2023/04] **_Estimated_**. Model checkpoints release `v2.0`


- [2023/03/29] Code & model initial release `v1.0`
- [2023/03/21] :rocket::rocket: UniAD is accepted by CVPR 2023, as an **Award Candidate** (12 out of 2360 accepted papers)!
- [2022/12/21] UniAD [paper](https://arxiv.org/abs/2212.10156) is available on arXiv.



<!-- ## Table of Contents:
1. [Installation](docs/INSTALL.md)
2. [Prepare Data](docs/DATA_PREP.md)
3. [Evaluation Example](docs/TRAIN_EVAL.md#example)
4. [UniAD Training](docs/TRAIN_EVAL.md#train)
5. [UniAD Evaluation](docs/TRAIN_EVAL.md#eval)
6. [Results and Models](#models)
7. [TODO List](#todos)
7. [License](#license)
8. [Citing](#citation) -->


## Getting Started <a name="start"></a>
- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/DATA_PREP.md)
- [Evaluation Example](docs/TRAIN_EVAL.md)
- [GPU Requirements](docs/TRAIN_EVAL.md)
- [Train/Eval](docs/TRAIN_EVAL.md)

## Results and Pre-trained Models <a name="models"></a>
UniAD is trained in two stages. Pretrained checkpoints of both stages will be released and the results of each model are listed in the following tables.

### Stage-one: Perception training
> We first train the perception modules (i.e., track and map) to obtain a stable initlization for the next stage.

| Method | Encoder | Tracking<br>AMOTA | Mapping<br>IoU-lane | config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| 
| UniAD-S | R50 | -  | - | TBA | TBA |
| UniAD-B | R101 | 0.390 | 0.297 |  [base-stage1](projects/configs/track_map/base_stage1.py) | [base-stage1](https://github.com/OpenDriveLab/UniAD/releases/download/untagged-d7e1d5e20eded789eee9/uniad_base_track_map.pth) |
| UniAD-L | V2-99 | - | - | TBA | TBA |



### Stage-two: End-to-end training
> We optimize all task modules together, including track, map, motion, occupancy and planning.

<!-- 
Pre-trained models and results under main metrics are provided below. We refer you to the [paper](https://arxiv.org/abs/2212.10156) for more details. -->

| Method | Encoder | Tracking<br>AMOTA | Mapping<br>IoU-lane | Motion<br>minADE |Occupancy<br>IoU-n. | Planning<br>avg.Col. | config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: | :---: |
| UniAD-S | R50 | 0.241  | 0.315 | 0.788 | 59.4  | 0.32 | TBA | TBA |
| UniAD-B | R101 | 0.359 | 0.313 | 0.708 | 63.4 | 0.31 |  TBA | TBA |
| UniAD-L | V2-99 | 0.409 | 0.323 | 0.723 | 64.1 | 0.29 | TBA | TBA |

### Checkpoint Usage
* Download the checkpoints you need into `UniAD/ckpts/` directory.
* You can evaluate these checkpoints to reproduce the results, following the `evaluation` section in [TRAIN_EVAL.md](docs/TRAIN_EVAL.md).
* You can also initialize your own model with the provided weights. Change the `load_from` field to `path/of/ckpt` in the config and follow the `train` section in [TRAIN_EVAL.md](docs/TRAIN_EVAL.md) to start training.


### Model Structure
The overall pipeline of UniAD is controlled by [uniad_e2e.py](projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py) which coordinates all the task modules in `UniAD/projects/mmdet3d_plugin/uniad/dense_heads`. If you are interested in the implementation of a specific task module, please refer to its corresponding file, e.g., [motion_head](projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py).

## TODO List <a name="todos"></a>
- [ ] Base-model configs & checkpoints [Est. 2023/04]
- [ ] Separating BEV encoder and tracking module [Est. 2023/04]
- [ ] Support larger batch size [Est. 2023/04]
- [ ] (Long-term) Improve flexibility for future extensions
- [ ] All configs & checkpoints
- [x] Code initialization


## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{hu2023_uniad,
 title={Planning-oriented Autonomous Driving}, 
 author={Yihan Hu and Jiazhi Yang and Li Chen and Keyu Li and Chonghao Sima and Xizhou Zhu and Siqi Chai and Senyao Du and Tianwei Lin and Wenhai Wang and Lewei Lu and Xiaosong Jia and Qiang Liu and Jifeng Dai and Yu Qiao and Hongyang Li},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 year={2023},
}
```
## Related resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) (:rocket:Ours!)
- [ST-P3](https://github.com/OpenPerceptionX/ST-P3) (:rocket:Ours!)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [FIERY](https://github.com/wayveai/fiery)
- [MOTR](https://github.com/megvii-research/MOTR)
- [BEVerse](https://github.com/zhangyp15/BEVerse)

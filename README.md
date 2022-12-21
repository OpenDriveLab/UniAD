<div align="center">   
  
# Goal-oriented Autonomous Driving
</div>

<p align="center">
 <a href="">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-Open-blue.svg" target="_blank" />
  </a>
  <a href="https://github.com/OpenPerceptionX/UniAD/blob/master/LICENSE">
    <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg" target="_blank" />
  </a>
  <a href="https://github.com/OpenPerceptionX/UniAD/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
    <img alt="Good first issue" src="https://img.shields.io/github/issues/OpenPerceptionX/UniAD/good%20first%20issue" target="_blank" />
  </a>
</p>

<h3 align="center">
  <a href="https://opendrivelab.github.io/UniAD/">project page</a> |
  <a href="https://arxiv.org/abs/2212.10156">arXiv</a> |
  <a href="">video</a> 
</h3>

https://user-images.githubusercontent.com/48089846/202974395-15fe83ac-eebb-4f38-8172-b8ca8c65127e.mp4

This repository will host the code of UniAD.

> Goal-oriented Autonomous Driving
>
> Yihan Hu*, Jiazhi Yang*, [Li Chen*](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en&authuser=1), Keyu Li*, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, [Hongyang Li](https://lihongyang.info/)
> - Primary contact: Li Chen ( lichen@pjlab.org.cn )

![teaser](sources/pipeline.png)

## Highlights

- :oncoming_automobile: **Goal-oriented philosophy**: UniAD is a Unified Autonomous Driving algorithm framework devised following a goal-oriented philosophy. Instead of standalone modular design and multi-task learning, perception, prediciton and planning tasks/components should opt in and be prioritized hierarchically, and we demonstrate the performance can be enhanced to a new level.
- :trophy: **SOTA performance**: All tasks among UniAD achieve SOTA performance, especially prediction and planning (motion: 0.71m minADE, occ: 63.4% IoU-n., plan: 0.31% avg.Col)

## News

- [] UniAD [paper](https://arxiv.org/abs/2212.10156) is available on arXiv! Please stay tuned for code release!

<!-- 
## Getting started

- [Installation]()
- [Dataset preparation]()
- [Train and eval]()
-->

## Main results

Pre-trained models and results under main metrics are provided below. We refer you to the [paper](https://arxiv.org/abs/2212.10156) for more details.

## License

All assets and code are under the [Apache 2.0 license](https://github.com/OpenPerceptionX/UniAD/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

```
@article{uniad,
 title={Goal-oriented Autonomous Driving}, 
 author={Yihan Hu and Jiazhi Yang and Li Chen and Keyu Li and Chonghao Sima and Xizhou Zhu and Siqi Chai and Senyao Du and Tianwei Lin and Wenhai Wang and Lewei Lu and Xiaosong Jia and Qiang Liu and Jifeng Dai and Yu Qiao and Hongyang Li},
 journal={arXiv preprint arXiv:},
 year={2022},
}
```
## Related resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) (:rocket:Ours!)
- [ST-P3](https://github.com/OpenPerceptionX/ST-P3) (:rocket:Ours!)
- [FIERY](https://github.com/wayveai/fiery)
- [MOTR](https://github.com/megvii-research/MOTR)
- [BEVerse](https://github.com/zhangyp15/BEVerse)
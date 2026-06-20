#!/usr/bin/env python
#---------------------------------------------------------------------------------#
# UniAD 评测+误差归因工作台 - Stage 1.5: 归因图表                                   #
# 读 label_scene_attribution.py 的带标签 CSV, 画 路口 vs 直路 的 L2-随时域 图,      #
# 并打印 worst-K 路口率等 README 用的关键数字。                                     #
#---------------------------------------------------------------------------------#
import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='路口/直路 L2 归因图表')
    p.add_argument('--csv', default='output/planning_per_frame_attr.csv',
                   help='label_scene_attribution.py 输出的带标签 CSV')
    p.add_argument('--out', default='docs/stage1_demo/intersection_l2.png')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    valid = df[df['l2_3s'] > 0]  # 剔除 3s-GT 被 mask 的 scene-end 帧
    inter = valid[valid['is_intersection']]
    straight = valid[~valid['is_intersection']]

    horizons = [1, 2, 3]
    cols = ['l2_1s', 'l2_2s', 'l2_3s']
    inter_mean = [inter[c].mean() for c in cols]
    straight_mean = [straight[c].mean() for c in cols]
    inter_p90 = [np.percentile(inter[c], 90) for c in cols]
    straight_p90 = [np.percentile(straight[c], 90) for c in cols]

    # ---- 画图: mean (实线) + p90 (虚线), 路口 vs 直路 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(horizons, inter_mean, 'o-', color='#d62728', lw=2, label='Intersection (mean)')
    ax.plot(horizons, straight_mean, 'o-', color='#1f77b4', lw=2, label='Straight road (mean)')
    ax.plot(horizons, inter_p90, 'o--', color='#d62728', alpha=0.5, label='Intersection (p90)')
    ax.plot(horizons, straight_p90, 'o--', color='#1f77b4', alpha=0.5, label='Straight road (p90)')
    ax.set_xticks(horizons)
    ax.set_xlabel('Planning horizon (s)')
    ax.set_ylabel('Planning L2 error (m)')
    ax.set_title('UniAD planning error: intersection vs straight road\n'
                 '(nuScenes mini, single-trajectory regression diverges at intersections)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'已保存图表: {args.out}')

    # ---- README 用的关键数字 ----
    print('\n=== 关键数字 ===')
    print(f'有效帧 {len(valid)}: 路口 {len(inter)} / 直路 {len(straight)} '
          f'(路口基率 {len(inter)/len(valid)*100:.0f}%)')
    for c, im, sm in zip(cols, inter_mean, straight_mean):
        print(f'  {c}: 路口 {im:.3f} / 直路 {sm:.3f} = {im/sm:.2f}x')
    print('worst-K 路口率 (在全部 81 帧按 l2_3s 降序):')
    ranked = df.sort_values('l2_3s', ascending=False)
    for k in (5, 10, 15, 20):
        sub = ranked.head(k)
        print(f'  top-{k}: {int(sub["is_intersection"].sum())}/{k} '
              f'= {sub["is_intersection"].mean()*100:.0f}%')


if __name__ == '__main__':
    main()

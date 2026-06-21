#!/usr/bin/env python
#---------------------------------------------------------------------------------#
# UniAD 评测+误差归因工作台 - Stage 1.5: 场景标签归因 (路口 vs 直路)               #
#                                                                                 #
# 假设 (来自 FiftyOne 肉眼观察): 长时域 planning 失效 (l2_3s 飘) 集中在"路口"——    #
# 路口未来多模态, 而 UniAD planning head 是单轨迹回归 -> 1s 准、3s 发散。          #
#                                                                                 #
# 本脚本用 nuScenes map API 给每帧打"是否路口"标签 (ego 位置半径内是否有           #
# road_segment.is_intersection), 然后把帧按 路口/直路 分桶, 对比 l2_1s vs l2_3s,  #
# 量化验证: 是否路口帧的 l2_3s 显著更高, 且差距随时域增大。                         #
#                                                                                 #
# 注意: l2_3s==0 的帧是 3s GT 被 mask 的 scene-end 帧 (非真实误差), 分桶时剔除。   #
#                                                                                 #
# 用法: python tools/label_scene_attribution.py --csv output/planning_per_frame.csv #
#---------------------------------------------------------------------------------#
import argparse
import os.path as osp

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='路口/直路场景标签归因')
    parser.add_argument('--csv', default='output/planning_per_frame.csv',
                        help='逐帧 planning 指标 CSV')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--radius', type=float, default=3.0,
                        help='ego 位置查路口的半径 (米, 默认 3.0)')
    parser.add_argument('--out', default=None,
                        help='输出带标签的 CSV (默认 <csv>_attr.csv)')
    return parser.parse_args()


def build_labeler(dataroot, version, radius):
    """返回 token -> (is_intersection, location) 的函数。"""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    maps = {}

    def label(token):
        s = nusc.get('sample', token)
        sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        x, y = pose['translation'][0], pose['translation'][1]
        log = nusc.get('log', nusc.get('scene', s['scene_token'])['log_token'])
        loc = log['location']
        if loc not in maps:
            maps[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)
        m = maps[loc]
        recs = m.get_records_in_radius(x, y, radius, ['road_segment'])['road_segment']
        is_inter = any(bool(m.get('road_segment', t)['is_intersection']) for t in recs)
        return is_inter, loc

    return label


def bucket_stats(sub, col):
    if len(sub) == 0:
        return dict(n=0, mean=float('nan'), median=float('nan'), p90=float('nan'))
    v = sub[col].values
    return dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)))


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    print(f'读入 {len(df)} 帧: {args.csv}')

    label = build_labeler(args.dataroot, args.version, args.radius)
    inters, locs, fails = [], [], 0
    for token in df['token']:
        try:
            is_inter, loc = label(token)
        except Exception as e:
            is_inter, loc, fails = False, '', fails + 1
        inters.append(is_inter)
        locs.append(loc)
    df['is_intersection'] = inters
    df['map_location'] = locs
    # l2_3s==0 => 3s GT 被 mask (scene-end), 非真实误差, 标记剔除
    df['valid_3s'] = df['l2_3s'] > 0
    if fails:
        print(f'  [warn] {fails} 个 token map 查询失败, 记为 非路口')

    out = args.out or args.csv.replace('.csv', '_attr.csv')
    df.to_csv(out, index=False)

    # ---- 分桶对比 (仅用 valid_3s 帧) ----
    valid = df[df['valid_3s']]
    n_masked = len(df) - len(valid)
    inter = valid[valid['is_intersection']]
    straight = valid[~valid['is_intersection']]

    print(f'\n=== 路口 vs 直路 归因 (半径 {args.radius}m, '
          f'剔除 {n_masked} 个 3s-GT 被 mask 的帧) ===')
    print(f'有效帧 {len(valid)}: 路口 {len(inter)} / 直路 {len(straight)}')
    print(f"\n{'指标':<8} {'分桶':<8} {'n':>4} {'mean':>8} {'median':>8} {'p90':>8}")
    for col in ['l2_1s', 'l2_2s', 'l2_3s']:
        for name, sub in [('路口', inter), ('直路', straight)]:
            st = bucket_stats(sub, col)
            print(f"{col:<8} {name:<8} {st['n']:>4} {st['mean']:>8.3f} "
                  f"{st['median']:>8.3f} {st['p90']:>8.3f}")

    # ---- 关键结论: 路口/直路 的 mean 比值, 看是否随时域放大 ----
    print('\n=== 路口 mean / 直路 mean (>1 = 路口更差; 随时域放大 = 支持假设) ===')
    for col in ['l2_1s', 'l2_2s', 'l2_3s']:
        mi = bucket_stats(inter, col)['mean']
        ms = bucket_stats(straight, col)['mean']
        ratio = mi / ms if ms > 0 else float('nan')
        print(f'  {col}: 路口 {mi:.3f} / 直路 {ms:.3f} = {ratio:.2f}x')

    print(f'\n已导出带标签 CSV: {out}')
    print('注: mini 仅 ~81 帧, 统计不追求显著性结论, 重点是管线与方向正确。')


if __name__ == '__main__':
    main()

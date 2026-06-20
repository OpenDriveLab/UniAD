#!/usr/bin/env python
#---------------------------------------------------------------------------------#
# UniAD 评测+误差归因工作台 - Stage 1: 难例挖掘                                     #
# 从逐帧 planning 指标 CSV 中按指定指标排出 worst-K 帧并导出。                      #
#                                                                                 #
# 输入: tools/test.py 导出的 planning_per_frame.csv                                #
#   列: token, command, L2_0.5..L2_3.0, l2_1s/2s/3s, col_0.5..col_3.0, col_any    #
# 输出: worst-K 帧的子表 (CSV), 供后续根因归因 / FiftyOne 难例浏览器使用。          #
#---------------------------------------------------------------------------------#
import argparse
import os.path as osp

import pandas as pd

# UniAD high-level command 编码 (planning_gt['command'])
COMMAND_NAMES = {0: 'RIGHT', 1: 'LEFT', 2: 'FORWARD'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='按 planning 指标挖掘 worst-K 难例帧')
    parser.add_argument('--csv', default='output/planning_per_frame.csv',
                        help='逐帧 planning 指标 CSV (tools/test.py 导出)')
    parser.add_argument('--k', type=int, default=20,
                        help='导出最差的 K 帧 (默认 20)')
    parser.add_argument('--sort-by', default='l2_3s',
                        help='排序依据列 (默认 l2_3s; 可选 l2_1s/l2_2s/L2_3.0/col_any 等)')
    parser.add_argument('--collision-first', action='store_true',
                        help='先把碰撞帧 (col_any=1) 排到最前, 再按 --sort-by 排')
    parser.add_argument('--out', default=None,
                        help='worst-K 输出 CSV 路径 (默认 <csv目录>/worst_<k>_<sortby>.csv)')
    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.isfile(args.csv):
        raise FileNotFoundError(f'找不到输入 CSV: {args.csv} (先跑一遍 eval 生成它)')

    df = pd.read_csv(args.csv)
    if args.sort_by not in df.columns:
        raise ValueError(
            f'--sort-by="{args.sort_by}" 不在列中. 可用列: {list(df.columns)}')

    n = len(df)
    k = min(args.k, n)

    # ---- 全局分布概览 (帮助判断 worst-K 阈值是否极端) ----
    print(f'\n输入: {args.csv}  (共 {n} 帧)')
    desc = df['l2_3s'].describe(percentiles=[0.5, 0.9, 0.95])
    print(f"l2_3s 分布: mean={desc['mean']:.3f}  median={desc['50%']:.3f}  "
          f"p90={desc['90%']:.3f}  p95={desc['95%']:.3f}  max={desc['max']:.3f}")
    n_col = int((df['col_any'] > 0).sum()) if 'col_any' in df.columns else 0
    print(f"碰撞帧 (col_any=1): {n_col} / {n}")

    # ---- 排序: 可选先碰撞优先, 再按指标降序 ----
    sort_cols, ascending = [args.sort_by], [False]
    if args.collision_first and 'col_any' in df.columns:
        sort_cols = ['col_any'] + sort_cols
        ascending = [False] + ascending
    worst = df.sort_values(by=sort_cols, ascending=ascending).head(k).reset_index(drop=True)

    # ---- 输出文件 ----
    out = args.out
    if out is None:
        out = osp.join(osp.dirname(osp.abspath(args.csv)),
                       f'worst_{k}_{args.sort_by}.csv')
    worst.to_csv(out, index=False)

    # ---- 打印 worst-K 概览 ----
    show_cols = ['token', 'command', 'l2_1s', 'l2_2s', 'l2_3s', 'col_any']
    show_cols = [c for c in show_cols if c in worst.columns]
    print(f'\n=== Worst-{k} 帧 (按 {args.sort_by} 降序'
          f"{', 碰撞优先' if args.collision_first else ''}) ===")
    header = f"{'rank':>4}  {'token':<34} {'cmd':>7}  {'l2_1s':>6} {'l2_2s':>6} {'l2_3s':>6} {'col':>3}"
    print(header)
    for i, row in worst.iterrows():
        cmd = COMMAND_NAMES.get(int(row['command']), str(row['command'])) \
            if 'command' in row else ''
        print(f"{i+1:>4}  {str(row['token']):<34} {cmd:>7}  "
              f"{row.get('l2_1s', float('nan')):>6.3f} "
              f"{row.get('l2_2s', float('nan')):>6.3f} "
              f"{row.get('l2_3s', float('nan')):>6.3f} "
              f"{int(row.get('col_any', 0)):>3}")
    print(f'\n已导出 worst-{k} 到: {out}')


if __name__ == '__main__':
    main()

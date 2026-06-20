#!/usr/bin/env python
#---------------------------------------------------------------------------------#
# UniAD 评测+误差归因工作台 - Stage 1: FiftyOne 难例浏览器                          #
#                                                                                 #
# 把 worst-K 难例帧的 6 路相机图像导入 FiftyOne (grouped dataset, 每帧一个 group, #
# 6 个相机 slice), 并附上 l2_1s/2s/3s、command、rank、col_any 字段, 用眼睛看清     #
# 这些帧为什么规划失效。                                                            #
#                                                                                 #
# 输入: tools/mine_hard_cases.py 导出的 worst_K CSV (含 token/l2_3s/command...)    #
# 用法:                                                                            #
#   # 仅构建 (持久化) 数据集, 不开 app, 适合先验证管线:                             #
#   python tools/fiftyone_hard_cases.py --worst-csv output/worst_15_l2_3s.csv     #
#   # 构建并打开浏览器 app (在你自己的终端用 ! 前缀跑, 这样浏览器能弹出):           #
#   python tools/fiftyone_hard_cases.py --launch                                  #
#---------------------------------------------------------------------------------#
import argparse
import os.path as osp

import pandas as pd

# 6 路相机, 按 前左/前/前右 - 后左/后/后右 排列, 浏览时方位直观
CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
COMMAND_NAMES = {0: 'RIGHT', 1: 'LEFT', 2: 'FORWARD'}


def parse_args():
    parser = argparse.ArgumentParser(description='worst-K 难例帧导入 FiftyOne')
    parser.add_argument('--worst-csv', default='output/worst_15_l2_3s.csv',
                        help='worst-K CSV (tools/mine_hard_cases.py 导出)')
    parser.add_argument('--dataroot', default='data/nuscenes',
                        help='nuScenes 数据根目录 (软链即可)')
    parser.add_argument('--version', default='v1.0-mini', help='nuScenes 版本')
    parser.add_argument('--name', default='uniad_hard_cases',
                        help='FiftyOne 数据集名 (持久化)')
    parser.add_argument('--overwrite', action='store_true',
                        help='同名数据集已存在时先删除重建')
    parser.add_argument('--launch', action='store_true',
                        help='构建后打开 FiftyOne app (需在交互终端运行)')
    parser.add_argument('--port', type=int, default=5151, help='app 端口')
    return parser.parse_args()


def main():
    args = parse_args()
    import fiftyone as fo
    from nuscenes.nuscenes import NuScenes

    if not osp.isfile(args.worst_csv):
        raise FileNotFoundError(
            f'找不到 worst-K CSV: {args.worst_csv} (先跑 tools/mine_hard_cases.py)')
    df = pd.read_csv(args.worst_csv)
    print(f'读入 {len(df)} 个难例帧: {args.worst_csv}')

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # 处理同名数据集
    if fo.dataset_exists(args.name):
        if args.overwrite:
            fo.delete_dataset(args.name)
        else:
            raise RuntimeError(
                f'数据集 "{args.name}" 已存在, 加 --overwrite 重建, 或换 --name')

    dataset = fo.Dataset(args.name)
    dataset.add_group_field('group', default='CAM_FRONT')

    samples = []
    n_missing = 0
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        token = row.token
        try:
            sample_rec = nusc.get('sample', token)
        except KeyError:
            print(f'  [warn] token 不在该 nuScenes 版本: {token}')
            continue
        cmd_int = int(row.command)
        group = fo.Group()
        for cam in CAMERAS:
            sd = nusc.get('sample_data', sample_rec['data'][cam])
            filepath = osp.abspath(osp.join(args.dataroot, sd['filename']))
            if not osp.exists(filepath):
                n_missing += 1
                continue
            s = fo.Sample(filepath=filepath, group=group.element(cam))
            s['token'] = token
            s['camera'] = cam
            s['rank'] = rank
            s['command'] = COMMAND_NAMES.get(cmd_int, str(cmd_int))
            s['l2_1s'] = float(row.l2_1s)
            s['l2_2s'] = float(row.l2_2s)
            s['l2_3s'] = float(row.l2_3s)
            s['col_any'] = bool(int(getattr(row, 'col_any', 0)))
            # 场景归因标签 (若 CSV 来自 label_scene_attribution.py 则带这些列)
            if 'is_intersection' in df.columns:
                s['is_intersection'] = bool(row.is_intersection)
            if 'map_location' in df.columns:
                s['map_location'] = str(row.map_location)
            samples.append(s)

    dataset.add_samples(samples)
    dataset.persistent = True
    if n_missing:
        print(f'  [warn] {n_missing} 张图缺失 (已跳过)')
    print(f'已构建 FiftyOne 数据集 "{args.name}": '
          f'{len(df)} 帧 x {len(CAMERAS)} 相机, 共 {len(samples)} 张图 (持久化)')
    print(f'按 l2_3s 降序排 (rank 字段), 字段: token/camera/rank/command/'
          f'l2_1s/l2_2s/l2_3s/col_any')

    if args.launch:
        print(f'\n打开 FiftyOne app: http://localhost:{args.port}')
        session = fo.launch_app(dataset, port=args.port)
        session.wait()
    else:
        print('\n下次直接看 (在交互终端):')
        print(f'  python -c "import fiftyone as fo; '
              f'fo.launch_app(fo.load_dataset(\'{args.name}\')).wait()"')


if __name__ == '__main__':
    main()

import argparse
from os import path as osp
import sys
from data_converter import uniad_nuscenes_converter as nuscenes_converter
sys.path.append('.')

#------------------------------数据集预处理函数--------------------------------------
def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    #见开头from data_converter import uniad_nuscenes_converter as nuscenes_converter
    #即用的都是./data_converter/uniad_nuscenes_converter中的函数


    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_test.pkl')   #(文件名前缀信息)
        nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)




#------------------------------解析参数--------------------------------------
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example') #Number of input consecutive frames. Default: 10
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument('--workers', type=int, default=4, help='number of threads to be used')

args = parser.parse_args()




#------------------------------函数入口--------------------------------------
if __name__ == '__main__':

    #处理完整的nuscenes v1.0数据集
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        #------------#v1.0-trainval数据集------------
        train_version = f'{args.version}-trainval'                 
        nuscenes_data_prep(
            root_path=args.root_path,         #e.g:  ./data/nuscenes
            can_bus_root_path=args.canbus,    #e.g:  ./data/nuscenes
            info_prefix=args.extra_tag,       #e.g:  nuscenes (文件名前缀信息)
            version=train_version,            #e.g:  v1.0-trainval
            dataset_name='NuScenesDataset',   #e.g:  'NuScenesDataset'
            out_dir=args.out_dir,             #e.g:  ./data/infos
            max_sweeps=args.max_sweeps)       #e.g:  10
        #有的参数你虽然没有在sh脚本中设置，但是往上看，解析函数或者这个函数的定义中对于这些参数有默认值

        #-------------#v1.0-test数据集--------------
        test_version = f'{args.version}-test'                       
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)

    #处理nuscenes v1.0-mini数据集
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
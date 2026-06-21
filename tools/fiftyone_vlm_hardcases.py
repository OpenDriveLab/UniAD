import argparse, os
import pandas as pd
import fiftyone as fo
from nuscenes.nuscenes import NuScenes

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

def as_bool(v):
    if pd.isna(v): return False
    if isinstance(v, str): return v.strip().lower() in ("true", "1", "yes")
    return bool(v)

def as_str(v):
    return "" if pd.isna(v) else str(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.expanduser("~/UniAD/output/vlm_attribution.csv"))
    ap.add_argument("--dataroot", default=os.path.expanduser("~/UniAD/data/nuscenes"))
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--name", default="uniad_vlm_hardcases")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if fo.dataset_exists(args.name):
        fo.delete_dataset(args.name)
    dataset = fo.Dataset(args.name)
    dataset.add_group_field("group", default="CAM_FRONT")

    samples = []
    for _, row in df.iterrows():
        token = str(row["token"])
        rec = nusc.get("sample", token)
        group = fo.Group()
        map_int = as_bool(row.get("map_is_intersection"))
        vlm_int = as_bool(row.get("vlm_is_intersection"))
        for cam in CAMERAS:
            sd = nusc.get("sample_data", rec["data"][cam])
            path = os.path.join(args.dataroot, sd["filename"])
            s = fo.Sample(filepath=path, group=group.element(cam))
            s["token"] = token
            s["l2_3s"] = float(row.get("l2_3s")) if not pd.isna(row.get("l2_3s")) else None
            s["map_is_intersection"] = map_int
            s["vlm_is_intersection"] = vlm_int
            s["intersection_mismatch"] = (map_int != vlm_int)
            s["vlm_scene"] = as_str(row.get("vlm_scene"))
            s["vlm_key_obstacle"] = as_str(row.get("vlm_key_obstacle"))
            s["vlm_difficulty_reason"] = as_str(row.get("vlm_difficulty_reason"))
            s["vlm_meta_action"] = as_str(row.get("vlm_meta_action"))
            samples.append(s)

    dataset.add_samples(samples)
    dataset.persistent = True
    n_mis = sum(1 for _, r in df.iterrows()
                if as_bool(r.get("map_is_intersection")) != as_bool(r.get("vlm_is_intersection")))
    print(f"[ok] {len(samples)} imgs / {len(df)} frames -> '{args.name}'  |  path-mismatch frames: {n_mis}/{len(df)}")
    session = fo.launch_app(dataset, remote=True, port=5151)
    session.wait()

if __name__ == "__main__":
    main()

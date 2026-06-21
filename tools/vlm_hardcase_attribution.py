#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vlm_hardcase_attribution.py
============================
给 UniAD 规划失效难例做「VLM 语义归因」—— uniad-planning-failure-analysis 的下一步。

定位:
  现有管线 = 几何归因(逐帧 L2 -> worst-K -> nuScenes 地图 is_intersection 客观标签)。
  本脚本 = 语义归因。对同一批 worst-K 难例,用一个开源 VLM 看相机图,
  输出「场景描述 / 是否路口 / 关键障碍 / 为什么难 / 该怎么开」。

为什么值钱(面试点):
  把 VLM 的 is_intersection 判断 和 我们已有的地图客观标签 对照,
  得到一个**可量化结果**:VLM 在难例上的路口识别一致率/grounding 准确率。
  -> 不是"接了个大模型玩玩",而是把 VLM 也纳入同一套评测框架。正好踩中 JD 的「VLM + 评测」。

依赖(独立于 UniAD,放干净环境跑,别污染 uniad2.0):
  pip install "transformers>=4.45" accelerate qwen-vl-utils pillow pandas nuscenes-devkit
  (torch 装 CUDA 版;4060 8G 用 fp16 跑 2B/3B 模型绰绰有余)

用法:
  python vlm_hardcase_attribution.py \
      --metrics-csv  ~/UniAD/output/planning_per_frame_attr.csv \
      --dataroot     ~/UniAD/data/nuscenes \
      --version      v1.0-mini \
      --topk 20 \
      --model Qwen/Qwen2-VL-2B-Instruct \
      --cameras CAM_FRONT_LEFT CAM_FRONT CAM_FRONT_RIGHT \
      --out-csv      ./outputs/vlm_attribution.csv \
      --fiftyone-json ./outputs/vlm_fields.json
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from PIL import Image


# ----------------------------------------------------------------------------- #
# 1. Prompt:让 VLM 当"驾驶场景分析员",强制结构化 JSON 输出(便于落 CSV + 对照)
# ----------------------------------------------------------------------------- #
SYSTEM_HINT = (
    "You are an autonomous-driving scene analyst. Look at the front camera "
    "view(s) of the ego vehicle and reason about why the next-3s trajectory is "
    "hard to predict here."
)

PROMPT = (
    SYSTEM_HINT
    + "\nAnswer with a SINGLE JSON object, no markdown, with EXACTLY these keys:\n"
    '{\n'
    '  "scene_description": "<one sentence, what is around the ego>",\n'
    '  "is_intersection": <true|false>,            // is the ego at/approaching an intersection?\n'
    '  "key_obstacle": "<the single most planning-relevant agent/object, or none>",\n'
    '  "difficulty_reason": "<why is the future multimodal / hard to plan here>",\n'
    '  "suggested_meta_action": "<keep_lane|decelerate|stop|turn_left|turn_right|yield>"\n'
    '}\n'
)

# 期望的 JSON 字段(解析失败时用来补空)
VLM_KEYS = [
    "scene_description",
    "is_intersection",
    "key_obstacle",
    "difficulty_reason",
    "suggested_meta_action",
]


# ----------------------------------------------------------------------------- #
# 2. 选难例:复用现有口径 —— 按 l2_3s 降序取 worst-K,剔除无效帧
# ----------------------------------------------------------------------------- #
def pick_hard_cases(metrics_csv: str, topk: int) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "token" not in df.columns or "l2_3s" not in df.columns:
        sys.exit(f"[FATAL] CSV 需要列 token / l2_3s,实际列={list(df.columns)}")
    # 剔除 scene-end masked 帧:优先用 valid_3s 掩码列(没有未来 3s 真值=False),和现有归因口径一致
    if "valid_3s" in df.columns:
        df = df[df["valid_3s"].astype(str).str.lower() == "true"]
    else:
        df = df.dropna(subset=["l2_3s"])
    df = df.sort_values("l2_3s", ascending=False).head(topk).reset_index(drop=True)
    print(f"[INFO] 选出 worst-{len(df)} 难例(按 l2_3s 降序)")
    return df


# ----------------------------------------------------------------------------- #
# 3. token -> 相机图绝对路径(nuScenes-devkit)
# ----------------------------------------------------------------------------- #
def build_token_to_images(dataroot: str, version: str, cameras):
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    def get_paths(sample_token: str):
        """返回 [(cam_name, abs_path), ...],保留相机身份以便给 VLM 打标签。"""
        sample = nusc.get("sample", sample_token)
        pairs = []
        for cam in cameras:
            sd_token = sample["data"].get(cam)
            if sd_token is None:
                continue
            p = os.path.join(dataroot, nusc.get("sample_data", sd_token)["filename"])
            pairs.append((cam, p))
        return pairs

    return get_paths


# 相机名 -> 人类可读方位(喂给 VLM 的标签,帮它建立左/中/右空间关系)
CAM_LABEL = {
    "CAM_FRONT_LEFT": "Front-left camera",
    "CAM_FRONT": "Front camera",
    "CAM_FRONT_RIGHT": "Front-right camera",
    "CAM_BACK_LEFT": "Back-left camera",
    "CAM_BACK": "Back camera",
    "CAM_BACK_RIGHT": "Back-right camera",
}


# ----------------------------------------------------------------------------- #
# 4. 加载 VLM(默认 Qwen2-VL-2B-Instruct;支持 Qwen2.5-VL 与 4bit)
# ----------------------------------------------------------------------------- #
def load_vlm(model_id: str, load_4bit: bool):
    import torch
    from transformers import AutoProcessor

    if "Qwen2.5-VL" in model_id or "Qwen2_5" in model_id:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as VLMClass

    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if load_4bit:  # 7B 想塞进 8G 才需要;2B/3B 不用
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        kwargs.pop("torch_dtype")

    model = VLMClass.from_pretrained(model_id, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[INFO] 已加载 VLM: {model_id} (4bit={load_4bit})")
    return model, processor


def run_vlm(model, processor, cam_images, max_new_tokens: int) -> str:
    """单帧推理。cam_images=[(cam_name, path), ...];每张图前插一句方位标签。"""
    from qwen_vl_utils import process_vision_info

    content = []
    for cam, p in cam_images:
        content.append({"type": "text", "text": CAM_LABEL.get(cam, cam) + ":"})
        content.append({"type": "image", "image": f"file://{p}"})
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ----------------------------------------------------------------------------- #
# 5. 鲁棒解析 JSON(模型偶尔加废话/markdown,抠出第一个 {...})
# ----------------------------------------------------------------------------- #
def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {k: None for k in VLM_KEYS}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {k: None for k in VLM_KEYS}
    return {k: obj.get(k) for k in VLM_KEYS}


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1")
    return None


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True, help="现有逐帧指标 CSV(含 sample_token,l2_3s,可选 is_intersection)")
    ap.add_argument("--dataroot", required=True, help="nuScenes 根目录(含 v1.0-mini/samples/maps)")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--cameras", nargs="+", default=["CAM_FRONT"],
                    help="喂给 VLM 的相机;路口建议 CAM_FRONT_LEFT CAM_FRONT CAM_FRONT_RIGHT")
    ap.add_argument("--load-4bit", action="store_true", help="7B 想塞 8G 才开")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-csv", default="vlm_attribution.csv")
    ap.add_argument("--fiftyone-json", default=None,
                    help="可选:导出 {sample_token: {vlm字段}},供 fiftyone_hard_cases.py 作为样本字段挂上")
    args = ap.parse_args()

    hard = pick_hard_cases(args.metrics_csv, args.topk)
    get_paths = build_token_to_images(args.dataroot, args.version, args.cameras)
    model, processor = load_vlm(args.model, args.load_4bit)

    rows, fo_fields = [], {}
    for i, r in hard.iterrows():
        tok = r["token"]
        imgs = get_paths(tok)
        if not imgs:
            print(f"[WARN] {tok} 找不到相机图,跳过")
            continue
        raw = run_vlm(model, processor, imgs, args.max_new_tokens)
        vlm = parse_json(raw)
        vlm_inter = to_bool(vlm["is_intersection"])

        row = {
            "token": tok,
            "l2_3s": r["l2_3s"],
            "map_is_intersection": r.get("is_intersection"),   # 已有客观标签(planning_per_frame_attr.csv)
            "vlm_is_intersection": vlm_inter,
            "vlm_scene": vlm["scene_description"],
            "vlm_key_obstacle": vlm["key_obstacle"],
            "vlm_difficulty_reason": vlm["difficulty_reason"],
            "vlm_meta_action": vlm["suggested_meta_action"],
        }
        rows.append(row)
        fo_fields[tok] = {k: v for k, v in row.items() if k.startswith("vlm_")}
        print(f"  [{i+1}/{len(hard)}] {tok[:8]} l2_3s={r['l2_3s']:.2f} "
              f"VLM路口={vlm_inter} reason={str(vlm['difficulty_reason'])[:50]}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[OK] 写出 {args.out_csv} ({len(out)} 行)")

    if args.fiftyone_json:
        with open(args.fiftyone_json, "w", encoding="utf-8") as f:
            json.dump(fo_fields, f, ensure_ascii=False, indent=2)
        print(f"[OK] 写出 FiftyOne 字段 {args.fiftyone_json}")

    # ---- 头条量化结果:VLM 路口判断 vs 地图客观标签的一致率 ---------------- #
    if "map_is_intersection" in out.columns and out["map_is_intersection"].notna().any():
        cmp = out.dropna(subset=["map_is_intersection", "vlm_is_intersection"]).copy()
        cmp["map_b"] = cmp["map_is_intersection"].astype(bool)
        cmp["vlm_b"] = cmp["vlm_is_intersection"].astype(bool)
        agree = (cmp["map_b"] == cmp["vlm_b"]).mean()
        recall_inter = (cmp[cmp["map_b"]]["vlm_b"].mean()
                        if cmp["map_b"].any() else float("nan"))
        print("\n========== VLM 语义归因 vs 地图客观标签 ==========")
        print(f"  样本数(有双标签)         : {len(cmp)}")
        print(f"  is_intersection 一致率     : {agree:.1%}")
        print(f"  地图判路口中 VLM 也判路口   : {recall_inter:.1%}  (VLM 路口召回)")
        print("  -> 这就是把 VLM 纳入评测框架后能写进项目/简历的可量化结论。")
    else:
        print("\n[NOTE] CSV 无 is_intersection 列;先跑 label_scene_attribution.py 打地图标签再来对照。")


if __name__ == "__main__":
    main()

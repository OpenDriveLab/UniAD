#!/usr/bin/env bash
set -uo pipefail
cd ~/UniAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uniad2.0
CFG=./projects/configs/stage2_e2e/base_e2e.py
CKPT=./ckpts/uniad_base_e2e.pth

echo "==== [1/3] baseline eval (command ON) $(date +%T) ===="
unset ABLATE_COMMAND
./tools/uniad_dist_eval.sh $CFG $CKPT 1 --planning-csv output/plan_baseline.csv
test -s output/plan_baseline.csv || { echo "FATAL: plan_baseline.csv missing"; exit 1; }

echo "==== [2/3] ablated eval (command ZEROED) $(date +%T) ===="
export ABLATE_COMMAND=1
./tools/uniad_dist_eval.sh $CFG $CKPT 1 --planning-csv output/plan_nocmd.csv
unset ABLATE_COMMAND
test -s output/plan_nocmd.csv || { echo "FATAL: plan_nocmd.csv missing"; exit 1; }

echo "==== [3/3] L2 comparison $(date +%T) ===="
python - <<'PY'
import pandas as pd, pathlib
o = pathlib.Path.home()/"UniAD/output"
b = pd.read_csv(o/"plan_baseline.csv"); n = pd.read_csv(o/"plan_nocmd.csv")
def valid(df): return df[df["valid_3s"]==True] if "valid_3s" in df.columns else df[df["l2_3s"]>0]
def m(df):
    df = valid(df)
    return {c: round(df[c].mean(),4) for c in ["l2_1s","l2_2s","l2_3s"] if c in df}
mb, mn = m(b), m(n)
print(f"{'metric':8}{'baseline':>10}{'no_command':>12}{'delta(n-b)':>12}")
for k in mb: print(f"{k:8}{mb[k]:>10}{mn[k]:>12}{round(mn[k]-mb[k],4):>12}")
mv = b.merge(n, on="token", suffixes=("_b","_n")); mv = mv[mv["l2_3s_b"]>0]
print(f"\npaired mean|delta l2_3s| ({len(mv)} frames): {round((mv['l2_3s_n']-mv['l2_3s_b']).abs().mean(),4)}")
PY
echo "==== DONE $(date +%T) ===="

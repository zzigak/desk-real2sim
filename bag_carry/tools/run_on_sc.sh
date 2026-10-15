#!/usr/bin/env bash
# Run one bag_carry script as a 1-GPU batch job on the Stanford sc cluster (account viscam).
#
#   tools/run_on_sc.sh sim/run_bag.py --task paper --load heavy --finger-armature 2.0 --eps-v 1e-3 --export exp --out bag/heavy
#
# Env knobs: PARTITION [viscam]  GPU [] (gres type)  CONSTRAINT [ada|hopper|blackwell] (libuipc's CUDA backend ships
# sm_89 code only)  CPUS [2]  MEM [16G]  TIME [4:00:00]. Relative --out goes under $OUT_ROOT. Logs: $B/logs/<jobid>-<stem>.log.
# The folder is mirrored to $PROJ at submission: a pending job runs whatever is in the mirror when it starts.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/viscam/group/ziga/sim
REPO=$B/genesis-world PROJ=$B/bag_carry OUT_ROOT=$B/out
script=${1:?usage: run_on_sc.sh <script.py relative to bag_carry/> [args...]}; shift
stem=$(basename "$script" .py)
args=() prev= out=
for a in "$@"; do
  if [[ $prev == --out ]]; then [[ $a == /* ]] || a=$OUT_ROOT/$a; out=$a; fi
  args+=("$a"); prev=$a
done
[[ -n $out ]] || { out=$OUT_ROOT/bag_carry/$stem/$(date +%Y%m%d_%H%M%S); args+=(--out "$out"); }
rsync -a --exclude out/ --exclude __pycache__/ --exclude render/tex/ --exclude 'assets/_panda_hand_box.local.xml*' "$HERE/" sc:$PROJ/
cmd=$(printf '%q ' "$REPO/.venv/bin/python" "$PROJ/$script" "${args[@]}")
job=$(cat <<EOS
#!/bin/bash
#SBATCH --job-name=bag-$stem
#SBATCH --account=viscam
#SBATCH --partition=${PARTITION:-viscam}
#SBATCH --gres=gpu:${GPU:+$GPU:}1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell}
#SBATCH --cpus-per-task=${CPUS:-2}
#SBATCH --mem=${MEM:-16G}
#SBATCH --time=${TIME:-4:00:00}
#SBATCH --output=$B/logs/%j-$stem.log
export PYOPENGL_PLATFORM=egl XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
echo "\$(hostname) \$SLURM_JOB_PARTITION \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
mkdir -p $(printf '%q' "$out") && cd $REPO && SECONDS=0
$cmd
rc=\$?; echo "exit \$rc after \${SECONDS} s wall"; exit \$rc
EOS
)
jid=$(ssh sc "sbatch --parsable" <<<"$job")
echo "job $jid"; echo "log: sc:$B/logs/$jid-$stem.log"; echo "out: sc:$out"
echo "fetch: tools/fetch.sh $out out/${out#$OUT_ROOT/}"

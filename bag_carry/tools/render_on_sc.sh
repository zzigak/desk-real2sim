#!/usr/bin/env bash
# Blender render of a bag export on sc (Cycles OptiX, one GPU per job, Blender 5.2.2 = peefton's snap build).
#
#   tools/render_on_sc.sh <export dir on sc, relative to $B/out> <tag> <res> <samples> <chunks> [render_bag.py args...]
#   e.g. tools/render_on_sc.sh bag/heavy/exp wide_heavy 1920x1080 96 4 --orbit 14
#
# Frames go to $B/out/render/<tag>/NNNNN.png; job k renders frames k, k+chunks, ... (interleaved, so every job sees
# the slow lift frames). Jobs are requeueable: frames are written under a temp name and renamed, and --skip-existing
# skips finished ones. Syncs the project (render/ and its textures) first.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/viscam/group/ziga/sim
PROJ=$B/bag_carry
BL=$B/tools/blender-5.2.2-linux-x64/blender
exp=$1 tag=$2 res=$3 samples=$4 chunks=$5; shift 5
rsync -a --exclude out/ --exclude __pycache__/ "$HERE/" sc:$PROJ/
out=$B/out/render/$tag
for ((k = 0; k < chunks; k++)); do
  job=$(cat <<EOS
#!/bin/bash
#SBATCH --job-name=bag-render-$tag
#SBATCH --account=viscam
#SBATCH --partition=${PARTITION:-viscam}
#SBATCH --gres=gpu:1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell|ampere}
#SBATCH --cpus-per-task=${CPUS:-4}
#SBATCH --mem=${MEM:-24G}
#SBATCH --time=${TIME:-3:00:00}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=$B/logs/%j-render-$tag.log
export XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
echo "\$(date) \$(hostname) \$SLURM_JOB_PARTITION restart=\${SLURM_RESTART_COUNT:-0} \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
mkdir -p $out
cd $PROJ && $BL -b -P $PROJ/render/render_bag.py -- --export $B/out/$exp --out $out --res $res --samples $samples \\
  --frames $k::$chunks --skip-existing $* 2>&1 | grep -v "^Fra:"
EOS
)
  echo "$job" | ssh sc "sbatch --parsable" | sed "s/^/job /"
done
echo "frames: sc:$out"

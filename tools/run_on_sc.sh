#!/usr/bin/env bash
# Run one desk-real2sim script as a 1-GPU batch job on the Stanford sc cluster (shared Genesis + libuipc env, see
# ~/projects/yam-arcade/tools/SC_README.md). Syncs this project and the yam-arcade robot code first.
#
#   tools/run_on_sc.sh sim/robot_demos.py marker --out out/robot/sc_marker
#   EXTRA_ENV='ROBOT_FITS_JSON={"marker":{"seat_gap":0.00018}}' tools/run_on_sc.sh sim/robot_demos.py marker ...
#   (EXTRA_ENV is split on spaces: several NAME=VALUE words, no spaces inside a value)
#   tools/run_on_sc.sh --sync-only
# Outputs land in $B/out/desk-real2sim (the mirror's out/ is a symlink there); fetch with tools/fetch.sh <path under out/>.
# Knobs: PARTITION [auto], GPU [], CONSTRAINT [ada|hopper|blackwell], CPUS [4], MEM [32G], TIME [4:00:00], SYNC [1].
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/viscam/group/ziga/sim
REPO=$B/genesis-world PROJ=$B/desk-real2sim OUT_ROOT=$B/out/desk-real2sim YA=$B/yam-arcade-main

sync_project() {
  ssh sc "mkdir -p $PROJ $OUT_ROOT $B/logs && { [ -e $PROJ/out ] || ln -s $OUT_ROOT $PROJ/out; }"
  rsync -a --exclude=__pycache__/ --exclude=dbg/ --exclude='*.log' "$HERE/sim" "$HERE/tools" "$HERE/layout.json" sc:$PROJ/ || [[ $? == 24 ]]
  rsync -a --exclude='urdf_robot/var_*' --exclude='robot_lift/' "$HERE/assets" sc:$PROJ/ || [[ $? == 24 ]]
  rsync -a --include='*/' --exclude='*' "$HERE/out/" sc:$OUT_ROOT/
  (cd ~/projects/yam-arcade && tools/run_on_sc.sh --sync-only) | tail -1
  echo "synced $HERE -> sc:$PROJ"
}

script=${1:?usage: run_on_sc.sh <script relative to the project root> [args...] | --sync-only}; shift
[[ ${SYNC:-1} == 0 ]] || sync_project
[[ $script == --sync-only ]] && exit 0
stem=$(basename "$script" .py)
cmd=$(printf '%q ' "$REPO/.venv/bin/python" "$PROJ/$script" "$@")
job=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=desk-$stem
#SBATCH --account=viscam
#SBATCH --gres=gpu:${GPU:+$GPU:}1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell}
#SBATCH --cpus-per-task=${CPUS:-4}
#SBATCH --mem=${MEM:-32G}
#SBATCH --time=${TIME:-4:00:00}
#SBATCH --output=$B/logs/%j-desk-$stem.log
export PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
export YAM_ARCADE=$YA
$(for w in ${EXTRA_ENV:-}; do printf 'export %q\n' "$w"; done)
echo "\$(hostname) \$SLURM_JOB_PARTITION \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
cd $PROJ && SECONDS=0
$cmd
rc=\$?; echo "exit \$rc after \${SECONDS} s wall"; exit \$rc
JOB
)
jid=$(ssh sc "p=${PARTITION:-auto}
if [ \$p = auto ]; then
  if [ \$(squeue -h -u \$USER -p viscam-hi | wc -l) -lt 8 ]; then p=viscam-hi
  elif [ \$(squeue -h -u \$USER -p viscam-interactive | wc -l) -lt 2 ]; then p=viscam-interactive
  else p=viscam; fi
fi
echo \$(sbatch --parsable -p \$p) \$p" <<<"$job")
[[ ${jid% *} =~ ^[0-9]+$ ]] || { echo "sbatch failed: $jid" >&2; exit 1; }
echo "job ${jid% *} on ${jid#* }  log: $B/logs/${jid% *}-desk-$stem.log"

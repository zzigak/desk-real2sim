#!/usr/bin/env bash
# Run several robot_demos.py runs side by side in ONE 1-GPU sc job (a busy queue frees one GPU long before eight).
# Each argument is "<demo> <run dir under out/> [ENV=VAL ...]"; each run logs to $B/logs/<jobid>-<run name>.log.
#
#   tools/run_pack_on_sc.sh "jar robot/t7_jar" "monitor robot/t7_mon DESK_DRIVE_monitor=0.00606 DESK_SLIP_monitor=0.0028"
# Knobs as tools/run_on_sc.sh: PARTITION [viscam,sc-loprio], CONSTRAINT [ada|hopper|blackwell], CPUS [8], MEM [64G],
# TIME [3:00:00], SYNC [1], NODELIST (pin a node).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/viscam/group/ziga/sim
REPO=$B/genesis-world PROJ=$B/desk-real2sim YA=$B/yam-arcade-main
[[ ${SYNC:-1} == 0 ]] || "$HERE/tools/run_on_sc.sh" --sync-only | tail -1
runs=""
for spec in "$@"; do
  read -r demo run envs <<<"$spec"
  name=$(basename "$run")
  runs+="env ${envs:-} $REPO/.venv/bin/python $PROJ/sim/robot_demos.py $demo --out out/$run > $B/logs/\${SLURM_JOB_ID}-$name.log 2>&1 & pids+=\" \$!\"
"
done
job=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=desk-pack
#SBATCH --account=viscam
#SBATCH --gres=gpu:1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell}
#SBATCH --cpus-per-task=${CPUS:-8}
#SBATCH --mem=${MEM:-64G}
#SBATCH --time=${TIME:-3:00:00}
#SBATCH --output=$B/logs/%j-desk-pack.log
${NODELIST:+#SBATCH --nodelist=$NODELIST}
export PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296 YAM_ARCADE=$YA
echo "\$(hostname) \$SLURM_JOB_PARTITION \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
cd $PROJ && SECONDS=0 && pids=""
$runs
( while sleep 60; do echo "\$(date +%T) gpu mem \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; done ) & mon=\$!
wait \$pids; kill \$mon
echo "all runs done after \${SECONDS} s wall"
JOB
)
jid=$(ssh sc "sbatch --parsable -p ${PARTITION:-viscam,sc-loprio}" <<<"$job")
[[ $jid =~ ^[0-9]+$ ]] || { echo "sbatch failed: $jid" >&2; exit 1; }
echo "pack job $jid  logs: $B/logs/$jid-<run>.log"

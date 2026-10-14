#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=sc-loprio
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=desk-bisect
#SBATCH --output=/viscam/group/ziga/desk-real2sim/out/slurm/%x_%j.out
P=/viscam/group/ziga/desk-real2sim; PY=/viscam/group/ziga/simquarry/genesis-world-1.3.1/.venv/bin/python
export XDG_CACHE_HOME=/viscam/group/ziga/.cache PYOPENGL_PLATFORM=egl
cd $P/sim; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
i=0
while IFS= read -r args; do
  i=$((i+1)); echo "=== [$i] $args"
  timeout 900 $PY dbg/min_cloth.py $args --steps 3 > $P/out/bisect_$i.log 2>&1; echo "exit $?"
  grep -E "^built|^step|^OK|submatrix_extent|what\(\)|xception" $P/out/bisect_$i.log | sort | uniq -c | head -5 | cut -c1-200
done <<'CFG'
--patch franka --plane 1 --table box --rg 0 --nocoll 0 --mesh towel
--patch franka --plane 1 --table box --rg 0
--plane 1 --table box --rg 0
--plane 0 --table box --rg 0

--rg 0
--plane 0
--patch none
--mesh towel
CFG

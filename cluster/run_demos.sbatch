#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=sc-loprio
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=desk-demo
#SBATCH --output=/viscam/group/ziga/desk-real2sim/out/slurm/%x_%A_%a.out
#SBATCH --array=0-10
# settle + one demo per array task, each on its own A100 (sc-loprio: preemptible, requeued)
DEMOS=(settle laptop monitor jar tin marker stick cloth paper pad mouse)
D=${DEMOS[$SLURM_ARRAY_TASK_ID]}
P=/viscam/group/ziga/desk-real2sim; PY=/viscam/group/ziga/simquarry/genesis-world-1.3.1/.venv/bin/python
export XDG_CACHE_HOME=/viscam/group/ziga/.cache PYOPENGL_PLATFORM=egl
cd $P/sim; hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
if [ "$D" = settle ]; then
    $PY settle.py --T ${SETTLE_T:-2.0} --cams p10 p09 p08 --out $P/out/settle
else
    $PY demos.py $D --out $P/out/demos ${DEMO_ARGS}
fi
echo "exit $?"

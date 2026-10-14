#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=sc-loprio
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=desk-smoke
#SBATCH --output=/viscam/group/ziga/desk-real2sim/out/slurm/%x_%j.out
P=/viscam/group/ziga/desk-real2sim; PY=/viscam/group/ziga/simquarry/genesis-world-1.3.1/.venv/bin/python
export XDG_CACHE_HOME=/viscam/group/ziga/.cache PYOPENGL_PLATFORM=egl
cd $P/sim; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
time $PY -c "import genesis, uipc, torch; print('OK genesis', genesis.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
$PY demos.py tin --res 640 480 --out $P/out/demos_sc
echo "exit $?"

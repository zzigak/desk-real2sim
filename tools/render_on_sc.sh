#!/usr/bin/env bash
# Render one robot-demo export (robot_desk.start_export -> out/<run>/export) with Blender 5.2.2 Cycles on sc.
#   tools/render_on_sc.sh robot/marker [res=1280x720] [samples=64] [frames=a:b | still=f1,f2]
#   CHUNKS=4 tools/render_on_sc.sh robot/marker 1280x720 48      # split all frames over 4 jobs (encode with --encode-only)
#   tools/render_on_sc.sh robot/marker --encode-only                # fetch the frames and encode on peefton
# Frames -> $B/out/desk-real2sim/<run>/frames, video -> <run>/<demo>_render.mp4 (fetch with tools/fetch.sh <run>).
# RARGS='--eye x,y,z --target x,y,z --lens 40' overrides the export's camera (desk export frame: z is 2.5 cm lower);
# FRAMES_DIR=stills_a writes elsewhere than frames/. PARTITION=svl works (any GPU; Cycles doesn't need Ada).
set -euo pipefail
B=/viscam/group/ziga/sim
run=${1:?usage: render_on_sc.sh <run under out/> [res] [samples] [frames|still=...]}; res=${2:-1280x720}; spp=${3:-64}; sel=${4:-}
O=$B/out/desk-real2sim/$run; name=$(basename "$run")
if [[ $res == --encode-only ]]; then   # sc has no ffmpeg: fetch the frames and encode here
  HERE=$(cd "$(dirname "$0")/.." && pwd)
  "$HERE/tools/fetch.sh" "$run/frames" >/dev/null
  ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$HERE/out/$run/frames/[0-9]*.png" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    "$HERE/out/$run/${name}_render.mp4" && ls -la "$HERE/out/$run/${name}_render.mp4"
  exit 0
fi
if [[ -n ${CHUNKS:-} ]]; then
  n=$(ssh sc "python3 -c \"import json; print(json.load(open('$O/export/meta.json'))['n_frames'])\"")
  per=$(( (n + CHUNKS - 1) / CHUNKS ))
  for ((k = 0; k < CHUNKS; k++)); do a=$((k * per)); b=$(( (k + 1) * per < n ? (k + 1) * per : n ))
    CHUNKS= ENC=0 "$0" "$run" "$res" "$spp" "$a:$b"; done
  exit 0
fi
extra=""
if [[ $sel == still=* ]]; then extra="--still ${sel#still=}"; elif [[ -n $sel ]]; then extra="--frames $sel"; fi
job=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=desk-render-$name
#SBATCH --account=viscam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS:-4}
#SBATCH --mem=${MEM:-32G}
#SBATCH --time=${TIME:-3:00:00}
#SBATCH --output=$B/logs/%j-desk-render-$name.log
export ARCADE=$B/yam-arcade-main XDG_CACHE_HOME=$B/cache
echo "\$(hostname) \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
$B/tools/blender-5.2.2-linux-x64/blender -b -P $B/desk-real2sim/sim/render_desk.py -- --camera desk --source desk=$O/export \
  --res $res --samples $spp --out $O/${FRAMES_DIR:-frames} --no-wait --skip-existing $extra ${RARGS:-}
echo "exit \$?"
JOB
)
jid=$(ssh sc "sbatch --parsable -p ${PARTITION:-viscam}" <<<"$job")
echo "render job $jid  log: $B/logs/$jid-desk-render-$name.log"

#!/usr/bin/env bash
# Download the CC0 Poly Haven textures render_bag.py uses (concrete floor, studio HDRI) into render/tex/ (~125 MB).
# Optional extras (FreePBR, not redistributable, get them from freepbr.com yourself): white-plain-paper_normal-ogl.png
# (paper relief on the bag), bare-wood1_* / bamboo-wood-semigloss-* (--wood oak|bamboo). Missing files are skipped.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)/tex
mkdir -p "$T"
PH=https://dl.polyhaven.org/file/ph-assets
for u in Textures/jpg/4k/concrete_wall_009/concrete_wall_009_diff_4k.jpg \
         Textures/exr/4k/concrete_wall_009/concrete_wall_009_nor_gl_4k.exr \
         Textures/exr/4k/concrete_wall_009/concrete_wall_009_rough_4k.exr \
         HDRIs/exr/4k/poly_haven_studio_4k.exr; do
  f=$T/$(basename "$u")
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$PH/$u"
  echo "$(du -h "$f" | cut -f1)  $f"
done

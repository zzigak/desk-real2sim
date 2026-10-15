#!/usr/bin/env bash
# Copy a run's outputs back from sc.   tools/fetch.sh <remote out dir> <local dir>
# A relative remote dir is taken under the sc output root /viscam/group/ziga/sim/out.
set -euo pipefail
src=${1:?usage: fetch.sh <remote out dir> <local dir>} dst=${2:?usage: fetch.sh <remote out dir> <local dir>}
[[ $src == /* ]] || src=/viscam/group/ziga/sim/out/$src
mkdir -p "$dst"
rsync -a --info=progress2 "sc:${src%/}/" "$dst/"

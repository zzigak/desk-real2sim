#!/usr/bin/env bash
# tools/fetch.sh <path under out/> : copy sc results into the local out/ tree
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
p=${1:?path under out/}
mkdir -p "$HERE/out/$p"
rsync -a "sc:/viscam/group/ziga/sim/out/desk-real2sim/$p/" "$HERE/out/$p/"
echo "fetched out/$p"

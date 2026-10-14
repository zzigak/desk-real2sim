#!/bin/bash
# push the project (code + assets, no outputs) to sc
set -e
cd "$(dirname "$0")/.."
rsync -az --delete --exclude out --exclude work --exclude recon --exclude sam3d --exclude photos --exclude '__pycache__' \
    ./ sc:/viscam/group/ziga/desk-real2sim/
echo synced

#!/usr/bin/env bash
#   https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full
set -euo pipefail

mkdir -p data
cd data

# id taken from DNGaussian README
pip install -q gdown
gdown 11PhkBXZZNYTD2emdG1awALlhCnkq7aN- -O nerf_llff_data.zip
unzip -q nerf_llff_data.zip
mv nerf_llff_data llff
rm nerf_llff_data.zip

echo "scenes:"; ls llff

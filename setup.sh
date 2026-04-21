#!/bin/bash
set -euo pipefail

CONDA_ENV_PATH=''
mamba env create -f WorldFM.yaml --prefix WorldFM
mamba activate WorldFM
uv pip install -r requirements.txt
git submodule update --init --recursive


# HunyuanWorld-1.0 requirements
#   real-esrgan
cd submodules/Real-ESRGAN
uv pip install basicsr-fixed facexlib gfpgan
python setup.py develop
#   zim anything
cd ../ZIM
uv pip install -e .

# MoGe version.
cd ../MoGe
git checkout 7807b5de2bc0c1e80519f5f3d1f38a606f8f9925
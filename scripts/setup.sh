#!/bin/bash

apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y
# apt-get install -y libsm6 libxext6 libxrender-dev


conda create -n sceneego python=3.9 -y
conda activate sceneego


# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/c


pip install -r requirements.txt



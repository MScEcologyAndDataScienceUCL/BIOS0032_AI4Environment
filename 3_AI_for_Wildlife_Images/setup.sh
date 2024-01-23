#!/bin/env bash

# Cloning required repos
git clone --depth 1 https://github.com/tensorflow/models
git clone https://github.com/microsoft/CameraTraps
git clone https://github.com/microsoft/ai4eutils
git clone https://github.com/ultralytics/yolov5/

# Installing the Object Detection API
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
cd ../..

# Installing the dependencies
pip install \
	torch==1.10.1+cu113 \
	torchvision==0.11.2+cu113 \
	-f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install humanfriendly jsonpickle

# Downloading the model weights
wget -O /content/md_v5a.0.0.pt https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt

cd yolov5 && git checkout c23a441c9df7ca9b1f275e8c8719c949269160d1

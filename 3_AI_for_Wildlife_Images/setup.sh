#!/bin/env bash

echo "Cloning required repositories"
if [ ! -d "models" ]; then
	echo -n "Cloning tensorflow/models repository"
	git clone --depth 1 https://github.com/tensorflow/models >>git.log 2>&1
	echo " - Done"
fi
if [ ! -d "CameraTraps" ]; then
	echo -n "Cloning CameraTraps repository"
	git clone https://github.com/microsoft/CameraTraps >>git.log 2>&1
	cd CameraTraps
	git checkout v5.0 >>git.log 2>&1
	cd ..
	echo " - Done"
fi
if [ ! -d "ai4eutils" ]; then
	echo -n "Cloning ai4eutils repository"
	git clone https://github.com/microsoft/ai4eutils >>git.log 2>&1
	echo " - Done"
fi
if [ ! -d "yolov5" ]; then
	echo -n "Cloning yolov5 repository"
	git clone https://github.com/ultralytics/yolov5/ >>git.log 2>&1
	cd yolov5
	git checkout c23a441c9df7ca9b1f275e8c8719c949269160d1 >>git.log 2>&1
	cd ..
	echo " - Done"
fi
echo " "

echo -n "Installing the Object Detection API"
cd models/research
protoc object_detection/protos/*.proto --python_out=. >>objectapi.log 2>&1
cp object_detection/packages/tf2/setup.py . >>objectapi.log 2>&1
python -m pip install . >>objectapi.log 2>&1
cd ../..
echo " - Done"

echo -n "Installing Python dependencies"
pip install \
	torch==1.10.1+cu113 \
	torchvision==0.11.2+cu113 \
	-f https://download.pytorch.org/whl/cu113/torch_stable.html >>install.log 2>&1
pip install humanfriendly jsonpickle bios0032utils==0.1.1 >>install.log 2>&1
echo " - Done"

echo -n "Downloading the model weights"
wget -O /content/md_v5a.0.0.pt https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt
echo " - Done"

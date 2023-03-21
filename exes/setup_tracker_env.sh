#!/bin/bash

git clone https://github.com/ifzhang/ByteTrack.git
source ../timelapse_env/bin/activate
#conda activate timelapse
cd ByteTrack || return
sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt
#sed -i 's/onnxruntime==1.8.0/onnxruntime==1.9.0/g' requirements.txt
pip install -r requirements.txt
python setup.py develop
pip install cython_bbox
pip install onemetric

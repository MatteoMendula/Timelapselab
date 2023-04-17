#!/bin/bash

sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y libopenmpi-dev libopenblas-base libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

cd ..
python3 -m venv timelapse_env --system-site-packages
source timelapse_env/bin/activate

pip install -U pip wheel gdown

# pytorch 1.12.0 from nvidia website
gdown https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
pip install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

# torchvision from source
sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
cd torchvision/ || exit
python setup.py install
cd ..

cd Timelapse_code || exit
sed -i 's/torch>=1.7.0,!=1.12.0/# torch>=1.7.0,!=1.12.0/g' requirements.txt
sed -i 's/torchvision>=0.8.1,!=0.13.0/# torchvision>=0.8.1,!=0.13.0/g' requirements.txt
sed -i 's/ultralytics>=8.0.55/# ultralytics>=8.0.55/g' requirements.txt
sed -i 's/onnxsim>=0.3.6/# onnxsim>=0.3.6/g' requirements.txt
sed -i 's/# tensorflow>=2.4.1/tensorflow>=2.4.1/g' requirements.txt
sed -i 's/# pycuda/pycuda/g' requirements.txt
pip install -r requirements.txt
sed -i 's/# torch>=1.7.0,!=1.12.0/torch>=1.7.0,!=1.12.0/g' requirements.txt
sed -i 's/# torchvision>=0.8.1,!=0.13.0/torchvision>=0.8.1,!=0.13.0/g' requirements.txt
sed -i 's/# ultralytics>=8.0.55/ultralytics>=8.0.55/g' requirements.txt
sed -i 's/# onnxsim>=0.3.6/onnxsim>=0.3.6/g' requirements.txt
sed -i 's/tensorflow>=2.4.1/# tensorflow>=2.4.1/g' requirements.txt
sed -i 's/pycuda/# pycuda/g' requirements.txt
cd ..

# ultralytics
git clone https://github.com/ultralytics/ultralytics
cd ultralytics || exit
sed -i 's/torch>=1.7.0/# torch>=1.7.0/g' requirements.txt
sed -i 's/torchvision>=0.8.1/# torchvision>=0.8.1/g' requirements.txt
pip install .
sed -i 's/# torch>=1.7.0/torch>=1.7.0/g' requirements.txt
sed -i 's/# torchvision>=0.8.1/torchvision>=0.8.1/g' requirements.txt
cd ..
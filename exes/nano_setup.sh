#!/bin/bash

sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

cd ..
python3.8 -m venv timelapse_env
source timelapse_env/bin/activate

git clone https://github.com/ultralytics/ultralytics
cd ultralytics

pip install -U pip wheel gdown

# pytorch 1.11.0
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM -O torch-1.11.010+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV -O torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
python3.8 -m pip install torch-1.11.010+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
python3.8 -m pip install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl

python3.8 -m pip install .

cd ..
cd Timelapselab
sed -i 's/# tensorflow>=2.4.1/tensorflow>=2.4.1/g' requirements.txt
sed -i 's/# tensorflowjs>=3.9.0/tensorflowjs>=3.9.0/g' requirements.txt
sed -i 's/# openvino-dev/openvino-dev/g' requirements.txt
python3.8 -m pip install -r requirements.txt
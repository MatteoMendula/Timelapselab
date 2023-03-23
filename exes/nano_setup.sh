#!/bin/bash

sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

cd ..
git clone https://github.com/ultralytics/ultralytics
cd ultralytics

python3.8 -m venv timelapse_env
source timelapse_env/bin/activate

pip install -U pip wheel gdown

# pytorch 1.11.0
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
python3.8 -m pip install torch-*.whl torchvision-*.whl

python3.8 -m pip install .

cd ..
cd Timelapselab
sed -i 's/# tensorflow>=2.4.1/tensorflow>=2.4.1/g' requirements.txt
sed -i 's/# tensorflowjs>=3.9.0/tensorflowjs>=3.9.0/g' requirements.txt
sed -i 's/# openvino-dev/openvino-dev/g' requirements.txt
python3.8 -m pip install -r requirements.txt
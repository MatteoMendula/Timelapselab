#!/bin/bash

version="8.5.2.2"
arch="x86_64"
cuda="cuda-11.8.cudnn8.6"

cd  || exit
mkdir tensorrt
cd tensorrt || exit

gdown https://drive.google.com/uc?id=1hnSEFMrde8WiQI7_uD8-IWd6G6hB1-aB -O TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz

tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz
ls TensorRT-${version}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/tensorrt/TensorRT-${version}/lib

cd TensorRT-${version}/python || exit
python3 -m pip install tensorrt-${version}-cp38-none-linux_x86_64.whl
python3 -m pip install tensorrt_lean-${version}-cp38-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-${version}-cp38-none-linux_x86_64.whl
cd ~/tensorrt || exit

cd TensorRT-${version}/uff || exit
python3 -m pip install uff-0.6.9-py2.py3-none-any.whl
which convert-to-uff
cd ~/tensorrt || exit

cd TensorRT-${version}/graphsurgeon || exit
python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
cd ~/tensorrt || exit

cd TensorRT-${version}/onnx_graphsurgeon || exit
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
cd ~/tensorrt || exit

cd ~ || exit
tree -d
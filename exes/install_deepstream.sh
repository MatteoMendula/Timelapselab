#!/bin/bash

sudo apt install \
libssl1.1 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev

git clone https://github.com/edenhill/librdkafka.git
cd librdkafka || exit
git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a
./configure
make
sudo make install

sudo mkdir -p /opt/nvidia/deepstream/deepstream-6.2/lib
sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-6.2/lib

cd ..
gdown https://drive.google.com/uc?id=1ZxGQGZtaMExKioWpYxrbl10TUD63P2_a -O deepstream-6.2_6.2.0-1_arm64.deb
sudo apt-get install ./deepstream-6.2_6.2.0-1_arm64.deb

#gdown https://developer.nvidia.com/downloads/deepstream-sdk-v620-jetson-tbz2 -O deepstream_sdk_v6.2.0_jetson.tbz2
#sudo tar -xvf deepstream_sdk_v6.2.0_jetson.tbz2 -C /
#cd /opt/nvidia/deepstream/deepstream-6.2 || exit
#sudo ./install.sh
#sudo ldconfig

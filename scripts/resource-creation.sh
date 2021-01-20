#!/bin/bash

sudo apt-get update

sudo apt-get install -y build-essential

# install cuda for tf2.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}

# install cudnn
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn8_8.0.2.39-1+cuda11.0_amd64.deb
sudo dpkg -i libcudnn8_8.0.2.39-1+cuda11.0_amd64.deb

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn8-dev_8.0.2.39-1+cuda11.0_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.2.39-1+cuda11.0_amd64.deb


# install needed packages
sudo apt-get install -y cmake \
    git \
    python3-setuptools \
    python3-dev \
    python3-pip \
    libopencv-dev \
    htop \
    tmux \
    tree \
    p7zip-full

pip3 install -U pip
pip3 install --upgrade setuptools
pip3 uninstall crcmod -y
pip3 install --no-cache-dir crcmod
pip3 install --upgrade pyasn1
cd necstlab-damage-segmentation && pip3 install -r requirements.txt


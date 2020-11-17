#!/bin/bash

sudo apt-get update

sudo apt-get install -y build-essential

# install cuda for tf2.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1604-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}

# install cudnn
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb

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

pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 uninstall crcmod -y
pip3 install --no-cache-dir crcmod
pip3 install --upgrade pyasn1
cd necstlab-damage-segmentation && pip3 install -r requirements.txt



#!/bin/bash

sudo apt-get update

sudo apt-get install -y build-essential

# install cuda
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo apt-get update
sudo apt-get install -y --allow-unauthenticated cuda
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}

# install cudnn
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.5.1.10-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7_7.5.1.10-1+cuda10.0_amd64.deb

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.5.1.10-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.5.1.10-1+cuda10.0_amd64.deb

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

sudo pip3 uninstall crcmod
sudo pip3 install pipenv
sudo pip3 install --no-cache-dir -U crcmod

cd necstlab-damage-segmentation && pipenv install
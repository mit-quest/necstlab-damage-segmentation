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


# Install requirements
sudo apt-get install -y \
    checkinstall\
    libreadline-gplv2-dev\
    liblzma-dev\
    libncursesw5-dev\
    libssl-dev\
    libsqlite3-dev\
    tk-dev\
    libgdbm-dev\
    libc6-dev\
    libbz2-dev\
    zlib1g-dev\
    openssl\
    libffi-dev\
    python3-dev\
    python3-setuptools\
    wget\
    zlib1g-dev

# install needed packages
sudo apt-get install -y cmake \
	git \
	libopencv-dev \
	htop \
	tmux \
	tree \
	p7zip-full

cd ~
mkdir tmp
cd tmp
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar zxvf Python-3.7.9.tgz
cd Python-3.7.9
./configure --prefix=$HOME/opt/python-3.7.9
make
make install
cd ~
echo 'export PATH=$HOME/opt/python-3.7.9/bin:$PATH' >> .bash_profile
. ~/.bash_profile
cd ~

pip3 install -U pip
pip3 install --upgrade setuptools
pip3 install --no-cache-dir crcmod
pip3 install --upgrade pyasn1
cd necstlab-damage-segmentation && pip3 install -r requirements.txt



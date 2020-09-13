#!/bin/bash

# install script for PyTorch 1.6 + CUDA 10.2 combination on Ubuntu 18.04 (also works for 20.04)
# see https://pytorch.org/get-started/locally/

# see https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda-repo-ubuntu1804-10-2-local_10.2.1-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local_10.2.1-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

python -c "import torch; print('CUDA available: {} ({})'.format(torch.cuda.is_available(), torch.cuda.get_device_name(torch.cuda.current_device())))"

# clean up
rm cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
rm cuda-repo-ubuntu1804-10-2-local_10.2.1-1_amd64.deb

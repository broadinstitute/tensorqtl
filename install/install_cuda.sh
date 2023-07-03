#!/bin/bash
# install script for PyTorch 2 + CUDA 11.8 on Ubuntu 22.04
# for torch, see https://pytorch.org/get-started/locally/
# for CUDA drivers, see https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
# for other versions, see https://developer.nvidia.com/cuda-toolkit-archive

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
rm cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb

# test
python -c "import torch; print('CUDA available: {} ({})'.format(torch.cuda.is_available(), torch.cuda.get_device_name(torch.cuda.current_device())))"

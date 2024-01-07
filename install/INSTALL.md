### Setup CUDA drivers and PyTorch on GCP

Launch a new instance configured with Ubuntu 22.04 LTS and a GPU, clone this repository, and run the following:
#### Install CUDA
```bash
sudo ./install_cuda.sh
sudo reboot
# verify
nvidia-smi
```

#### Install R
Required for computing q-values. Follow instructions [here](https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-22-04), then install the 'qvalue' package with
```bash
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("qvalue")
```

#### Install Python 3
Using a [conda](https://github.com/conda-forge/miniforge) environment is recommended. The `tensorqtl_env.yml` configuration contains all required packages, including `torch` and `tensorqtl`.
```bash
mamba env create -f tensorqtl_env.yml
conda activate tensorqtl

# verify
python -c "import torch; print(torch.__version__); print('CUDA available: {} ({})'.format(torch.cuda.is_available(), torch.cuda.get_device_name(torch.cuda.current_device())))"

# this should print something like
# 2.1.2+cu121
# CUDA available: True (Tesla P100-PCIE-16GB)
```

#### Install rmate (optional)
```bash
sudo apt install -y ruby
mkdir ~/bin
curl -Lo ~/bin/rmate https://raw.githubusercontent.com/textmate/rmate/master/bin/rmate
chmod a+x ~/bin/rmate
echo 'export RMATE_PORT=${rmate_port}' >> ~/.bashrc
```

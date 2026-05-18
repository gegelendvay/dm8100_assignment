# HPC assignment

## Overview
Matrix-matrix multiplications are a common problem that appears in many codes aimed at
HPC systems. This project aims to implement a matrix-matrix multiplication using different
parallelisation paradigms. We will look at shared-memory systems with OpenMP. Distributed
memory system with MPI. And accelerator (GPU) systems with CUDA.

## Setup
1. Clone the repository by running: `git clone git@github.com:gegelendvay/dm8100_assignment.git`
2. Install the required packages
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt -y update
sudo apt -y install cuda-toolkit
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
sudo apt -y install libopenblas-dev
source ~/.bashrc
```

> [!IMPORTANT]  
> **Operating System:** Ubuntu 24.04  
> **Platform:** UCloud (https://docs.cloud.sdu.dk/Apps/terminal.html)
> **GPU:** NVIDIA B200  

## Usage
Compile and execute the files using the commands specified in each file's header.

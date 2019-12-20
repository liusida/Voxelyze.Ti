# Voxelyze.Ti

Attempt to GPUlize Voxelyze. 96.16% of runtime spend on the function `CVX_Link::updateForces`, so if we can move this part to GPU...

## Build and Run

```bash
mkdir build
cd build
cmake ..
make -j8
./Voxelyze.Ti
```

## Dev Environment

Ubuntu 19.10

CUDA 10.1 from https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux

add cuda $PATH to `~/.bashrc`:

```bash
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
```

```bash
sudo apt install gcc-8 g++-8
sudo ln -sf /usr/bin/gcc-8 /usr/bin/gcc
sudo ln -sf /usr/bin/g++-8 /usr/bin/g++
```

## Framework of Voxelyze

The main stage is CVoxelyze. Two main concepts are Link and Voxel.

![Link Voxel](doc/framework.png)

## How to do profiling

```bash
sudo apt install linux-tools-common linux-tools-`uname -r`
sudo perf record -g ./Voxelyze.Ti
perf report
```

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
CUDA 10.1

## Framework of Voxelyze

The main stage is CVoxelyze. Two main concepts are Link and Voxel.

![Link Voxel](doc/framework.png)
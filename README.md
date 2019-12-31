# Voxelyze.Ti

Attempt to GPUlize Voxelyze. 96.16% of runtime spend on the function `CVX_Link::updateForces`, so if we can move this part to GPU...

## Build and Run

```bash
mkdir build
cd build
cmake ..
cmake --build .
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

```bash
sudo nvprof ./Voxelyze.Ti
sudo nvprof --print-gpu-trace ./Voxelyze.Ti
```

## time

GPU: 30k voxels, 20k steps, 46s.

## Enabling cuda kernel debugging

![https://github.com/liusida/NewProjectTemplate/blob/master/doc/cuda_gdb_debugging_step_into_kernels.png?raw=true](https://github.com/liusida/NewProjectTemplate/blob/master/doc/cuda_gdb_debugging_step_into_kernels.png)

1. Make sure there's `/usr/local/cuda/bin/cuda-gdb`;

2. In Visual Studio Code, create `launch.json` for debug tool and add these key lines:
```
"MIMode": "gdb",
"miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb",
```

3. In `CMakeLists.txt` of CUDA-related sub-projects, add these flags for `nvcc`:
```
#enable cuda debug
target_compile_options(cudaProj1_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
```

4. Visual Code extension `vscode-cpptools` from `Microsoft` should be installed. (Just search `c++` in vscode extension marketplace and install it.)

5. CUDA files can be colored using extension `vscode-cuda`.

6. Maybe there're several libraries need to be installed. Such as `libncurses`. (If you see lacking of `libxxxx.so`)

7. Adding `set(CMAKE_VERBOSE_MAKEFILE ON)` on the bottom of main `CMakeLists.txt` could show every step in building. (Easy for error checking.)

8. !! The most tricky thing is: we need to set another breakpoint on the line which calls the kernel.
```
break on @    Kernel<<<1,3>>>();
```
and when program hit this line, press F11 to step in, then F10 to step over, then the breakpoints inside the kernel will work!

## Enabling IntelliSense via Clangd

1. Install extension `vscode-clangd` in Visual Studio Code and install `clangd` in Ubuntu:
```
sudo apt install clangd
```

2. Disable intellisense via `vscode-cpp` by adding those lines in `settings.json`:
```
    "C_Cpp.formatting": "Disabled",
    "C_Cpp.autocomplete": "Disabled",
    "C_Cpp.errorSquiggles": "Disabled",
    "C_Cpp.intelliSenseEngine": "Disabled",
    "C_Cpp.configurationWarnings": "Disabled",
    "C_Cpp.autoAddFileAssociations": false,
    "C_Cpp.vcpkg.enabled": false,
```

3. Ctrl Click something, intellisense should work!



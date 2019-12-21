#include <cuda_runtime.h>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
        //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort) {
            char buffer[200];
            snprintf(buffer, sizeof(buffer), "GPUassert error in CUDA kernel: %s %s %d\n", cudaGetErrorString(code), file, line);
            std::string buffer_string = buffer;
            throw std::runtime_error(buffer_string);
            exit(code);
        }
    }
}
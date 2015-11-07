#include <iostream>
#include <cuda/array_abstraction.hpp>
#include <common/cu_error.hpp>
#include <cuda/noise.hpp>

typedef float Float;

typedef Array::CPUArray<Float> CPUArray;
typedef Array::GPUArray<Float> GPUArray;

int main(int argc, char* argv[]){
    if(argc != 5){
        std::cerr << "usage: " << argv[0] << " <n0> <n1> <n2> <outfile.npy> # n0 >> n1 >> n2" << std::endl;
        exit(EXIT_FAILURE);
    }
    CUERR(cudaSetDevice(0));

    int3 shape = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    std::string outfile(argv[4]);

    CPUArray host_array(shape);
    GPUArray dev_array(host_array);

    NoiseHostApi<Float> noise;

    noise.fill_kspace(dev_array);
    dev_array >> host_array;
    host_array.save(outfile);

    return 0;
}
// nvcc -ccbin=g++-4.8 -std=c++11 -I/home/mcstar/src/mcstar-libs -arch=sm_35 -lcurand -o main main.cu

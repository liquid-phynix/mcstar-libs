#include <iostream>
#include <cuda/array_abstraction.hpp>
#include <common/cu_error.hpp>
#include <cuda/noise.hpp>
#include <ctime>

typedef float Float;

typedef Array::CPUArray<Float> CPUArray;
typedef Array::GPUArray<Float> GPUArray;
using Array::AsCmpl;

int main(int argc, char* argv[]){
    if(argc != 5){
        std::cerr << "usage: " << argv[0] << " <n0> <n1> <n2> <outfile.npy> # n0 >> n1 >> n2" << std::endl;
        exit(EXIT_FAILURE);
    }
    CUERR(cudaSetDevice(0));

    int3 shape = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    std::string outfile(argv[4]);

    CPUArray host_array(shape);
    host_array.set_to(0);
    GPUArray dev_array(host_array);

    NoiseHostApi<Float> noise(clock() % 1234);

    noise.fill_kspace(dev_array);
    dev_array >> host_array;
    host_array.save<AsCmpl>(outfile);

    return 0;
}

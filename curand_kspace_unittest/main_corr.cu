#include <iostream>
#include <cuda/array_abstraction.hpp>
#include <cuda/cufft_wrapper.hpp>
#include <common/cu_error.hpp>
#include <cuda/noise.hpp>
#include <ctime>

typedef float Float;

typedef Array::CPUArray<Float> CPUArray;
typedef Array::GPUArray<Float> GPUArray;
using Array::AsCmpl;
typedef FFT::CufftPlan<Float, FFT::C2R> PlanC2R;

int main(int argc, char* argv[]){
    if(argc != 4){
        std::cerr << "usage: " << argv[0] << " <n0> <n1> <n2> # n0 >> n1 >> n2" << std::endl;
        exit(EXIT_FAILURE);
    }
    CUERR(cudaSetDevice(0));

    int3 shape = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};

    CPUArray host1(shape);
    CPUArray host2(shape);
    CPUArray host3(shape);

    GPUArray dev1(host1);

    NoiseHostApi<Float> noise(clock() % 1234);
    PlanC2R c2r(shape);

    int elems = shape.x * shape.y * shape.z;

    int iters = 1000000;
    for(int i = 0; i < iters; i++){
        noise.fill_kspace(dev1);
        c2r.execute(dev1);
        dev1 >> host1;
        host1.over([elems, &host2, &host3](Float& x, int idx){ Float v = x / elems; host2[idx] += v; host3[idx] += v * v; });
        if(i % 1000 == 0){
            std::cout << "it=" << i << std::endl;
        }
    }
    host2.over([iters](Float& x){ x /= iters; });
    host3.over([iters](Float& x){ x /= iters; });
    host2.save("avg_x.npy");
    host3.save("avg_x2.npy");

    return 0;
}
// nvcc -ccbin=g++-4.8 -std=c++11 -I/home/mcstar/src/mcstar-libs -arch=sm_35 -lcurand -lcufft -o main_corr main_corr.cu

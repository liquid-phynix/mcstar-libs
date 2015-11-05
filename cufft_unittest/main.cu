#include <iostream>
#include <array_abstraction.hpp>
#include <cufft_wrapper.hpp>
#include <common/cu_error.hpp>

// FLOAT is defined during compilation
typedef FLOAT Float;
// as well as CYCLE=1 meaning full fft cycle wo/ normalization
//            CYCLE=0 meaning saving the result of the FFT
// and INPLACE=0,1 as you would guess

typedef Array::CPUArray<Float> CPUArray;
typedef Array::GPUArray<Float> GPUArray;

typedef FFT::CufftPlan<Float, FFT::R2C> PlanR2C;
typedef FFT::CufftPlan<Float, FFT::C2R> PlanC2R;

bool factors235(int n, int i = 2){
    if(i * i > n) return n <= 5;
    else if(n % i == 0) return factors235(i) and factors235(n / i);
    else return factors235(n, i+1);
}

int main(int argc, char* argv[]){
    if(argc != 6){
        std::cerr << "usage: " << argv[0] << " <n0> <n1> <n2> <infile.npy> <outfile.npy> # n0 >> n1 >> n2" << std::endl;
        exit(EXIT_FAILURE);
    }
    CUERR(cudaSetDevice(0));

    int3 shape = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    // I restrict the extent of the dimensions, because I wouldnt use other factorization anyway
    if(not (factors235(shape.x) and factors235(shape.y) and factors235(shape.z))){
        std::cerr << "length along a dimension is not factorized by 2,3,5" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string infile(argv[4]);
    std::string outfile(argv[5]);

    CPUArray host_array(shape);
    GPUArray dev_array_1(host_array);
    GPUArray dev_array_2(host_array);
    // FFT objects for GPU arrays
    PlanR2C r2c(host_array.real_vext());
    PlanC2R c2r(host_array.real_vext());

    host_array.load(infile);
    /*host_array.save(outfile);*/
    /*return 0;*/
    host_array >> dev_array_1;

#if INPLACE==0
    r2c.execute(dev_array_1, dev_array_2);
    dev_array_2 >> host_array;
#elif INPLACE==1
    r2c.execute(dev_array_1);
    dev_array_1 >> host_array;
#endif

#if CYCLE==0
    host_array.save<Array::AsCmpl>(outfile);
#elif CYCLE==1
#if INPLACE==0
    c2r.execute(dev_array_2, dev_array_1);
    dev_array_1 >> host_array;
#elif INPLACE==1
    c2r.execute(dev_array_1);
    dev_array_1 >> host_array;
#endif
    host_array.save(outfile);
#endif

    return 0;
}

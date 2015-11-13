#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <cuda/curand_error.hpp>
#include <cuda/kernels/common.hpp>

template <typename F>
__global__
void kernel_correct_random_spectrum_read_2d(typename Array::TP<F>::CT* arr,
                                            typename Array::TP<F>::CT* dev_1,
                                            typename Array::TP<F>::CT* dev_2,
                                            int3 cdims, bool real_d0_even){
    int i0s[2] = {0, cdims.x - 1};
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    typename Array::TP<F>::CT* dev_ptr[2] = {dev_1, dev_2};
    if(i1 >= cdims.y) return;
    int idx_2d_wo_i0 = i1 * cdims.x;// + i0;
    for(int i = 0; i <= real_d0_even; i++)
        dev_ptr[i][i1] = arr[idx_2d_wo_i0 + i0s[i]];
}

template <typename F>
__global__
void kernel_correct_random_spectrum_write_2d(typename Array::TP<F>::CT* arr,
                                             typename Array::TP<F>::CT* dev_1,
                                             typename Array::TP<F>::CT* dev_2,
                                             int3 cdims, bool real_d0_even){
    int i0s[2] = {0, cdims.x - 1};
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    typename Array::TP<F>::CT* dev_ptr[2] = {dev_1, dev_2};
    if(i1 >= cdims.y) return;
    int idx_2d_wo_i0 = i1 * cdims.x;// + i0;
    int i1_m = posrem(-i1, cdims.y);

    for(int i = 0; i <= real_d0_even; i++){
        typename Array::TP<F>::CT c1 = dev_ptr[i][i1];
        if(i1 == i1_m){
            // these are the real elements
            arr[idx_2d_wo_i0 + i0s[i]] = {sqrtf(2.f) * c1.x, 0};
        } else {
            typename Array::TP<F>::CT c2 = dev_ptr[i][i1_m];
            arr[idx_2d_wo_i0 + i0s[i]] = {(c1.x + c2.x) / sqrtf(2.f), (c1.y - c2.y) / sqrtf(2.f)};
        }
    }
}

template <typename F>
__global__
void kernel_correct_random_spectrum_read_3d(typename Array::TP<F>::CT* arr,
                                            typename Array::TP<F>::CT* dev_1,
                                            typename Array::TP<F>::CT* dev_2,
                                            int3 cdims, bool real_d0_even){
    int i0s[2] = {0, cdims.x - 1};
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.z * blockDim.z + threadIdx.z;
    typename Array::TP<F>::CT* dev_ptr[2] = {dev_1, dev_2};
    if(i1 >= cdims.y or i2 >= cdims.z) return;
    int idx_3d_wo_i0 = (i2 * cdims.y + i1) * cdims.x;// + i0;
    int idx_2d = i2 * cdims.y + i1;
    for(int i = 0; i <= real_d0_even; i++)
        dev_ptr[i][idx_2d] = arr[idx_3d_wo_i0 + i0s[i]];
}

template <typename F>
__global__
void kernel_correct_random_spectrum_write_3d(typename Array::TP<F>::CT* arr,
                                             typename Array::TP<F>::CT* dev_1,
                                             typename Array::TP<F>::CT* dev_2,
                                             int3 cdims, bool real_d0_even){
    int i0s[2] = {0, cdims.x - 1};
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.z * blockDim.z + threadIdx.z;
    typename Array::TP<F>::CT* dev_ptr[2] = {dev_1, dev_2};
    if(i1 >= cdims.y or i2 >= cdims.z) return;
    int idx_3d_wo_i0 = (i2 * cdims.y + i1) * cdims.x;// + i0;
    int i1_m = posrem(-i1, cdims.y);
    int i2_m = posrem(-i2, cdims.z);
    int idx_2d = i2 * cdims.y + i1;
    int idx_2d_mirror = i2_m * cdims.y + i1_m;

    for(int i = 0; i <= real_d0_even; i++){
        auto c1 = dev_ptr[i][idx_2d];
        if(i1 == i1_m and i2 == i2_m){
            // these are the real elements
            arr[idx_3d_wo_i0 + i0s[i]] = {sqrtf(2.f) * c1.x, 0};
        } else {
            auto c2 = dev_ptr[i][idx_2d_mirror];
            arr[idx_3d_wo_i0 + i0s[i]] = {(c1.x + c2.x) / sqrtf(2.f), (c1.y - c2.y) / sqrtf(2.f)};
        }
    }
}

template <typename F>
class NoiseHostApi {
    private:
        curandGenerator_t gen;
        // pseudo random number generator types
        const curandRngType T1 = CURAND_RNG_PSEUDO_XORWOW;
        const curandRngType T2 = CURAND_RNG_PSEUDO_MRG32K3A;
        const curandRngType T3 = CURAND_RNG_PSEUDO_MTGP32;
        void _fill(F*, int, F);
    public:
        NoiseHostApi(unsigned long long int seed = 1234L){
            CURAND_CALL(curandCreateGenerator(&gen, T1));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed)); }
        ~NoiseHostApi(){
            CURAND_CALL(curandDestroyGenerator(gen)); }
        void fill(Array::GPUArray<F>& arr, F sigma){
            _fill(arr.ptr_real(), arr.real_elems(), sigma); }
        void fill_kspace(Array::GPUArray<F>& arr){
            const int3 rshape = arr.real_vext();
            const int3 cshape = arr.cmpl_vext();
            int alloc_elements = 0;
            dim3 bs, gs;
            if(rshape.z == 1){ // 2d array
                assert(rshape.x >= 1 and rshape.y >= 1 and "not a real 2d array");
                alloc_elements = cshape.y;
                bs = {1, 256};
                gs = {1, div_up(cshape.y, bs.y)};
            } else {           // 3d array
                assert(rshape.x >= 1 and rshape.y >= 1 and rshape.z >= 1 and "not a real 3d array");
                alloc_elements = cshape.y * cshape.z;
                bs = {1, 16, 32};
                gs = {1, div_up(cshape.y, bs.y), div_up(cshape.z, bs.z)};
            }
            const int alloc_size = alloc_elements * 2 * sizeof(F);
            typename Array::TP<F>::CT* dev_ptr_1 = NULL;
            typename Array::TP<F>::CT* dev_ptr_2 = NULL;
            CUERR(cudaMalloc((void**)&dev_ptr_1, alloc_size));
            CUERR(cudaMalloc((void**)&dev_ptr_2, alloc_size));
            _fill(arr.ptr_real(), 2 * arr.cmpl_elems(), 1./sqrt(2));
            if(rshape.z == 1){ // 2d array
                kernel_correct_random_spectrum_read_2d<F><<<gs, bs>>>(arr.ptr_cmpl(), dev_ptr_1, dev_ptr_2, arr.cmpl_vext(), rshape.x % 2 == 0);
                kernel_correct_random_spectrum_write_2d<F><<<gs, bs>>>(arr.ptr_cmpl(), dev_ptr_1, dev_ptr_2, arr.cmpl_vext(), rshape.x % 2 == 0);
            } else {           // 3d array
                kernel_correct_random_spectrum_read_3d<F><<<gs, bs>>>(arr.ptr_cmpl(), dev_ptr_1, dev_ptr_2, arr.cmpl_vext(), rshape.x % 2 == 0);
                kernel_correct_random_spectrum_write_3d<F><<<gs, bs>>>(arr.ptr_cmpl(), dev_ptr_1, dev_ptr_2, arr.cmpl_vext(), rshape.x % 2 == 0);
            }
            CUERR(cudaFree(dev_ptr_1));
            CUERR(cudaFree(dev_ptr_2));
            CUERR(cudaThreadSynchronize());
        }
};

template <> void NoiseHostApi<float>::_fill(float* ptr, int len, float sigma){
    CURAND_CALL(curandGenerateNormal(gen, ptr, len, 0, sigma));
    CUERR(cudaThreadSynchronize()); }
template <> void NoiseHostApi<double>::_fill(double* ptr, int len, double sigma){
    CURAND_CALL(curandGenerateNormalDouble(gen, ptr, len, 0, sigma));
    CUERR(cudaThreadSynchronize()); }

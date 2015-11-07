#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <curand_error.hpp>

template <typename F>
class NoiseHostApi {
    private:
        curandGenerator_t gen;
        // pseudo random number generator types
        const curandRngType T1 = CURAND_RNG_PSEUDO_XORWOW;
        const curandRngType T2 = CURAND_RNG_PSEUDO_MRG32K3A;
        const curandRngType T3 = CURAND_RNG_PSEUDO_MTGP32;
    public:
        NoiseHostApi(unsigned long long int seed = 1234L){
            CURAND_CALL(curandCreateGenerator(&gen, T1));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed)); }
        ~NoiseHostApi(){
            CURAND_CALL(curandDestroyGenerator(gen)); }
        void fill(Array::GPUArray<F>&, F);
        void fill_kspace(Array::GPUArray<F>&);
};
template <> void NoiseHostApi<float>::fill(Array::GPUArray<float>& arr, float sigma){
    CURAND_CALL(curandGenerateNormal(gen, arr.ptr_real(), arr.real_elems(), 0, sigma));
    CUERR(cudaThreadSynchronize()); }
template <> void NoiseHostApi<double>::fill(Array::GPUArray<double>& arr, double sigma){
    CURAND_CALL(curandGenerateNormalDouble(gen, arr.ptr_real(), arr.real_elems(), 0, sigma));
    CUERR(cudaThreadSynchronize()); }

__global__ void kernel_correct_random_spectrum_read(float2* arr, float2* dev_1, float2* dev_2, int3 cdims, bool real_d0_even){
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.z * blockDim.z + threadIdx.z;
    int i0s[2] = {0, cdims.x - 1};
    float2* dev_ptr[2] = {dev_1, dev_2};
    if(i1 >= cdims.y or i2 >= cdims.z) return;
    int idx_3d_wo_i0 = (i2 * cdims.y + i1) * cdims.x;// + i0;
    int idx_2d = i2 * cdims.y + i1;

    for(int i = 0; i <= real_d0_even; i++)
        dev_ptr[i][idx_2d] = arr[idx_3d_wo_i0 + i0s[i]];
}

__global__ void kernel_correct_random_spectrum_write(float2* arr, float2* dev_1, float2* dev_2, int3 cdims, bool real_d0_even){
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i2 = blockIdx.z * blockDim.z + threadIdx.z;
    int i0s[2] = {0, cdims.x - 1};
    float2* dev_ptr[2] = {dev_1, dev_2};
    if(i1 >= cdims.y or i2 >= cdims.z) return;
    int idx_3d_wo_i0 = (i2 * cdims.y + i1) * cdims.x;// + i0;
    int i1_m = (-i1) % cdims.y;
    int i2_m = (-i2) % cdims.z;
    int idx_2d = i2 * cdims.y + i1;
    int idx_2d_mirror = i2_m * cdims.y + i1_m;

    for(int i = 0; i <= real_d0_even; i++){
        float2 c1 = dev_ptr[i][idx_2d];
        if(i1 == i1_m and i2 == i2_m){
            // these are the real elements
            arr[idx_3d_wo_i0 + i0s[i]] = {sqrtf(2.f) * c1.x, 0};
        } else {
            float2 c2 = dev_ptr[i][idx_2d_mirror];
            arr[idx_3d_wo_i0 + i0s[i]] = {(c1.x + c2.x) / sqrtf(2.f), (c1.y - c2.y) / sqrtf(2.f)};
        }
    }
}

template <> void NoiseHostApi<float>::fill_kspace(Array::GPUArray<float>& arr){
    int3 rshape = arr.real_vext();
    assert(rshape.x > 1 and rshape.y > 1 and rshape.z > 1 and "not a 3d array");

    const int3 shape = arr.cmpl_vext();
    const int alloc_elements = shape.y * shape.z;
    const int alloc_size = alloc_elements * 2 * sizeof(float);
    float2* dev_ptr_1 = NULL;
    float2* dev_ptr_2 = NULL;
    CUERR(cudaMalloc((void**)&dev_ptr_1, alloc_size));
    CUERR(cudaMalloc((void**)&dev_ptr_2, alloc_size));
    CURAND_CALL(curandGenerateNormal(gen, arr.ptr_real(), 2 * arr.cmpl_elems(), 0, 1./sqrt(2)));

    const dim3 bs = {1, 16, 32};
    const dim3 gs = {1, div_up(shape.y, bs.y), div_up(shape.z, bs.z)};
    kernel_correct_random_spectrum_read<<<gs, bs>>>(arr.ptr_cmpl(), dev_ptr_1, dev_ptr_2, arr.cmpl_vext(), rshape.x % 2 == 0);
    kernel_correct_random_spectrum_write<<<gs, bs>>>(arr.ptr_cmpl(), dev_ptr_1, dev_ptr_2, arr.cmpl_vext(), rshape.x % 2 == 0);

    CUERR(cudaFree(dev_ptr_1));
    CUERR(cudaFree(dev_ptr_2));
    CUERR(cudaThreadSynchronize()); }


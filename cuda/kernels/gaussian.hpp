#include <cuda/kernels/common.hpp>

/*
__global__ void kernel_gaussian_mult(Float2* arr_kpsi, int3 rdims, int3 cdims, Float3 lens, Float3 hh, Float divs_per_sigma){
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float k0 = K(i0, rdims.x, lens.x);
    const Float k1 = K(i1, rdims.y, lens.y);
    const Float k2 = K(i2, rdims.z, lens.z);
    const Float3 sig = {Float(1.1) * divs_per_sigma * hh.x, Float(1.1) * divs_per_sigma * hh.y, Float(1.1) * divs_per_sigma * hh.z};
    const Float3 tmp = {sig.x * k0, sig.y * k1, sig.z * k2};
    const Float e = exp(- Float(0.5) * (tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z));
    arr_kpsi[idx] = {e * kpsi.x, e * kpsi.y};
}
void call_kernel_gaussian_mult(GPUArray& arr_kpsi, Float3 lens, Float3 hh, Float divs_per_sigma){
  Launch l(arr_kpsi.cmpl_vext());
  kernel_gaussian_mult<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), lens, hh, divs_per_sigma);
  CUERR(cudaThreadSynchronize());
  CUERR(cudaPeekAtLastError());
}
*/

__global__ void kernel_gaussian_init(Float2* arr_kpsi, int3 rdims, int3 cdims, Float3 lens, Float3 hh, Float divs_per_sigma){
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float k0 = K(i0, rdims.x, lens.x);
    const Float k1 = K(i1, rdims.y, lens.y);
    const Float k2 = K(i2, rdims.z, lens.z);
    const Float3 sig = {Float(1.1) * divs_per_sigma * hh.x, Float(1.1) * divs_per_sigma * hh.y, Float(1.1) * divs_per_sigma * hh.z};
    const Float3 tmp = {sig.x * k0, sig.y * k1, sig.z * k2};
    const Float e = exp(- Float(0.5) * (tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z));
    arr_kpsi[idx] = {e, Float(1) - e};
}
void call_kernel_gaussian_init(GPUArray& arr_kpsi, Float3 lens, Float3 hh, Float divs_per_sigma){
  Launch l(arr_kpsi.cmpl_vext());
  kernel_gaussian_init<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), lens, hh, divs_per_sigma);
  CUERR(cudaThreadSynchronize());
  CUERR(cudaPeekAtLastError());
}

__global__ void kernel_kmult_real(Float2* arr_1, Float2* arr_2, int3 rdims, int3 cdims){
    IDX012(cdims);
    const Float2 c1 = arr_1[idx];
    const Float2 c2 = arr_2[idx];
    arr_1[idx] = {c2.x * c1.x, c2.x * c1.y};
}
void call_kernel_kmult_real(GPUArray& arr_1, GPUArray& arr_2){
    assert(arr_1.real_vext() == arr_2.real_vext() and "array dimensions do not match");
    Launch l(arr_1.cmpl_vext());
    kernel_kmult_real<<<l.get_gs(), l.get_bs()>>>(arr_1.ptr_cmpl(), arr_2.ptr_cmpl(), arr_1.real_vext(), arr_1.cmpl_vext());
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError());
}

__global__ void kernel_kmult_imag(Float2* arr_1, Float2* arr_2, int3 rdims, int3 cdims){
    IDX012(cdims);
    const Float2 c1 = arr_1[idx];
    const Float2 c2 = arr_2[idx];
    arr_1[idx] = {c2.y * c1.x, c2.y * c1.y};
}
void call_kernel_kmult_imag(GPUArray& arr_1, GPUArray& arr_2){
    assert(arr_1.real_vext() == arr_2.real_vext() and "array dimensions do not match");
    Launch l(arr_1.cmpl_vext());
    kernel_kmult_imag<<<l.get_gs(), l.get_bs()>>>(arr_1.ptr_cmpl(), arr_2.ptr_cmpl(), arr_1.real_vext(), arr_1.cmpl_vext());
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError());
}


#include <cuda/kernels/common.hpp>

/*
__global__ void kernel_update_eom(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_knoise, int3 rdims, int3 cdims, Float dt, Float eps, Float3 lens){
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise = arr_knoise[idx];
    const Float2 dop = L_L2(K(i0, rdims.x, lens.x), K(i1, rdims.y, lens.y), K(i2, rdims.z, lens.z));
    const Float denom = Float(1) - dt * ((Float(2) - eps) * dop.x + Float(2) * dop.y + dop.x * dop.y);
    arr_kpsi[idx] = { (kpsi.x + dt * dop.x * knonlin.x + dt * knoise.x) / denom,
                      (kpsi.y + dt * dop.x * knonlin.y + dt * knoise.y) / denom }; }
*/

__global__ void kernel_update_eom(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_knoise, Float namp, int3 rdims, int3 cdims, Float dt, Float eps, Float3 lens){
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise = arr_knoise[idx];
    const Float2 dop = L_L2(K(i0, rdims.x, lens.x), K(i1, rdims.y, lens.y), K(i2, rdims.z, lens.z));
    const Float denom = Float(1) - dt * ((Float(2) - eps) * dop.x + Float(2) * dop.y + dop.x * dop.y);
    namp *= sqrtf(rdims.x * rdims.y * rdims.z);
    arr_kpsi[idx] = { (kpsi.x + dt * dop.x * knonlin.x + dt * namp * knoise.x) / denom,
                      (kpsi.y + dt * dop.x * knonlin.y + dt * namp * knoise.y) / denom }; }

void call_kernel_update_eom(GPUArray& arr_kpsi, GPUArray& arr_knonlin, GPUArray& arr_knoise, Float namp, Float dt, Float eps, Float3 lens){
  Launch l(arr_kpsi.cmpl_vext());
  kernel_update_eom<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), arr_knoise.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), namp, dt, eps, lens);
  CUERR(cudaThreadSynchronize()); }

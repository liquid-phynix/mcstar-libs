#include <cuda/kernels/common.hpp>

/*
// a zaj nem spektrumban volt generalva
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

__global__ void kernel_update_eom(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_knoise,
                                  int3 rdims, int3 cdims, Float namp, Float dt, Float eps, Float3 lens){
    namp *= sqrtf(rdims.x * rdims.y * rdims.z);
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise = arr_knoise[idx];
    const Float2 dop = L_L2(K(i0, rdims.x, lens.x), K(i1, rdims.y, lens.y), K(i2, rdims.z, lens.z));
    const Float k = sqrtf(- dop.x);
    const Float denom = Float(1) - dt * ((Float(2) - eps) * dop.x + Float(2) * dop.y + dop.x * dop.y);
    arr_kpsi[idx] = { (kpsi.x + dt * dop.x * knonlin.x + dt * namp * k * knoise.x) / denom,
                      (kpsi.y + dt * dop.x * knonlin.y + dt * namp * k * knoise.y) / denom };
}
void call_kernel_update_eom(GPUArray& arr_kpsi, GPUArray& arr_knonlin, GPUArray& arr_knoise,
                            Float namp, Float dt, Float eps, Float3 lens){
  Launch l(arr_kpsi.cmpl_vext());
  kernel_update_eom<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), arr_knoise.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), namp, dt, eps, lens);
  CUERR(cudaThreadSynchronize());
}

const Float py_c0  =  -57.94636917114257;
const Float py_c2  =   164.4074088310241;
const Float py_c4  =  -181.1481995196213;
const Float py_c6  =   100.6942650417575;
const Float py_c8  =  -30.20985042718309;
const Float py_c10 =   4.678065276755401;
const Float py_c12 =  -0.293423255328944;

__global__ void kernel_update_eom_py(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_knoise,
                                  int3 rdims, int3 cdims, Float namp, Float dt, Float eps, Float3 lens){
    namp *= sqrtf(rdims.x * rdims.y * rdims.z);
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise = arr_knoise[idx];
    const Float2 dop = L_L2(K(i0, rdims.x, lens.x), K(i1, rdims.y, lens.y), K(i2, rdims.z, lens.z));
    const Float k2 = - dop.x;
    const Float k = sqrtf(k2);

    Float denom = py_c12;
    denom *= k2; denom += py_c10;
    denom *= k2; denom += py_c8;
    denom *= k2; denom += py_c6;
    denom *= k2; denom += py_c4;
    denom *= k2; denom += py_c2;
    denom *= k2; denom += py_c0;

    denom = Float(1) - dt * dop.x * (Float(1) - eps - denom);
    arr_kpsi[idx] = { (kpsi.x + dt * dop.x * knonlin.x + dt * namp * k * knoise.x) / denom,
                      (kpsi.y + dt * dop.x * knonlin.y + dt * namp * k * knoise.y) / denom };
}

void call_kernel_update_eom_py(GPUArray& arr_kpsi, GPUArray& arr_knonlin, GPUArray& arr_knoise,
                               Float namp, Float dt, Float eps, Float3 lens){
  Launch l(arr_kpsi.cmpl_vext());
  kernel_update_eom_py<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), arr_knoise.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), namp, dt, eps, lens);
  CUERR(cudaThreadSynchronize());
}

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

__global__ void kernel_update_eom_hpfcnoise(Float2* arr_kpsi, Float2* arr_knonlin,
                                            Float2* arr_knoise1, Float2* arr_knoise2, Float2* arr_knoise3,
                                            int3 rdims, int3 cdims, Float namp, Float dt, Float eps, Float3 lens){
    namp *= sqrtf(rdims.x * rdims.y * rdims.z);
    IDX012(cdims);

    const Float kappa = Float(4) / Float(3);
    const Float c2 = sqrtf(Float(1) + kappa / Float(2) + sqrtf(Float(1) + kappa));
    const Float c1 = kappa / c2 / Float(2);

    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise1 = arr_knoise1[idx];
    const Float2 knoise2 = arr_knoise2[idx];
    const Float2 knoise3 = arr_knoise3[idx];
    const Float k0 = K(i0, rdims.x, lens.x);
    const Float k1 = K(i1, rdims.y, lens.y);
    const Float2 dop = L_L2(k0, k1, K(i2, rdims.z, lens.z));
    const Float k = sqrtf(- dop.x);
    const Float denom = Float(1) - dt * ((Float(2) - eps) * dop.x + Float(2) * dop.y + dop.x * dop.y);

    Float2 noise;
    noise.x = (c1 * k0 * k0 + c2 * k1 * k1) * knoise1.x + (c2 * k0 * k0 + c1 * k1 * k1) * knoise2.x + Float(2) * k0 * k1 * knoise3.x;
    noise.y = (c1 * k0 * k0 + c2 * k1 * k1) * knoise1.y + (c2 * k0 * k0 + c1 * k1 * k1) * knoise2.y + Float(2) * k0 * k1 * knoise3.y;
    noise.x = k <= Float(0.25) ? noise.x : Float(0);
    noise.y = k <= Float(0.25) ? noise.y : Float(0);

    arr_kpsi[idx] = { (kpsi.x + dt * dop.x * knonlin.x + dt * namp * noise.x) / denom,
                      (kpsi.y + dt * dop.x * knonlin.y + dt * namp * noise.y) / denom };
}
void call_kernel_update_eom_hpfcnoise(GPUArray& arr_kpsi, GPUArray& arr_knonlin,
                                      GPUArray& arr_knoise1, GPUArray& arr_knoise2, GPUArray& arr_knoise3,
                                      Float namp, Float dt, Float eps, Float3 lens){
  int3 shape = arr_kpsi.real_vext();
  if(shape.z != 1){
      std::cerr << "fuuuck this is only 2D, mate!" << std::endl;
      std::exit(EXIT_FAILURE);
  }
  Launch l(arr_kpsi.cmpl_vext());
  kernel_update_eom_hpfcnoise<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), arr_knoise1.ptr_cmpl(), arr_knoise2.ptr_cmpl(), arr_knoise3.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), namp, dt, eps, lens);
  CUERR(cudaThreadSynchronize());
  CUERR(cudaPeekAtLastError());
}

__global__ void kernel_update_eom(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_knoise,
                                  int3 rdims, int3 cdims, Float namp, Float dt, Float eps, Float3 lens){
    namp *= sqrtf(rdims.x * rdims.y * rdims.z);
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise = arr_knoise == NULL ? Float2({0,0}) : arr_knoise[idx];
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
  CUERR(cudaPeekAtLastError());
}
void call_kernel_update_eom(GPUArray& arr_kpsi, GPUArray& arr_knonlin,
                            Float namp, Float dt, Float eps, Float3 lens){
  Launch l(arr_kpsi.cmpl_vext());
  kernel_update_eom<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), NULL, arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), namp, dt, eps, lens);
  CUERR(cudaThreadSynchronize());
  CUERR(cudaPeekAtLastError());
}

__global__ void kernel_update_eom_with_fen(Float2* arr_kpsi, Float2* arr_knonlin, Float2* arr_knoise,
                                           int3 rdims, int3 cdims, Float namp, Float dt, Float eps, Float3 lens){
    namp *= sqrtf(rdims.x * rdims.y * rdims.z);
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 knoise = arr_knoise[idx];
    const Float2 dop = L_L2(K(i0, rdims.x, lens.x), K(i1, rdims.y, lens.y), K(i2, rdims.z, lens.z));
    const Float k = sqrtf(- dop.x);
    const Float lk = (Float(2) - eps) + Float(2) * dop.x + dop.y;
    const Float denom = Float(1) - dt * dop.x * lk;
    arr_kpsi[idx] = { (kpsi.x + dt * dop.x * knonlin.x + dt * namp * k * knoise.x) / denom,
        (kpsi.y + dt * dop.x * knonlin.y + dt * namp * k * knoise.y) / denom };
    arr_knoise[idx] = {lk * kpsi.x, lk * kpsi.y};
}
void call_kernel_update_eom_with_fen(GPUArray& arr_kpsi, GPUArray& arr_knonlin, GPUArray& arr_knoise,
        Float namp, Float dt, Float eps, Float3 lens){
    Launch l(arr_kpsi.cmpl_vext());
    kernel_update_eom_with_fen<<<l.get_gs(), l.get_bs()>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), arr_knoise.ptr_cmpl(), arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), namp, dt, eps, lens);
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError());
}

__global__ void kernel_calc_fen(Float* arr_psi, Float* arr_lpsi, int3 rdims, float* sum){
    extern __shared__ Float fen[];
    int fen_idx = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    fen[fen_idx] = Float(0);
    IDX012(rdims);
    const Float psi = arr_psi[idx];
    const Float lpsi = arr_lpsi[idx];
    fen[fen_idx] = 0.5 * psi * (lpsi - psi + 0.5 * psi * psi * psi);
    arr_lpsi[idx] = fen[fen_idx];

    __syncthreads();
    Float partial_sum = 0;
    if(threadIdx.x==0 and threadIdx.y==0 and threadIdx.z==0){
        for(int ii = 0; ii < blockDim.x * blockDim.y * blockDim.z; ii++)
            partial_sum += fen[ii];
        atomicAdd(sum, float(partial_sum));
    }
}

float call_kernel_calc_fen(GPUArray& arr_psi, GPUArray& arr_lpsi, Float3 hh){
    Launch l(arr_psi.real_vext());
    dim3 bs = l.get_bs();
    float* sum;
    CUERR(cudaMallocHost(&sum, sizeof(float)));
    kernel_calc_fen<<<l.get_gs(), bs, bs.x * bs.y * bs.z * sizeof(Float)>>>(arr_psi.ptr_real(), arr_lpsi.ptr_real(), arr_psi.real_vext(), sum);
    CUERR(cudaThreadSynchronize());
    float _sum = *sum * hh.x * hh.y * hh.z;
    CUERR(cudaFreeHost(sum));
    CUERR(cudaPeekAtLastError());
    return _sum;
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
  CUERR(cudaPeekAtLastError());
}

#include <cuda/kernels/common.hpp>

__global__ void kernel_update_el(Float2* arr_kpsi, Float2* arr_knonlin,
                                 float* maxes,
                                 int3 rdims, int3 cdims,
                                 Float dt, Float eps, Float3 lens){
    extern __shared__ float diff[];
    int diff_idx = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    diff[diff_idx] = 0.f;
    IDX012(cdims);
    const Float2 kpsi = arr_kpsi[idx];
    const Float2 knonlin = arr_knonlin[idx];
    const Float2 dop = L_L2(K(i0, rdims.x, lens.x), K(i1, rdims.y, lens.y), K(i2, rdims.z, lens.z));
    const Float denom = Float(1) + dt * ((Float(2) - eps) + Float(2) * dop.x + dop.y);
    const Float2 kpsip = { (kpsi.x - dt * knonlin.x) / denom,
                           (kpsi.y - dt * knonlin.y) / denom };
    arr_kpsi[idx] = kpsip;
    diff[diff_idx] = sqrt((kpsi.x - kpsip.x) * (kpsi.x - kpsip.x) + (kpsi.y - kpsip.y) * (kpsi.y - kpsip.y));
    __syncthreads();
    float maximum = 0;
    if(threadIdx.x==0 and threadIdx.y==0 and threadIdx.z==0){
        for(int ii = 0; ii < blockDim.x * blockDim.y * blockDim.z; ii++)
            maximum = diff[ii] > maximum ? diff[ii] : maximum;
        maxes[(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x] = maximum;
    }
}
Float call_kernel_update_el(GPUArray& arr_kpsi, GPUArray& arr_knonlin,
                           Float dt, Float eps, Float3 lens){
  Launch l(arr_kpsi.cmpl_vext());
  dim3 gs = l.get_gs();
  dim3 bs = l.get_bs();
  float* maxes = new float[gs.x * gs.y * gs.z];
  kernel_update_el<<<gs, bs, bs.x * bs.y * bs.z * sizeof(float)>>>(arr_kpsi.ptr_cmpl(), arr_knonlin.ptr_cmpl(), maxes, arr_kpsi.real_vext(), arr_kpsi.cmpl_vext(), dt, eps, lens);
  float max = 0;
  for(int ii = 0; ii < gs.x * gs.y * gs.z; ii++){
      max = max > maxes[ii] ? max : maxes[ii]; }
  delete[] maxes;
  CUERR(cudaThreadSynchronize());
  return max;
}


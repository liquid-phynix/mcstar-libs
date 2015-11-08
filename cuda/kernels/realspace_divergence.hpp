// calculate divergence of noise vector

// 2D
// XXX: KERNEL
__global__ void kernel_calc_noise_div_2d(Float* arr_eta0, Float* arr_eta1, Float* arr_divnoise, int3 rdims, Float3 hh){
  IDX012(rdims);
  int i0p = wrap(i0 + 1, rdims.x);
  int i0m = wrap(i0 - 1, rdims.x);
  int i1p = wrap(i1 + 1, rdims.y);
  int i1m = wrap(i1 - 1, rdims.y);
  arr_divnoise[idx] = (arr_eta0[calc_idx(i0p, i1, 0, rdims)] - arr_eta0[idx]) / hh.x +
                      (arr_eta1[calc_idx(i0, i1p, 0, rdims)] - arr_eta1[idx]) / hh.y; }

// 3D
// XXX: KERNEL
__global__ void kernel_calc_noise_div_3d(Float* arr_eta0, Float* arr_eta1, Float* arr_eta2, Float* arr_divnoise, int3 rdims, Float3 hh){
  IDX012(rdims);
  int i0p = wrap(i0 + 1, rdims.x);
  int i0m = wrap(i0 - 1, rdims.x);
  int i1p = wrap(i1 + 1, rdims.y);
  int i1m = wrap(i1 - 1, rdims.y);
  int i2p = wrap(i2 + 1, rdims.z);
  int i2m = wrap(i2 - 1, rdims.z);
  arr_divnoise[idx] = (arr_eta0[calc_idx(i0p, i1, i2, rdims)] - arr_eta0[idx]) / hh.x +
                      (arr_eta1[calc_idx(i0, i1p, i2, rdims)] - arr_eta1[idx]) / hh.y +
                      (arr_eta2[calc_idx(i0, i1, i2p, rdims)] - arr_eta2[idx]) / hh.z; }

void call_kernel_divergence(GPUArray* arr_eta0, GPUArray* arr_eta1, GPUArray* arr_eta2, GPUArray* arr_divnoise, Float3 hh){
  int3 shape = arr_divnoise->real_vext();
  Launch l(shape);
  if(arr_eta2 == NULL or shape.z == 1){
      assert(shape.x != 1 and shape.y != 1 and shape.z == 1 and "not a 2d array");
      kernel_calc_noise_div_2d<<<l.get_gs(), l.get_bs()>>>(arr_eta0->ptr_real(), arr_eta1->ptr_real(), arr_divnoise->ptr_real(), shape, hh);
  } else {
      assert(shape.x != 1 and shape.y != 1 and shape.z != 1 and "not a 3d array");
      kernel_calc_noise_div_3d<<<l.get_gs(), l.get_bs()>>>(arr_eta0->ptr_real(), arr_eta1->ptr_real(), arr_eta2->ptr_real(), arr_divnoise->ptr_real(), shape, hh);
  }
  CUERR(cudaThreadSynchronize()); }

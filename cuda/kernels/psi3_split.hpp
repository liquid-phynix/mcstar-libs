__global__ void kernel_psi3_split_pot(Float* arr, Float* pot, Float mul, int len){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  Float tmp;
  if(idx < len){
    tmp = arr[idx] * mul;
    arr[idx] = tmp * tmp * tmp - tmp + pot[idx]; }}

__global__ void kernel_psi3_split(Float* arr, Float mul, int len){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  Float tmp;
  if(idx < len){
    tmp = arr[idx] * mul;
    arr[idx] = tmp * tmp * tmp - tmp; }}

void call_kernel_psi3_split(GPUArray& arr, GPUArray* pot = NULL, bool norm = false){
    const int elems = arr.real_elems();
    Launch l(elems);
    Float nf = 1;
    if(norm)
        nf /= elems;
    if(pot == NULL)
        kernel_psi3_split<<<l.get_gs(), l.get_bs()>>>(arr.ptr_real(), nf, elems);
    else
        kernel_psi3_split_pot<<<l.get_gs(), l.get_bs()>>>(arr.ptr_real(), pot->ptr_real(), nf, elems);
    CUERR(cudaThreadSynchronize()); }

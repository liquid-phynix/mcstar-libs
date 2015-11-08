__global__ void kernel_norm(Float* arr, int len, Float nf){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < len) arr[idx] /= nf; }

void call_kernel_norm(GPUArray& arr){
    const int elems = arr.real_elems();
    Launch l(elems);
    kernel_norm<<<l.get_gs(), l.get_bs()>>>(arr.ptr_real(), elems, elems);
    CUERR(cudaThreadSynchronize()); }

// XXX: generate time correlated noise

// XXX: KERNEL
__global__ void kernel_calc_noise_ev(Float* arr_xi, Float* arr_eta, Float alpha, Float beta, int3 rdims){
    IDX012(rdims);
    arr_xi[idx] = arr_xi[idx] * alpha + arr_eta[idx] * beta; }
// XXX: CALL
void call_kernel_noise_evolution(GPUArray& arr_xi, GPUArray& arr_eta, Float alpha, Float beta){
    int3 shape = arr_eta.real_vext();
    Launch l(shape);
    kernel_calc_noise_ev<<<l.get_gs(), l.get_bs()>>>(arr_xi.ptr_real(), arr_eta.ptr_real(), alpha, beta, shape);
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError()); }


__global__ void _kernel_multassign_cmpl_with_real(Float2* arr1, Float2* arr2, int len){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Float2 val1, val2;
    if(idx < len){
        val1 = arr1[idx];
        val2 = arr2[idx];
        arr1[idx] = {val1.x * val2.x, val1.y * val2.x};
    }
}

void _call_kernel_multassign_cmpl_with_real(GPUArray& karr1, GPUArray& karr2){
    const int elems = karr1.cmpl_elems();
    Launch l(elems);
    _kernel_multassign_cmpl_with_real<<<l.get_gs(), l.get_bs()>>>(karr1.ptr_cmpl(), karr2.ptr_cmpl(), elems);
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError());
}

__global__ void _kernel_multassign_cmpl_with_imag(Float2* arr1, Float2* arr2, int len){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Float2 val1, val2;
    if(idx < len){
        val1 = arr1[idx];
        val2 = arr2[idx];
        arr1[idx] = {val1.x * val2.y, val1.y * val2.y};
    }
}

void _call_kernel_multassign_cmpl_with_imag(GPUArray& karr1, GPUArray& karr2){
    const int elems = karr1.cmpl_elems();
    Launch l(elems);
    _kernel_multassign_cmpl_with_imag<<<l.get_gs(), l.get_bs()>>>(karr1.ptr_cmpl(), karr2.ptr_cmpl(), elems);
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError());
}

__global__ void _kernel_square(Float* arr1, int len){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Float val1;
    if(idx < len){
        val1 = arr1[idx];
        arr1[idx] = val1 * val1;
    }
}

void _call_kernel_square(GPUArray& karr1){
    const int elems = karr1.real_elems();
    Launch l(elems);
    _kernel_square<<<l.get_gs(), l.get_bs()>>>(karr1.ptr_real(), elems);
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError());
}

void write_lowpass_2d(GPUArray& spectrum, GPUArray& kgaussian, CPUArray& host, Float3 ll, std::string fn){
    int3 shape = spectrum.real_vext();
    int3 cshape = spectrum.cmpl_vext();
    assert(shape == kgaussian.real_vext() and host.real_vext() == shape and "arrays have different dimensions");
    assert(shape.z == 1 and shape.y > 1 and shape.x > 1 and "array is not 2d");
    _call_kernel_multassign_cmpl_with_real(spectrum, kgaussian);
    spectrum >> host;
    char str2[32];
    char str1[256];
    sprintf(str1, "(S'n0'\nI%d\nS'n1'\nI%d\nS'l0'\nF%f\nS'l1'\nF%f\nd.\n", shape.x, shape.y, ll.x, ll.y);
    sprintf(str2, "LOWPASS\n%d\n", strlen(str1));
    std::ostringstream os;
    os << str2 << str1;
    Float2* ptr = host.ptr_cmpl();
    for(int i0 = 0; i0 < cshape.x; i0++){
        for(int i1 = 0; i1 < cshape.y; i1++){
            Float2 val = ptr[i1 * cshape.x + i0];
            if(val.x * val.x + val.y * val.y > 1e-6)
                os << i0 << " " << i1 << " " << val.x << " " << val.y << "\n";
        }
    }
    ozstream ozs(fn.c_str());
    ozs << os.str();
}
void calc_amplitude_lowpass(PlanR2C& r2c, PlanC2R& c2r, GPUArray& spectrum, GPUArray& aux, GPUArray& kgaussian, CPUArray& host, Float3 ll, std::string fn){
    int3 shape = spectrum.real_vext();
    int3 cshape = spectrum.cmpl_vext();
    assert(shape == aux.real_vext() and host.real_vext() == shape and shape == kgaussian.real_vext() and "arrays have different dimensions");
    assert(shape.z == 1 and shape.y > 1 and shape.x > 1 and "array is not 2d");

    _call_kernel_multassign_cmpl_with_imag(spectrum, kgaussian);
    c2r.execute(spectrum, aux);
    call_kernel_norm(aux);
    _call_kernel_square(aux);
    r2c.execute(aux, spectrum);
    write_lowpass_2d(spectrum, kgaussian, host, ll, fn);
    c2r.execute(spectrum, aux);
    call_kernel_norm(aux);
    aux >> host;
}

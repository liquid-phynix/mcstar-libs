#include <bond-order-analysis/bond-order.hpp>

typedef Maximum<Float> Max;

__device__ int gpu_maxlist_len;

__global__ void kernel_findpeaks(Float* arr, int3 rdims, Max* host_arr, Float ths){
    IDX012(rdims);
    Float localval = arr[idx];
    char local_peak = 1;
    for(int i0_dir = -1; i0_dir < 2; i0_dir++){
        for(int i1_dir = -1; i1_dir < 2; i1_dir++){
            for(int i2_dir = -1; i2_dir < 2; i2_dir++){
                if(i0_dir==0 and i1_dir==0 and i2_dir==0) continue;
                int oidx = calc_idx(wrap(i0 + i0_dir, rdims.x), wrap(i1 + i1_dir, rdims.y), wrap(i2 + i2_dir, rdims.z), rdims);
                local_peak *= localval > arr[oidx];
            }
        }
    }
    if(local_peak and localval >= ths){
        int host_idx = atomicAdd(&gpu_maxlist_len, 1);
        Max m;
        m.i0 = i0;
        m.i1 = i1;
        m.i2 = i2;
        m.field = localval;
        host_arr[host_idx] = m;
    }
}

void call_kernel_findpeaks(GPUArray& input, std::string peaks_outfn, std::string oorder_outfn, Float3 hh, Float3 len, Float ths, Float rlim){
    if(len.x != len.y or len.x != len.z){
        std::cerr << "orientational mapping needs a cube" << std::endl;
        return; }
    int elems = input.real_elems();
    Max* host_ptr;
    CUERR(cudaHostAlloc((void**)&host_ptr, elems * sizeof(Max), cudaHostAllocDefault));
    int num_of_maxima= 0;
    Launch l(input.real_vext());
    CUERR(cudaMemcpyToSymbol(gpu_maxlist_len, &num_of_maxima, sizeof(int)));
    kernel_findpeaks<<<l.get_gs(), l.get_bs()>>>(input.ptr_real(), input.real_vext(), host_ptr, ths);
    CUERR(cudaMemcpyFromSymbol(&num_of_maxima, gpu_maxlist_len, sizeof(int)));
    std::cerr << "found " << num_of_maxima << " maxima" << std::endl;

    FILE* outfile = fopen(peaks_outfn.c_str(), "w");
    if(outfile != NULL){
        for(int i = 0; i < num_of_maxima; i++){
            Max& m = host_ptr[i];
            float x0 = m.i0 * hh.x, x1 = m.i1 * hh.y, x2 = m.i2 * hh.z;
            fprintf(outfile, "%f %f %f %f\n", x0, x1, x2, m.field);
        }
        fclose(outfile);
    } else std::cout << "output file <" << peaks_outfn << "> cannot be opened for writing" << std::endl;

    outfile=fopen(oorder_outfn.c_str(), "w");
    if(outfile){
        bond_order_analysis(host_ptr, num_of_maxima, double(hh.x), double(hh.y), double(hh.z), double(ths), double(len.x), double(rlim), outfile);
        fclose(outfile);
    } else std::cout << "output file <" << oorder_outfn << "> cannot be opened for writing" << std::endl;

    CUERR(cudaFreeHost(host_ptr));
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError()); }

__global__ void kernel_findpeaks_2d(Float* arr, int3 rdims, Max* host_arr, Float ths){
    IDX012(rdims);
    Float localval = arr[idx];
    char local_peak = 1;
    for(int i0_dir = -1; i0_dir < 2; i0_dir++){
        for(int i1_dir = -1; i1_dir < 2; i1_dir++){
                if(i0_dir==0 and i1_dir==0) continue;
                int oidx = calc_idx(wrap(i0 + i0_dir, rdims.x), wrap(i1 + i1_dir, rdims.y), 0, rdims);
                local_peak *= localval > arr[oidx];
        }
    }
    if(local_peak and localval >= ths){
        int host_idx = atomicAdd(&gpu_maxlist_len, 1);
        Max m;
        m.i0 = i0;
        m.i1 = i1;
        m.i2 = 0;
        m.field = localval;
        host_arr[host_idx] = m;
    }
}

void call_kernel_findpeaks_2d(GPUArray& input, std::string peaks_outfn, Float3 hh, Float3 len, Float ths){
    int3 shape = input.real_vext();
    if(not (shape.x > 1 and shape.y > 1 and shape.z == 1)){
        std::cerr << "input not a 2d array" << std::endl;
        return; }
    int elems = input.real_elems();
    Max* host_ptr;
    CUERR(cudaHostAlloc((void**)&host_ptr, elems * sizeof(Max), cudaHostAllocDefault));
    int num_of_maxima= 0;
    Launch l(input.real_vext());
    CUERR(cudaMemcpyToSymbol(gpu_maxlist_len, &num_of_maxima, sizeof(int)));
    kernel_findpeaks_2d<<<l.get_gs(), l.get_bs()>>>(input.ptr_real(), input.real_vext(), host_ptr, ths);
    CUERR(cudaMemcpyFromSymbol(&num_of_maxima, gpu_maxlist_len, sizeof(int)));
    std::cerr << "found " << num_of_maxima << " maxima" << std::endl;

    FILE* outfile = fopen(peaks_outfn.c_str(), "w");
    if(outfile != NULL){
        for(int i = 0; i < num_of_maxima; i++){
            Max& m = host_ptr[i];
            float x0 = m.i0 * hh.x, x1 = m.i1 * hh.y, x2 = m.i2 * hh.z;
            fprintf(outfile, "%f %f %f %f\n", x0, x1, x2, m.field);
        }
        fclose(outfile);
    } else std::cout << "output file <" << peaks_outfn << "> cannot be opened for writing" << std::endl;

    CUERR(cudaFreeHost(host_ptr));
    CUERR(cudaThreadSynchronize());
    CUERR(cudaPeekAtLastError()); }

#pragma once

#include <cufft.h>
#include <cuda/array_abstraction.hpp>

namespace CuErr {
#include <cstdio>
    static const char* cufftGetErrorString(cufftResult error){
        switch(error){
            case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
            case CUFFT_UNALIGNED_DATA:            return "CUFFT_UNALIGNED_DATA";
#if CUDA_VERSION >= 5000
                                                  // newer versions of cufft.h define these too
            case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
            case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
            case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
            case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
#endif
            default:                              return "<unknown>";
        }
    }
    inline void gpuCufftAssert(cufftResult code, const char* file, int line){
        if(code != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFTERR: %s %s:%d\n", cufftGetErrorString(code), file, line);
            fflush(stderr);
            exit(code);
        }
    }
}
#define CUFFTERR(ans) CuErr::gpuCufftAssert((ans), __FILE__, __LINE__);

namespace FFT {
    struct R2C {};
    struct C2R {};

    template <typename, typename> struct CufftType { static cufftType type; };
    template <> cufftType CufftType<float,  R2C>::type = CUFFT_R2C;
    template <> cufftType CufftType<float,  C2R>::type = CUFFT_C2R;
    template <> cufftType CufftType<double, R2C>::type = CUFFT_D2Z;
    template <> cufftType CufftType<double, C2R>::type = CUFFT_Z2D;

    template <typename F, typename P> void exec(cufftHandle&, Array::GPUArray<F>&, Array::GPUArray<F>&);
    template <> void exec<float, R2C>(cufftHandle& plan, Array::GPUArray<float>& in, Array::GPUArray<float>& out){
        CUFFTERR(cufftExecR2C(plan, in.ptr_real(), out.ptr_cmpl())); };
    template <> void exec<float, C2R>(cufftHandle& plan, Array::GPUArray<float>& in, Array::GPUArray<float>& out){
        CUFFTERR(cufftExecC2R(plan, in.ptr_cmpl(), out.ptr_real())); };
    template <> void exec<double, R2C>(cufftHandle& plan, Array::GPUArray<double>& in, Array::GPUArray<double>& out){
        CUFFTERR(cufftExecD2Z(plan, in.ptr_real(), out.ptr_cmpl())); };
    template <> void exec<double, C2R>(cufftHandle& plan, Array::GPUArray<double>& in, Array::GPUArray<double>& out){
        CUFFTERR(cufftExecZ2D(plan, in.ptr_cmpl(), out.ptr_real())); };


    template <typename F, typename FT> class CufftPlan {
        private:
            int3 m_shape;
            int3 m_shape_tr;
            const cufftType m_type;
            cufftHandle m_plan;
        public:
            CufftPlan(int3 shape):
                m_shape(shape),
                m_shape_tr(shape_tr(shape)),
                m_type(CufftType<F, FT>::type){ // logical (real) problem dimensions
                    int dims = 3;
                    if(m_shape.z == 1) dims = 2;
                    if(m_shape.y == 1) dims = 1;
                    CUFFTERR(cufftPlanMany(&m_plan, dims, (int*)&m_shape_tr, NULL, 0, 0, NULL, 0, 0, m_type, 1));
                    // native data layout, nincs paddingelve a valos repr. mint fftw-ben
                    CUFFTERR(cufftSetCompatibilityMode(m_plan, CUFFT_COMPATIBILITY_NATIVE)); }

            ~CufftPlan(){ CUFFTERR(cufftDestroy(m_plan)); }
            void execute(Array::GPUArray<F>& arr){ execute(arr, arr); }
            void execute(Array::GPUArray<F>& in, Array::GPUArray<F>& out){
                assert(in.real_vext() == m_shape and out.real_vext() == m_shape
                        and "array dimensions in plan and execute phase do not match up");
                exec<F, FT>(m_plan, in, out);
                CUERR(cudaThreadSynchronize()); }
    };
}

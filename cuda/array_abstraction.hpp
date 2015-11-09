#pragma once

#include <iostream>
#include <complex>
#include <string>
#include <functional>
#include <sys/mman.h>
#include <cassert>
#include <stdexcept>
#include <future>
#include <common/npyutil.hpp>
#include <common/cu_error.hpp>
#include <common/technicalities.hpp>

namespace Array {

    struct AsReal {};
    struct AsCmpl {};

    // type provider
    template <typename> struct TP {};
    template <> struct TP <float> {
        typedef float  RT; typedef float2  CT; };
    template <> struct TP <double> {
        typedef double RT; typedef double2 CT; };

    template <typename F, typename Kind> void numpy_save(const std::string, int, int*, int*, void*);
    template <> void numpy_save<float, AsReal>(const std::string fn, int dims, int* rshape, int* cshape, void* ptr){
        aoba::SaveArrayAsNumpy(fn, dims, rshape, reinterpret_cast<float*>(ptr)); }
    template <> void numpy_save<double, AsReal>(const std::string fn, int dims, int* rshape, int* cshape, void* ptr){
        aoba::SaveArrayAsNumpy(fn, dims, rshape, reinterpret_cast<double*>(ptr)); }
    template <> void numpy_save<float, AsCmpl>(const std::string fn, int dims, int* rshape, int* cshape, void* ptr){
        aoba::SaveArrayAsNumpy(fn, dims, cshape, reinterpret_cast<std::complex<float>*>(ptr)); }
    template <> void numpy_save<double, AsCmpl>(const std::string fn, int dims, int* rshape, int* cshape, void* ptr){
        aoba::SaveArrayAsNumpy(fn, dims, cshape, reinterpret_cast<std::complex<double>*>(ptr)); }

    template <typename F> class Array {
        protected:
            F* m_ptr;
            const int m_dims;
            int m_real_elems;
            int3 m_ext_real;
            int3 m_ext_cmpl;
            int m_cmpl_elems;
            size_t m_bytes;
            Array(int3 ext):
                m_ptr(),
                m_dims(ext.z == 1 ? (ext.y == 1 ? 1 : 2) : 3),
                m_real_elems(ext.x * ext.y * ext.z),
                m_ext_real(ext),
                m_ext_cmpl({ext.x/2+1, ext.y, ext.z}),
                m_cmpl_elems(m_ext_cmpl.x * m_ext_cmpl.y * m_ext_cmpl.z),
                m_bytes(m_cmpl_elems * 2 * sizeof(F)){
                    if(ext.x < 1 or ext.y < 1 or ext.z < 1)
                        throw std::runtime_error("extent must be >=1 in all dimensions"); }
        public:
            int real_elems(){ return m_real_elems; }
            int cmpl_elems(){ return m_cmpl_elems; }
            int3 real_vext(){ return m_ext_real; }
            int3 cmpl_vext(){ return m_ext_cmpl; }
            int  real_ext(int i){ return ((int*)&m_ext_real)[i]; }
            int  cmpl_ext(int i){ return ((int*)&m_ext_cmpl)[i]; }
            // ptr accessors
            void*               ptr_void(){ return m_ptr; }
            F*                  ptr_real(){ return m_ptr; }
            typename TP<F>::CT* ptr_cmpl(){ return reinterpret_cast<typename TP<F>::CT*>(m_ptr); }

            typedef F RT;
            typedef typename TP<F>::CT CT;
    };

    template <typename> class CPUArray;

    template <typename F> struct GPUArray : public Array <F> {
        GPUArray(CPUArray<F>& from): Array<F>(from.real_vext()){
            CUERR(cudaMalloc((void**)&this->m_ptr, this->m_bytes));
            CUERR(cudaPeekAtLastError());
            from >> (*this); }
        GPUArray(int3 ext): Array<F>(ext){
            CUERR(cudaMalloc((void**)&this->m_ptr, this->m_bytes));
            CUERR(cudaPeekAtLastError()); }
        ~GPUArray(){ CUERR(cudaFree(this->m_ptr)); }
        void operator>>(GPUArray<F>& o){
            assert(o.real_vext() == this->m_ext_real and "array dimensions differ");
            cudaMemcpy(o.ptr_void(), this->m_ptr, this->m_bytes, cudaMemcpyDeviceToDevice);
            CUERR(cudaPeekAtLastError()); }
        void operator>>(CPUArray<F>& o){
            assert(o.real_vext() == this->m_ext_real and "array dimensions differ");
            cudaMemcpy(o.ptr_void(), this->m_ptr, this->m_bytes, cudaMemcpyDeviceToHost);
            CUERR(cudaPeekAtLastError()); }
        void operator%(GPUArray<F>& o){
            assert(o.real_vext() == this->m_ext_real and "array dimensions differ");
            F* ptr = o.m_ptr;
            o.m_ptr = this->m_ptr;
            this->m_ptr = ptr; }
    };

    template <typename F> struct CPUArray : public Array <F> {
        F* m_ptr_save;
        std::future<void> m_done_saving;
        CPUArray(CPUArray<F>& from): Array<F>(from.real_vext()){
            m_ptr_save = NULL;
            CUERR(cudaHostAlloc((void**)&this->m_ptr, this->m_bytes, cudaHostAllocDefault));
            CUERR(cudaPeekAtLastError());
            if(mlock(this->m_ptr, this->m_bytes) != 0)
                std::cerr << "*** host memory not pinned" << std::endl;
            from >> (*this); }
        CPUArray(int3 ext): Array<F>(ext){
            m_ptr_save = NULL;
            CUERR(cudaHostAlloc((void**)&this->m_ptr, this->m_bytes, cudaHostAllocDefault));
            CUERR(cudaPeekAtLastError());
            if(mlock(this->m_ptr, this->m_bytes) != 0)
                std::cerr << "*** host memory not pinned" << std::endl;
            memset(this->ptr_void(), 0, this->m_bytes); }
        ~CPUArray(){
            CUERR(cudaFreeHost(this->m_ptr));
            if(m_ptr_save)
                CUERR(cudaFreeHost(this->m_ptr_save)); }
        void operator>>(CPUArray<F>& o){
            if(m_done_saving.valid()) m_done_saving.wait();
            assert(o.real_vext() == this->m_ext_real and "array dimensions differ");
            memcpy(o.ptr_void(), this->m_ptr, this->m_bytes); }
        void operator>>(GPUArray<F>& o){
            assert(o.real_vext() == this->m_ext_real and "array dimensions differ");
            cudaMemcpy(o.ptr_void(), this->m_ptr, this->m_bytes , cudaMemcpyHostToDevice);
            CUERR(cudaPeekAtLastError()); }
        void operator%(CPUArray<F>& o){
            assert(o.real_vext() == this->m_ext_real and "array dimensions differ");
            F* ptr = o.m_ptr;
            o.m_ptr = this->m_ptr;
            this->m_ptr = ptr; }
        // idx[0] >> idx[1] >> idx[2]
        // idx.x >> idx.y >> idx.z
        inline F& operator[](int idx){
            return this->m_ptr[idx]; }
        inline F& operator[](int3 idx){
            return this->m_ptr[(idx.z * this->m_ext_real.y + idx.y) * this->m_ext_real.x + idx.x]; }
        void load(const std::string fn){
            std::vector<F> data; std::vector<int> shape;
            // numpy: 0 << 1 << 2
            aoba::LoadArrayFromNumpy(fn, shape, data);
            int3 _shape{1,1,1};
            switch(shape.size()){
                case 1: _shape.x = shape[0]; break;
                case 2: _shape.x = shape[1]; _shape.y = shape[0]; break;
                case 3: _shape.x = shape[2]; _shape.y = shape[1]; _shape.z = shape[0]; break;
                default: throw std::runtime_error("shouldn't happen"); }
                         if(not (_shape == this->m_ext_real)){
                             fprintf(stderr, "*** array dimension mismatch" );
                             fflush(stderr);
                             abort(); }
                         memcpy(this->m_ptr, data.data(), this->m_bytes);
                         fprintf(stderr, "*** array initialized from file <%s>\n", fn.c_str()); fflush(stderr); }
        template <typename Kind=AsReal> void save(const std::string fn){
            std::cout << "entered saving " << fn << std::endl;
            if(m_done_saving.valid()) m_done_saving.wait();
            if(this->m_ptr_save == NULL){
                CUERR(cudaHostAlloc((void**)&this->m_ptr_save, this->m_bytes, cudaHostAllocDefault));
                CUERR(cudaPeekAtLastError());
            }
            memcpy(this->m_ptr_save, this->m_ptr, this->m_bytes);
            m_done_saving = std::async([this, &fn](){
                std::cout << "started saving " << fn << std::endl;
                int3 rshape = shape_tr(this->real_vext());
                int3 cshape = shape_tr(this->cmpl_vext());
                numpy_save<F, Kind>(fn, this->m_dims, (int*)&rshape, (int*)&cshape, this->m_ptr_save);
                std::cout << "finished saving " << fn << std::endl;
            });
        }
        //template <typename Kind = AsReal> void save(const boost::format fmt){
            //this->save(fmt.str()); }
        void over(std::function<void (F&)> closure){
            int maxi = this->m_ext_real.x * this->m_ext_real.y * this->m_ext_real.z;
            for(int i = 0; i < maxi; i++) closure(this->m_ptr[i]); }
        void over(std::function<void (typename TP<F>::CT& v)> closure){
            int maxi = this->m_ext_cmpl.x * this->m_ext_cmpl.y * this->m_ext_cmpl.z;
            for(int i = 0; i < maxi; i++) closure(this->ptr_cmpl()[i]); }
        void over(std::function<void (F&, int)> closure){
            int maxi = this->m_ext_real.x * this->m_ext_real.y * this->m_ext_real.z;
            for(int i = 0; i < maxi; i++) closure(this->m_ptr[i], i); }
        void over(std::function<void (typename TP<F>::CT&, int)> closure){
            int maxi = this->m_ext_cmpl.x * this->m_ext_cmpl.y * this->m_ext_cmpl.z;
            for(int i = 0; i < maxi; i++) closure(this->ptr_cmpl()[i], i); }
        void square(){
            this->over([](F& v){ v *= v; }); }
        void set_to(F val){
            this->over([val](F& v){ v = val; }); }
        void mult(F val){
            this->over([val](F& v){ v *= val; }); }
        void add(F val){
            this->over([val](F& v){ v += val; }); }
        F max(){
            F _max = this->ptr_real()[0];
            this->over([&_max](F& v){ _max = _max > v ? _max : v; });
            return _max; }
        F min(){
            F _min = this->ptr_real()[0];
            this->over([&_min](F& v){ _min = _min < v ? _min : v; });
            return _min; }
    };
}

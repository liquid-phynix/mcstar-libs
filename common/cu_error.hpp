#pragma once

namespace CUERROR {
#include <cstdio>
    inline void gpuAssert(cudaError_t code, const char* file, int line){
        if(code != cudaSuccess){
            fprintf(stderr, "CUERR: %s %s:%d\n", cudaGetErrorString(code), file, line);
            fflush(stderr);
            exit(code); }}
}
#define CUERR(ans) CUERROR::gpuAssert((ans), __FILE__, __LINE__);

#pragma once

#include <cstdio>
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE);}} while(0)

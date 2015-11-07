#define IDX012(dims) \
int i0 = blockIdx.x * blockDim.x + threadIdx.x; \
int i1 = blockIdx.y * blockDim.y + threadIdx.y; \
int i2 = blockIdx.z * blockDim.z + threadIdx.z; \
int idx = calc_idx(i0, i1, i2, dims); \
if(i0 >= dims.x or i1 >= dims.y or i2 >= dims.z) return;

__forceinline__ __device__ int wrap(int i, int n){
    return i < 0 ? (i + n) : (i >= n ? (i - n) : i); }

__forceinline__ __device__ int calc_idx(int i0, int i1, int i2, int3 dims){
    return (i2 * dims.y + i1) * dims.x + i0; }

#ifdef defined(Float) && defined(Float2)
// laplace^2
__forceinline__ __device__ Float2 L_L2(Float k0, Float k1, Float k2){
    Float2 ret;
    ret.x = - k0 * k0 - k1 * k1 - k2 * k2;
    ret.y = ret.x * ret.x;
    return ret; }

// calculate wavenumber from array index and domain length
//__forceinline__ __device__ Float K(const int i, const int n, const int np2m1, const Float f){
    //return i < np2m1 ? (i * f) : (f * (i - n)); }
__forceinline__ __device__ Float K(int i, int n, Float len){
return (i < n / 2 + 1 ? i : i - n) * Float(6.283185307179586232) / len; }
#endif

unsigned int div_up(int a, int b){
    const div_t r = div(a, b);
    return r.quot + int(r.rem > 0); }

struct Launch {
    const dim3 bs1d = {512};
    const dim3 bs2d = {16, 32};
    const dim3 bs3d = {8, 8, 8};
    dim3 bs, gs;
    Launch(int n0): Launch(int3{n0, 1, 1}){}
    Launch(int3 shape){
        if(shape.y == 1 and shape.z ==1){
            bs = bs1d;
            gs = {div_up(shape.x, bs.x)};
        } else if(shape.z == 1){
            bs = bs2d;
            gs = {div_up(shape.x, bs.x), div_up(shape.y, bs.y)};
        } else {
            bs = bs3d;
            gs = {div_up(shape.x, bs.x), div_up(shape.y, bs.y), div_up(shape.z, bs.z)};
        }
    }
    dim3 get_bs(){ return bs; }
    dim3 get_gs(){ return gs; }
};

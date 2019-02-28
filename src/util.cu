#include "util.h"



#if FERMI
__shared__ int interBuff[GPU_THREADS_PER_BLOCK];
__inline__ __device__ int __emulated_shfl(const int scalarValue, const uint32_t source_lane) {
    const int warpIdx = threadIdx.x / WARP_SIZE;
    const int laneIdx = threadIdx.x % WARP_SIZE;
    volatile int *interShuffle = interBuff + (warpIdx * WARP_SIZE);
    interShuffle[laneIdx] = scalarValue;
    return(interShuffle[source_lane % WARP_SIZE]);
}
#endif


__inline__ __device__ int shfl_32(int scalarValue, const int lane) {
    #if FERMI
        return __emulated_shfl(scalarValue, (uint32_t)lane);
    #else
        return __shfl(scalarValue, lane);
    #endif
}


/*
__inline__ __device__ int shfl_up_32(int scalarValue, const int n) {         
    #if FERMI
        int lane = threadIdx.x % WARP_SIZE;
        lane -= n;
        return shfl_32(scalarValue, lane);
    #else
        return __shfl_up(scalarValue, n);
    #endif
}



__inline__ __device__ int shfl_down_32(int scalarValue, const int n) {
    #if FERMI
        int lane = threadIdx.x % WARP_SIZE;
        line += n;
        return shfl_32(scalarValue, lane);
    #else
        return __shfl_down(scalarValue, n);
    #endif
}
*/



/*
__inline__ __device__ int shfl_xor_32(int scalarValue, const int n) {
    #if FERMI
        int lane = threadIdx.x % WARP_SIZE;
        lane = lane ^ n;
        return shfl_32(scalarValue, lane);
    #else
        return __shfl_xor(scalarValue, n);
    #endif
}
*/



/*
__device__ __forceinline__ void uint32_to_uchars(const uint32_t s, int *u1, int *u2, int *u3, int *u4) {
    //*u1 = s & 0xff;
    *u1 = __byte_perm(s, 0, 0x4440);
    //*u2 = (s>>8) & 0xff;
    *u2 = __byte_perm(s, 0, 0x4441);
    //*u3 = (s>>16) & 0xff;
    *u3 = __byte_perm(s, 0, 0x4442);
    //*u4 = s>>24;
    *u4 = __byte_perm(s, 0, 0x4443);
}
*/


/*
__device__ __forceinline__ uint32_t uchars_to_uint32(int u1, int u2, int u3, int u4) {
    //return u1 | (u2<<8) | (u3<<16) | (u4<<24);
    //return __byte_perm(u1, u2, 0x7740) + __byte_perm(u3, u4, 0x4077);
    return u1 | (u2<<8) | __byte_perm(u3, u4, 0x4077);
}


__device__ __forceinline__ uint32_t uchar_to_uint32(int u1) {
    return __byte_perm(u1, u1, 0x0);
}
*/


/*
__device__ __forceinline__ void print_uchars(const char* str, const uint32_t s) {
    int u1, u2, u3, u4;
    uint32_to_uchars(s, &u1, &u2, &u3, &u4);
    printf("%s: %d %d %d %d\n", str, u1, u2, u3, u4);
}
*/


/*
__inline__ __device__ int warpReduceMinIndex2(int *val, int idx) {
    for(int d = 1; d < WARP_SIZE; d *= 2) {
        int tmp = shfl_xor_32(*val, d);
        int tmp_idx = shfl_xor_32(idx, d);
        if(*val > tmp) {
            *val = tmp;
            idx = tmp_idx;
        }
    }
    return idx;
}



__inline__ __device__ int warpReduceMinIndex(int val, int idx) {
    for(int d = 1; d < WARP_SIZE; d *= 2) {
        int tmp = shfl_xor_32(val, d);
        int tmp_idx = shfl_xor_32(idx, d);
        if(val > tmp) {
            val = tmp;
            idx = tmp_idx;
        }
    }
    return idx;
}
*/



/*
__inline__ __device__ int warpReduceMin(int val) {
    val = min(val, shfl_xor_32(val, 1));
    val = min(val, shfl_xor_32(val, 2));
    val = min(val, shfl_xor_32(val, 4));
    val = min(val, shfl_xor_32(val, 8));
    val = min(val, shfl_xor_32(val, 16));
    return val;
}
*/

/*

__inline__ __device__ int blockReduceMin(int val) {
    static __shared__ int shared[WARP_SIZE]; // Shared mem for WARP_SIZE partial sums
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMin(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : INT_MAX;

    if (wid==0) val = warpReduceMin(val); //Final reduce within first warp

    return val;
}



__inline__ __device__ int blockReduceMinIndex(int val, int idx) {
    static __shared__ int shared_val[WARP_SIZE]; // Shared mem for WARP_SIZE partial mins
    static __shared__ int shared_idx[WARP_SIZE]; // Shared mem for WARP_SIZE indexes
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    idx = warpReduceMinIndex2(&val, idx);     // Each warp performs partial reduction

    if (lane==0) {
        shared_val[wid]=val;
        shared_idx[wid]=idx;
    }

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_val[lane] : INT_MAX;
    idx = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_idx[lane] : INT_MAX;

    if (wid==0) {
        idx = warpReduceMinIndex2(&val, idx); //Final reduce within first warp
    }

    return idx;
}
*/



__inline__ __device__ bool blockAny(bool local_condition) {
    __shared__ bool conditions[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    local_condition = __any(local_condition);     // Each warp performs __any

    if (lane==0) {
        conditions[wid]=local_condition;
    }

    __syncthreads();              // Wait for all partial __any

    //read from shared memory only if that warp existed
    local_condition = (threadIdx.x < blockDim.x / WARP_SIZE) ? conditions[lane] : false;

    if (wid==0) {
        local_condition = __any(local_condition); //Final __any within first warp
    }

    return local_condition;
}


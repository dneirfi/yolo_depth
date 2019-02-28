/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/
#include "disparity_method.h"


static cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;
static uint8_t *h_disparity;
static uint16_t *d_S;
static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint8_t *d_L4;
static uint8_t *d_L5;
static uint8_t *d_L6;
static uint8_t *d_L7;
static uint8_t p1, p2;
static bool first_alloc;
static uint32_t cols, rows, size, size_cube_l;



__device__ __forceinline__ uint32_t uchar_to_uint32(int u1) {
    return __byte_perm(u1, u1, 0x0);
}



__device__ __forceinline__ uint32_t uchars_to_uint32(int u1, int u2, int u3, int u4) {
    return u1 | (u2<<8) | __byte_perm(u3, u4, 0x4077);
}


__inline__ __device__ int shfl_up_32(int scalarValue, const int n) 
{
    #if FERMI$
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
        lane += n;
        return shfl_32(scalarValue, lane);
    #else
        return __shfl_down(scalarValue, n);
    #endif
}



__inline__ __device__ int shfl_xor_32(int scalarValue, const int n) 
{
    #if FERMI
        int lane = threadIdx.x % WARP_SIZE;
        line = lane ^ n;
        return shfl_32(scalarValue, lane);
    #else
        return __shfl_xor(scalarValue, n);
    #endif
}



__inline__ __device__ int warpReduceMin(int val) 
{
    val = min(val, shfl_xor_32(val, 1));
    val = min(val, shfl_xor_32(val, 2));
    val = min(val, shfl_xor_32(val, 4));
    val = min(val, shfl_xor_32(val, 8));
    val = min(val, shfl_xor_32(val, 16));
    
    return val;
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





void init_disparity_method(const uint8_t _p1, const uint8_t _p2) {
	// We are not using shared memory, use L1
	//CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	//CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	// Create streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
	first_alloc = true;
	p1 = _p1;
	p2 = _p2;
    rows = 0;
    cols = 0;
}








template<class T, int iter_type, int min_type, int dir_type, bool first_iteration, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGenericIteration(int index, int index_im, int col, uint32_t *old_values, int *old_value1, int *old_value2, int *old_value3, int *old_value4, uint32_t *min_cost, uint32_t *min_cost_p2, uint8_t* d_cost, uint8_t *d_L, const int p1_vector, const int p2_vector, const T *_d_transform0, const T *_d_transform1, const int lane, const int MAX_PAD, const int dis, T *rp0, T *rp1, T *rp2, T *rp3, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const T __restrict__ *d_transform0 = _d_transform0;
    const T __restrict__ *d_transform1 = _d_transform1;
    uint32_t costs, next_dis, prev_dis;

    if(iter_type == ITER_NORMAL) {
        // First shuffle
        int prev_dis1 = shfl_up_32(*old_value4, 1);
        if(lane == 0) {
            prev_dis1 = MAX_PAD;
        }

        // Second shuffle
        int next_dis4 = shfl_down_32(*old_value1, 1);
        if(lane == 31) {
            next_dis4 = MAX_PAD;
        }

        // Shift + rotate
        //next_dis = __funnelshift_r(next_dis4, *old_values, 8);
        next_dis = __byte_perm(*old_values, next_dis4, 0x4321);
        prev_dis = __byte_perm(*old_values, prev_dis1, 0x2104);

        next_dis = next_dis + p1_vector;
        prev_dis = prev_dis + p1_vector;
    }
    if(recompute) {
        const int dif = col - dis;
        if(dir_type == DIR_LEFTRIGHT) {
            if(lane == 0) {
                // lane = 0 is dis = 0, no need to subtract dis
                *rp0 = d_transform1[index_im];
            }
        } else if(dir_type == DIR_RIGHTLEFT) {
            // First iteration, load D pixels
            if(first_iteration) {
                const uint4 right = reinterpret_cast<const uint4*>(&d_transform1[index_im-dis-3])[0];
                *rp3 = right.x;
                *rp2 = right.y;
                *rp1 = right.z;
                *rp0 = right.w;
            } else if(lane == 31 && dif >= 3) {
                *rp3 = d_transform1[index_im-dis-3];
            }
        } else {
    /*
            __shared__ T right_p[MAX_DISPARITY+32];
            const int warp_id = threadIdx.x / WARP_SIZE;
            if(warp_id < 5) {
                const int block_imindex = index_im - warp_id + 32;
                const int rp_index = warp_id*WARP_SIZE+lane;
                const int col_cpy = col-warp_id+32;
                right_p[rp_index] = ((col_cpy-(159-rp_index)) >= 0) ? ld_gbl_cs(&d_transform1[block_imindex-(159-rp_index)]) : 0;
            }*/

            __shared__ T right_p[128+32];
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int block_imindex = index_im - warp_id + 2;
            const int rp_index = warp_id*WARP_SIZE+lane;
            const int col_cpy = col-warp_id+2;
            right_p[rp_index] = ((col_cpy-(129-rp_index)) >= 0) ? d_transform1[block_imindex-(129-rp_index)] : 0;
            right_p[rp_index+64] = ((col_cpy-(129-rp_index-64)) >= 0) ? d_transform1[block_imindex-(129-rp_index-64)] : 0;
            //right_p[rp_index+128] = ld_gbl_cs(&d_transform1[block_imindex-(129-rp_index-128)]);
            if(warp_id == 0) {
                right_p[128+lane] = ld_gbl_cs(&d_transform1[block_imindex-(129-lane)]);
            }
            __syncthreads();

            const int px = MAX_DISPARITY+warp_id-dis-1;
            *rp0 = right_p[px];
            *rp1 = right_p[px-1];
            *rp2 = right_p[px-2];
            *rp3 = right_p[px-3];
        }
        const T left_pixel = d_transform0[index_im];
        *old_value1 = popcount(left_pixel ^ *rp0);
        *old_value2 = popcount(left_pixel ^ *rp1);
        *old_value3 = popcount(left_pixel ^ *rp2);
        *old_value4 = popcount(left_pixel ^ *rp3);
        if(iter_type == ITER_COPY) {
            *old_values = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
        } else {
            costs = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
        }
        // Prepare for next iteration
        if(dir_type == DIR_LEFTRIGHT) {
            *rp3 = shfl_up_32(*rp3, 1);
        } else if(dir_type == DIR_RIGHTLEFT) {
            *rp0 = shfl_down_32(*rp0, 1);
        }
    } else {
        if(iter_type == ITER_COPY) {
            *old_values = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
        } else {
            costs = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
        }
    }

    if(iter_type == ITER_NORMAL) {
        const uint32_t min1 = __vminu4(*old_values, prev_dis);
        const uint32_t min2 = __vminu4(next_dis, *min_cost_p2);
        const uint32_t min_prev = __vminu4(min1, min2);
        *old_values = costs + (min_prev - *min_cost);
    }
    if(iter_type == ITER_NORMAL || !recompute) {
        uint32_to_uchars(*old_values, old_value1, old_value2, old_value3, old_value4);
    }

    if(join_dispcomputation) {
        const uint32_t L0_costs = *((uint32_t*) (d_L0+index));
        const uint32_t L1_costs = *((uint32_t*) (d_L1+index));
        const uint32_t L2_costs = *((uint32_t*) (d_L2+index));
        #if PATH_AGGREGATION == 8
            const uint32_t L3_costs = *((uint32_t*) (d_L3+index));
            const uint32_t L4_costs = *((uint32_t*) (d_L4+index));
            const uint32_t L5_costs = *((uint32_t*) (d_L5+index));
            const uint32_t L6_costs = *((uint32_t*) (d_L6+index));
        #endif

        int l0_x, l0_y, l0_z, l0_w;
        int l1_x, l1_y, l1_z, l1_w;
        int l2_x, l2_y, l2_z, l2_w;
        #if PATH_AGGREGATION == 8
            int l3_x, l3_y, l3_z, l3_w;
            int l4_x, l4_y, l4_z, l4_w;
            int l5_x, l5_y, l5_z, l5_w;
            int l6_x, l6_y, l6_z, l6_w;
        #endif

        uint32_to_uchars(L0_costs, &l0_x, &l0_y, &l0_z, &l0_w);
        uint32_to_uchars(L1_costs, &l1_x, &l1_y, &l1_z, &l1_w);
        uint32_to_uchars(L2_costs, &l2_x, &l2_y, &l2_z, &l2_w);
        #if PATH_AGGREGATION == 8
            uint32_to_uchars(L3_costs, &l3_x, &l3_y, &l3_z, &l3_w);
            uint32_to_uchars(L4_costs, &l4_x, &l4_y, &l4_z, &l4_w);
            uint32_to_uchars(L5_costs, &l5_x, &l5_y, &l5_z, &l5_w);
            uint32_to_uchars(L6_costs, &l6_x, &l6_y, &l6_z, &l6_w);
        #endif

        #if PATH_AGGREGATION == 8
            const uint16_t val1 = l0_x + l1_x + l2_x + l3_x + l4_x + l5_x + l6_x + *old_value1;
            const uint16_t val2 = l0_y + l1_y + l2_y + l3_y + l4_y + l5_y + l6_y + *old_value2;
            const uint16_t val3 = l0_z + l1_z + l2_z + l3_z + l4_z + l5_z + l6_z + *old_value3;
            const uint16_t val4 = l0_w + l1_w + l2_w + l3_w + l4_w + l5_w + l6_w + *old_value4;
        #else
            const uint16_t val1 = l0_x + l1_x + l2_x + *old_value1;
            const uint16_t val2 = l0_y + l1_y + l2_y + *old_value2;
            const uint16_t val3 = l0_z + l1_z + l2_z + *old_value3;
            const uint16_t val4 = l0_w + l1_w + l2_w + *old_value4;
        #endif
        int min_idx1 = dis;
        uint16_t min1 = val1;
        if(val1 > val2) {
            min1 = val2;
            min_idx1 = dis+1;
        }

        int min_idx2 = dis+2;
        uint16_t min2 = val3;
        if(val3 > val4) {
            min2 = val4;
            min_idx2 = dis+3;
        }

        uint16_t minval = min1;
        int min_idx = min_idx1;
        if(min1 > min2) {
            minval = min2;
            min_idx = min_idx2;
        }

        const int min_warpindex = warpReduceMinIndex(minval, min_idx);
        if(lane == 0) {
            d_disparity[index_im] = min_warpindex;
        }
    } else {
        st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index]), *old_values);
    }
    if(min_type == MIN_COMPUTE) {
        int min_cost_scalar = min(min(*old_value1, *old_value2), min(*old_value3, *old_value4));
        *min_cost = uchar_to_uint32(warpReduceMin(min_cost_scalar));
        *min_cost_p2 = *min_cost + p2_vector;
    }
}





template<class T, int add_col, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGeneric(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int cols, int add_index, const T *_d_transform0, const T *_d_transform1, const int add_imindex, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int dis = 4*lane;
    int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
    int col, index_im;
    if(recompute || join_dispcomputation) {
        if(recompute) {
            col = initial_col;
        }
        index_im = initial_row*cols+initial_col;
    }

    const int MAX_PAD = UCHAR_MAX-P1;
    const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
    const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);
    int old_value1;
    int old_value2;
    int old_value3;
    int old_value4;
    uint32_t min_cost, min_cost_p2, old_values;
    T rp0, rp1, rp2, rp3;

    if(recompute) {
        if(dir_type == DIR_LEFTRIGHT) {
            CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            for(int i = 4; i < max_iter-3; i+=4) {
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            }
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        } else if(dir_type == DIR_RIGHTLEFT) {
            CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            for(int i = 4; i < max_iter-3; i+=4) {
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            }
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        } else {
            CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            for(int i = 1; i < max_iter; i++) {
                CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
                CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
            }
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        }
    } else {
        CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

        for(int i = 1; i < max_iter; i++) {
            CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        }
        CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
        CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}







template<class T>

__global__ void CostAggregationKernelLeftToRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_row < rows) {
        const int initial_col = 0;
        const int add_index = MAX_DISPARITY;
        const int add_imindex = 1;
        const int max_iter = cols-1;
        const int add_col = 1;
        const bool recompute = true;
        const bool join_dispcomputation = false;

        CostAggregationGeneric<T, add_col, DIR_LEFTRIGHT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>

__global__ void CostAggregationKernelRightToLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_row < rows) {
        const int initial_col = cols-1;
        const int add_index = -MAX_DISPARITY;
        const int add_imindex = -1;
        const int max_iter = cols-1;
        const int add_col = -1;
        const bool recompute = true;
        const bool join_dispcomputation = false;

        CostAggregationGeneric<T, add_col, DIR_RIGHTLEFT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}





template<class T>
__global__ void CostAggregationKernelDownToUp_8(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = rows-1;
        const int add_index = -cols*MAX_DISPARITY;
        const int add_imindex = -cols;
        const int max_iter = rows-1;
        const int add_col = 0;
        const bool recompute = false;
        //const bool join_dispcomputation = PATH_AGGREGATION == 4;
        const bool join_dispcomputation = false;

        CostAggregationGeneric<T, add_col, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}





template<class T>
__global__ void CostAggregationKernelDownToUp_4(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = rows-1;
        const int add_index = -cols*MAX_DISPARITY;
        const int add_imindex = -cols;
        const int max_iter = rows-1;
        const int add_col = 0;
        const bool recompute = false;
        //const bool join_dispcomputation = PATH_AGGREGATION == 4;
        const bool join_dispcomputation = true;

        CostAggregationGeneric<T, add_col, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}

template<class T>
//__launch_bounds__(64, 16)
__global__ void CostAggregationKernelUpToDown(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = cols*MAX_DISPARITY;
        const int add_imindex = cols;
        const int max_iter = rows-1;
        const int add_col = 0;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        CostAggregationGeneric<T, add_col, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}





template<int add_index, class T, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationDiagonalGeneric(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int col_nomin, const int col_copycost, const int cols, const T *_d_transform0, const T *_d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int dis = 4*lane;
    int col = initial_col;
    int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
    int index_im;
    if(recompute || join_dispcomputation) {
        index_im = initial_row*cols+col;
    }
    const int MAX_PAD = UCHAR_MAX-P1;
    const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
    const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);
    int old_value1;
    int old_value2;
    int old_value3;
    int old_value4;
    uint32_t min_cost, min_cost_p2, old_values;
    T rp0, rp1, rp2, rp3;

    CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    for(int i = 1; i < max_iter; i++) {
        CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, initial_row, i, dis);
        if(col == col_copycost) {
            CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        } else {
            CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        }
    }

    CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, max_iter, initial_row, dis);
    if(col == col_copycost) {
        CostAggregationGenericIteration<T, ITER_COPY, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    } else {
        CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}






template<class T>

__global__ void CostAggregationKernelDiagonalDownUpRightLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
    if(initial_col < cols) {
        const int initial_row = rows-1;
        const int add_index = -1;
        const int col_nomin = 0;
        const int col_copycost = cols-1;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}







template<class T>
__global__ void CostAggregationKernelDiagonalDownUpLeftRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
    if(initial_col >= 0) {
        const int initial_row = rows-1;
        const int add_index = 1;
        const int col_nomin = cols-1;
        const int col_copycost = 0;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}



template<class T>
__global__ void CostAggregationKernelDiagonalUpDownRightLeft_8(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = -1;
        const int col_nomin = 0;
        const int col_copycost = cols-1;
        const int max_iter = rows-1;
        const bool recompute = false;
        //const bool join_dispcomputation = PATH_AGGREGATION == 8;
        const bool join_dispcomputation = true;

        CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}






template<class T>
__global__ void CostAggregationKernelDiagonalUpDownRightLeft_4(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = -1;
        const int col_nomin = 0;
        const int col_copycost = cols-1;
        const int max_iter = rows-1;
        const bool recompute = false;
        //const bool join_dispcomputation = PATH_AGGREGATION == 8;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}






template<class T>

__global__ void CostAggregationKernelDiagonalUpDownLeftRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6) {
    const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if(initial_col < cols) {
        const int initial_row = 0;
        const int add_index = 1;
        const int col_nomin = cols-1;
        const int col_copycost = 0;
        const int max_iter = rows-1;
        const bool recompute = false;
        const bool join_dispcomputation = false;

        CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
}





static void free_memory() {
	CUDA_CHECK_RETURN(cudaFree(d_im0));
	CUDA_CHECK_RETURN(cudaFree(d_im1));
	CUDA_CHECK_RETURN(cudaFree(d_transform0));
	CUDA_CHECK_RETURN(cudaFree(d_transform1));
	CUDA_CHECK_RETURN(cudaFree(d_L0));
	CUDA_CHECK_RETURN(cudaFree(d_L1));
	CUDA_CHECK_RETURN(cudaFree(d_L2));
	CUDA_CHECK_RETURN(cudaFree(d_L3));
#if PATH_AGGREGATION == 8
	CUDA_CHECK_RETURN(cudaFree(d_L4));
	CUDA_CHECK_RETURN(cudaFree(d_L5));
	CUDA_CHECK_RETURN(cudaFree(d_L6));
	CUDA_CHECK_RETURN(cudaFree(d_L7));
#endif
	CUDA_CHECK_RETURN(cudaFree(d_disparity));
	CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
	CUDA_CHECK_RETURN(cudaFree(d_cost));

	delete[] h_disparity;
}

#ifdef OPENCV


cv::Ptr<cv::ximgproc::DisparityWLSFilter> initDisparityWLSFilter(int wsize, int max_disp, double lambda, double sigma) 
{

    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, max_disp, wsize);
    
    left_matcher->setP1(24 * wsize * wsize);
    left_matcher->setP2(96 * wsize * wsize);
    left_matcher->setPreFilterCap(63);
    left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);


    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);

    return wls_filter;
}

void post_disparity_filter(cv::Mat& filtered_disp, cv::Mat& left_disp, cv::Mat& right_disp, cv::Mat& im_l, cv::Ptr<cv::ximgproc::DisparityWLSFilter>& wls_filter) 
{
    //Mat filtered_disp;
    cv::Mat conf_map = cv::Mat(im_l.rows, im_l.cols, CV_8U);    conf_map = cv::Scalar(255);
    //cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);

    ////filter
    //wls_filter->setLambda(lambda);
    //wls_filter->setSigmaColor(sigma);
    wls_filter->filter(left_disp, im_l, filtered_disp, right_disp);
    conf_map = wls_filter->getConfidenceMap();
    //Rect ROI = wls_filter->getROI();

    //visualization
    //getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);



    return;
}

double apply_polynomial_calibration(double x, std::vector<double>& coef)
{
    int iO, order = coef.size() - 1;
    double sum = 0.0, xx = 1.0;  
    for(iO = 0; iO < order + 1; iO++)
    {
        sum += coef[iO] * xx;
        xx *= x;
    }
    return sum;
} 


double disparity_2_millimeter(int disp, double focal_in_pixel, double baseline_in_millimeter, struct_poly* param_poly) 
{
    double millimeter = 0;
    if(disp)
    {
        double millimeter_ori = focal_in_pixel * baseline_in_millimeter / (double)disp;
        if(param_poly)
        {
            if(param_poly->is_from_disparity_to_millimeter)
            {
                millimeter = apply_polynomial_calibration((double)disp, param_poly->coef_disparity);
            }
            else
            {
                millimeter = apply_polynomial_calibration(millimeter_ori, param_poly->coef_millimeter);
            }
        }
        else
        {
            millimeter = millimeter_ori;
        }
    }
    return millimeter;
}


//IplImage *disparity_2_meter(IplImage *ipl_dis) {
cv::Mat disparity_2_millimeter(cv::Mat& mat_dis_uc, double focal_in_pixel, double baseline_in_millimeter) 
//cv::Mat disparity_2_millimeter(cv::Mat& mat_dis, cv::Mat& mat_focal_mul_baseline_milli) 
{
    
    //double bunmo_center = mat_focal_mul_baseline_milli.at<double>(mat_dis.rows / 2, mat_dis.cols / 2);
    //std::cout << "bunmo_center : " << bunmo_center << std::endl;
    //std::cout << "mat_dis.type() : " << mat_dis.type() << std::endl;
    //int disp_center = mat_dis.at<unsigned char>(mat_dis.rows / 2, mat_dis.cols / 2);
    //std::cout << "disp_center : " << disp_center << std::endl;
    //double disp_min, disp_max;  cv::minMaxLoc(mat_dis, &disp_min, &disp_max);    
    //std::cout << "disp_min : " << disp_min << "\tdisp_max : " << disp_max << std::endl;
    //double bunmo_min, bunmo_max;  cv::minMaxLoc(mat_focal_mul_baseline_milli, &bunmo_min, &bunmo_max);    
    //std::cout << "bunmo_min : " << bunmo_min << "\tbunmo_max : " << bunmo_max << std::endl;

    //cv::Mat mat_depth_millimeter = (focal_in_pixel * baseline_in_millimeter) / mat_dis;
    //cv::Mat mat_depth_millimeter = mat_focal_mul_baseline_milli / mat_dis;
    //cv::Mat mat_depth_millimeter;   cv::divide(mat_focal_mul_baseline_milli, mat_dis, mat_depth_millimeter);
    double bunmo = focal_in_pixel * baseline_in_millimeter;
    cv::Mat mat_depth_millimeter, mat_dis_double;   
    mat_dis_uc.convertTo(mat_dis_double, CV_64F); 
    cv::divide(bunmo, mat_dis_double, mat_depth_millimeter);
    //double millimeter_center_computed = mat_depth_millimeter.at<double>(mat_depth_millimeter.rows / 2, mat_depth_millimeter.cols / 2);
    //std::cout << "millimeter_center_computed : " << cvRound(millimeter_center_computed) << std::endl;
    //double millimeter_center_expected = bunmo_center / disp_center;
    //std::cout << "millimeter_center_expected : " << millimeter_center_expected << std::endl;
    return mat_depth_millimeter;
}



//cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, const char* directory, const char* fname) {
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, int path_agg) {
//IplImage *compute_disparity_method(IplImage *left, IplImage *right, float *elapsed_time_ms) {
    //printf("disparity aaa\n");
	if(cols != left.cols || rows != left.rows) {
    //printf("cols or rows are different\n");
	//if(cols != left->width || rows != left->height) {
		debug_log("WARNING: cols or rows are different");
		if(!first_alloc) {
			debug_log("Freeing memory");
			free_memory();
		}
    first_alloc = false;
		cols = left.cols;
		//cols = left->width;
		rows = left.rows;
		//rows = left->height;
		size = rows * cols;
		size_cube_l = size * MAX_DISPARITY;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));

		int size_cube = size*MAX_DISPARITY;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));
        //printf("bbb\n");

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_S, sizeof(uint16_t)*size_cube_l));
    //printf("disparity bbb\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
    //printf("disparity ccc\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
    //printf("disparity ddd\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
    //printf("disparity eee\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));
    //printf("disparity fff\n");
//#if PATH_AGGREGATION == 8
    if (8 == path_agg) 
    {
		//printf("PATH_AGGREGATION = 8\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp aaa\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp bbb\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp ccc\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp ddd\n");
    }
    //printf("disparity ggg\n");
//#endif

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
		h_disparity = new uint8_t[size];
	}
    //printf("disparity iii\n");
	debug_log("Copying images to the GPU");
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	//CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left->imageData, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	//CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right->imageData, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

        //printf("ccc\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

        //printf("ddd\n");
	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

        //printf("eee\n");
	debug_log("Calling CSCT");
	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);

        //printf("fff\n");
	cudaError_t err = cudaGetLastError();

        //printf("ggg\n");
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

        //printf("hhh\n");
	// Hamming distance
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	debug_log("Calling Hamming Distance");
	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

        //printf("iii\n");
	// Cost Aggregation
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;
/*
    printf("p1 : %d\tp2 : %d\n", p1, p2); 
    printf("p1 : %d\tp2 : %d\n", p1, p2); 
    printf("p1 : %d\tp2 : %d\n", p1, p2); 
    printf("p1 : %d\tp2 : %d\n", p1, p2); 
    printf("p1 : %d\tp2 : %d\n", p1, p2); 
    printf("p1 : %d\tp2 : %d\n", p1, p2); 
*/
    debug_log("Calling Left to Right");
    CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    debug_log("Calling Right to Left");
    CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    debug_log("Calling Up to Down");
    CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    debug_log("Calling Down to Up");
    if (8 == path_agg)
    {
        CostAggregationKernelDownToUp_8<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
    else
    {        
        CostAggregationKernelDownToUp_4<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

    }
//#if PATH_AGGREGATION == 8
    if (8 == path_agg)
    {
        CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

        CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
        CostAggregationKernelDiagonalUpDownRightLeft_8<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    }
//#endif
    debug_log("Calling Median Filter");
    MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);

    cudaEventRecord(stop, 0);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaEventElapsedTime(elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    debug_log("Copying final disparity to CPU");
    CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    cv::Mat disparity(rows, cols, CV_8UC1, h_disparity);
    //IplImage* disparity = cvCreateImageHeader( cvSize(cols, rows), IPL_DEPTH_8U, 1);    cvSetData(disparity, h_disparity, cols);
    return disparity;
}






/*

//cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, const char* directory, const char* fname) {
cv::Mat compute_disparity_method_4(cv::Mat left, cv::Mat right, float *elapsed_time_ms) {
//IplImage *compute_disparity_method(IplImage *left, IplImage *right, float *elapsed_time_ms) {
	if(cols != left.cols || rows != left.rows) {
	//if(cols != left->width || rows != left->height) {
		debug_log("WARNING: cols or rows are different");
		if(!first_alloc) {
			debug_log("Freeing memory");
			free_memory();
		}
    first_alloc = false;
		cols = left.cols;
		//cols = left->width;
		rows = left.rows;
		//rows = left->height;
		size = rows * cols;
		size_cube_l = size * MAX_DISPARITY;
        //printf("aaa\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));

		int size_cube = size*MAX_DISPARITY;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));
        //printf("bbb\n");

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_S, sizeof(uint16_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));
//#if PATH_AGGREGATION == 8
#if 0
		//printf("PATH_AGGREGATION = 8\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp aaa\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp bbb\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp ccc\n");
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
		//printf("comp_disp ddd\n");
#endif

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
		h_disparity = new uint8_t[size];
	}
	debug_log("Copying images to the GPU");
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	//CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left->imageData, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	//CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right->imageData, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

        //printf("ccc\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

        //printf("ddd\n");
	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

        //printf("eee\n");
	debug_log("Calling CSCT");
	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);

        //printf("fff\n");
	cudaError_t err = cudaGetLastError();

        //printf("ggg\n");
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

        //printf("hhh\n");
	// Hamming distance
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	debug_log("Calling Hamming Distance");
	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

        //printf("iii\n");
	// Cost Aggregation
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;



    debug_log("Calling Left to Right");
    CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    debug_log("Calling Right to Left");
    CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    debug_log("Calling Up to Down");
    CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    debug_log("Calling Down to Up");
    CostAggregationKernelDownToUp_4<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

//#if PATH_AGGREGATION == 8
#if 0
    CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

    CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
    CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
#endif
    debug_log("Calling Median Filter");
    MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);

    cudaEventRecord(stop, 0);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaEventElapsedTime(elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    debug_log("Copying final disparity to CPU");
    CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    cv::Mat disparity(rows, cols, CV_8UC1, h_disparity);
    //IplImage* disparity = cvCreateImageHeader( cvSize(cols, rows), IPL_DEPTH_8U, 1);    cvSetData(disparity, h_disparity, cols);
    return disparity;
}
*/





#endif






void finish_disparity_method() {
	if(!first_alloc) {
		free_memory();
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
	}
}






#include <math.h>

// SHUFFLE

unsigned shufl_up_sync_shared_var_arr[32];
bool shufl_up_sync_updated[32] = {0};

__device__ unsigned shufl_up_sync(unsigned mask, unsigned var, unsigned int delta, int width=warpSize) {
    int tid = threadIdx.x;

    shufl_up_sync_shared_var_arr[tid] = var;
    shufl_up_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width); // subsection
        int sub_tid = tid % width;
        int srcLane = sub_tid - delta;
        if (srcLane >= 0) {
            while(!shufl_up_sync_updated[srcLane + (width * sub_id)]);
            var = shufl_up_sync_shared_var_arr[srcLane + (width * sub_id)];
        }
    }
    return var;
}

unsigned shufl_down_sync_shared_var_arr[32];
bool shufl_down_sync_updated[32] = {0};

__device__ unsigned shufl_down_sync(unsigned mask, unsigned var, unsigned int delta, int width=warpSize) {
    int tid = threadIdx.x;

    shufl_down_sync_shared_var_arr[tid] = var;
    shufl_down_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width); // subsection
        int sub_tid = tid % width;
        int srcLane = sub_tid + delta;
        if (srcLane < width) {
            while(!shufl_down_sync_updated[srcLane + (width * sub_id)]);
            var = shufl_down_sync_shared_var_arr[srcLane + (width * sub_id)];
        }
    }
    return var;
}

unsigned shufl_sync_shared_var_arr[32];
bool shufl_sync_updated[32] = {0};

__device__ unsigned shufl_sync(unsigned mask, unsigned var, int srcLane, int width=warpSize) {
    int tid = threadIdx.x;

    shufl_sync_shared_var_arr[tid] = var;
    shufl_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        if (srcLane >= 0) {
            while(!shufl_sync_updated[srcLane]);
            var = shufl_sync_shared_var_arr[srcLane];
        }
    }
    return var;
}

unsigned shufl_xor_sync_shared_var_arr[32];
bool shufl_xor_sync_updated[32] = {0};

__device__ unsigned shufl_xor_sync(unsigned mask, unsigned var, int laneMask, int width=warpSize) {
    int tid = threadIdx.x;

    shufl_xor_sync_shared_var_arr[tid] = var;
    shufl_xor_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width);
        int sub_tid = tid % width;
        int srcLane = sub_tid ^ laneMask;
        int src_sub_id = floor((float) srcLane / (float) width);
        if (src_sub_id <= sub_id) {
            while(!shufl_xor_sync_updated[srcLane + (width * src_sub_id)]);
            var = shufl_xor_sync_shared_var_arr[srcLane + (width * src_sub_id)];
        }
    }
    return var;
}
// #include <time.h>
// to randomly create an array
// srand(time(0));   
// unsigned correct_sum = 0;
// for (int i = 0; i < 32; i++) {
//     test_array[i] = rand() % 1000;
//     correct_sum += test_array[i];
// }

// REDUCE


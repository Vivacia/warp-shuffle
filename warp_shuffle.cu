#include <cstdio>
#include <cstdlib>
#include <math.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


/**
 * All of the threads exchange data simultaneously.
 * Mask used to determine which threads participate, usually -1.
 * Var is the data it is operating on/temporary store?
 * For xor, lane_mask ^ current_lane_ID/lane_index determines which lane to switch with.
 * Two threads switch their data based on the above. In shuffle up and down, delta = shift.
 * Indexing starts at 0 regardless of width (usually warp size).
 * var = variable to get the value from, which is different for each thread.
*/

/*
 * T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
 * returns the value of var held by the thread whose ID is given by srcLane.
 */
__global__ void shfl_sync(int* shared_var_arr, size_t* width,
    unsigned* mask, int* srcLane, size_t* warp_size) {
    int tid = threadIdx.x;
    if (((0x1 << tid) & *mask) != 0) {
        // If srcLane is outside the range [0:width-1],
        // the value returned corresponds to the value of var
        // held by the srcLane modulo width (i.e. within the same subsection)
        *srcLane = (floor((float)tid / (float)*width) * *width) + *srcLane % (*width-1);
        if (*srcLane > *warp_size)
            return;
        shared_var_arr[tid] = shared_var_arr[*srcLane];
    }
}

void shfl_sync_setup(int* _shared_var_arr, size_t _width,
    unsigned _mask, int _srcLane, size_t _warp_size) {
    int* shared_var_arr;
    size_t* width;
    unsigned* mask;
    int* srcLane;
    size_t* warp_size;

    checkCudaErrors(cudaMalloc((void**) &width,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(width, &_width,
        sizeof(size_t), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &shared_var_arr,
        *width * sizeof(int)));
    checkCudaErrors(cudaMemcpy(shared_var_arr, &_shared_var_arr,
        *width * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &mask, sizeof(unsigned)));
    checkCudaErrors(cudaMemcpy(mask, &_mask,
        sizeof(unsigned), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &srcLane, sizeof(int)));
    checkCudaErrors(cudaMemcpy(srcLane, &_srcLane,
        sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &warp_size,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(warp_size, &_warp_size,
        sizeof(size_t), cudaMemcpyHostToDevice));
     
    shfl_sync<<<1, _width>>>(shared_var_arr, width, mask, srcLane, warp_size);
}

/**
 * __shfl_up_sync() calculates a source lane ID by subtracting delta from the caller's lane ID.
 * The value of var held by the resulting lane ID is returned:
 * in effect, var is shifted up the warp by delta lanes.
 * If width is less than warpSize then each subsection of the warp behaves as a
 * separate entity with a starting logical lane ID of 0.
 * The source lane index will not wrap around the value of width,
 * so effectively the lower delta lanes will be unchanged.
 * T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=warpSize)
 */
__global__ void shfl_up_sync(int* shared_var_arr, size_t* width,
    unsigned* mask, unsigned* delta, size_t* warp_size) {
    int tid = threadIdx.x;
    if (((0x1 << tid) & *mask) != 0) {
        int sub_id = floor((float)tid / (float)*width); // subsection
        int sub_tid = tid % *width;
        int srcLane = sub_tid - *delta;
        if (srcLane < 0)
            return;
        shared_var_arr[tid] = shared_var_arr[srcLane + (*width * sub_id)];
    }
}

void shfl_up_sync_setup(int* _shared_var_arr, size_t _width,
    unsigned _mask, unsigned _delta, size_t _warp_size) {
    int* shared_var_arr;
    size_t* width;
    unsigned* mask;
    unsigned* delta;
    size_t* warp_size;

    checkCudaErrors(cudaMalloc((void**) &width,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(width, &_width,
        sizeof(size_t), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &shared_var_arr,
        *width * sizeof(int)));
    checkCudaErrors(cudaMemcpy(shared_var_arr, &_shared_var_arr,
        *width * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &mask, sizeof(unsigned)));
    checkCudaErrors(cudaMemcpy(mask, &_mask,
        sizeof(unsigned), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &delta, sizeof(int)));
    checkCudaErrors(cudaMemcpy(delta, &_delta,
        sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &warp_size,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(warp_size, &_warp_size,
        sizeof(size_t), cudaMemcpyHostToDevice));
     
    shfl_up_sync<<<1, _width>>>(shared_var_arr, width, mask, delta, warp_size);
}

/**
 * __shfl_down_sync() calculates a source lane ID by adding delta to the caller's lane ID.
 * The value of var held by the resulting lane ID is returned:
 * this has the effect of shifting var down the warp by delta lanes.
 * If width is less than warpSize then each subsection of the warp behaves as a
 * separate entity with a starting logical lane ID of 0. As for __shfl_up_sync(),
 * the ID number of the source lane will not wrap around the value of width and
 * so the upper delta lanes will remain unchanged.
 * T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=warpSize)
 */
__global__ void shfl_down_sync(int* shared_var_arr, size_t* width,
    unsigned* mask, unsigned* delta, size_t* warp_size) {
    int tid = threadIdx.x;
    if (((0x1 << tid) & *mask) != 0) {
        int sub_id = floor((float)tid / (float)*width); // subsection
        int sub_tid = tid % *width;
        int srcLane = sub_tid + *delta;
        if (srcLane >= *width)
            return;
        shared_var_arr[tid] = shared_var_arr[srcLane + (*width * sub_id)];
    }
}

void shfl_down_sync_setup(int* _shared_var_arr, size_t _width,
    unsigned _mask, unsigned _delta, size_t _warp_size) {
    int* shared_var_arr;
    size_t* width;
    unsigned* mask;
    unsigned* delta;
    size_t* warp_size;

    checkCudaErrors(cudaMalloc((void**) &width,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(width, &_width,
        sizeof(size_t), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &shared_var_arr,
        *width * sizeof(int)));
    checkCudaErrors(cudaMemcpy(shared_var_arr, &_shared_var_arr,
        *width * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &mask, sizeof(unsigned)));
    checkCudaErrors(cudaMemcpy(mask, &_mask,
        sizeof(unsigned), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &delta, sizeof(int)));
    checkCudaErrors(cudaMemcpy(delta, &_delta,
        sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &warp_size,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(warp_size, &_warp_size,
        sizeof(size_t), cudaMemcpyHostToDevice));
    
    /*
    #define FULL_MASK 0xffffffff
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    */
    for (*delta = 16; *delta > 0; *delta /= 2) {
        shfl_down_sync<<<1, _width>>>(shared_var_arr, width, mask, delta, warp_size);
    }
}

/**
 * __shfl_xor_sync() calculates a source line ID by performing a bitwise XOR of the
 * caller's lane ID with laneMask: the value of var held by the resulting lane ID is returned.
 * If width is less than warpSize then each group of width consecutive threads are
 * able to access elements from earlier groups of threads, however if they attempt
 * to access elements from later groups of threads their own value of var will be returned.
 * This mode implements a butterfly addressing pattern such as is used in tree reduction and broadcast.
 * T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize) 
 */
__global__ void shfl_xor_sync(int* shared_var_arr, size_t* width,
    unsigned* mask, int* laneMask, size_t* warp_size) {
    int tid = threadIdx.x;
    if (((0x1 << tid) & *mask) != 0) {
        int sub_id = floor((float)tid / (float)*width); // subsection
        int sub_tid = tid % *width;
        int srcLane = sub_tid ^ *laneMask;
        int src_sub_id = floor((float)srcLane / (float)*width);
        if (src_sub_id > sub_id)
            return;
        shared_var_arr[tid] = shared_var_arr[srcLane + (*width * src_sub_id)];
    }
}

void shfl_xor_sync_setup(int* _shared_var_arr, size_t _width,
    unsigned _mask, int _laneMask, size_t _warp_size) {
    int* shared_var_arr;
    size_t* width;
    unsigned* mask;
    int* laneMask;
    size_t* warp_size;

    checkCudaErrors(cudaMalloc((void**) &width,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(width, &_width,
        sizeof(size_t), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &shared_var_arr,
        *width * sizeof(int)));
    checkCudaErrors(cudaMemcpy(shared_var_arr, &_shared_var_arr,
        *width * sizeof(int), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**) &mask, sizeof(unsigned)));
    checkCudaErrors(cudaMemcpy(mask, &_mask,
        sizeof(unsigned), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &laneMask, sizeof(int)));
    checkCudaErrors(cudaMemcpy(laneMask, &_laneMask,
        sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &warp_size,
        sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(warp_size, &_warp_size,
        sizeof(size_t), cudaMemcpyHostToDevice));
     
    shfl_sync<<<1, _width>>>(shared_var_arr, width, mask, laneMask, warp_size);
}

int main(int argc, char *argv[]) {
    int *_shared_var_arr;
    size_t warp_size = 32;
    size_t width = 32;
    unsigned mask = 0xffffffff;
    int delta = 0;

    checkCudaErrors(cudaMalloc((void**) &_shared_var_arr,
        width * sizeof(int)));
    for (int i = 0; i < warp_size; i++) {
        _shared_var_arr[i] = 1;
    }
    
    shfl_down_sync_setup(_shared_var_arr, width, mask, delta, warp_size);
}
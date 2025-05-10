#include "linear.h"
#include "../utils/utils.h"


// A "linear layer" neural network performs a linear transformation on its input: Y = xw^t + b
// this is a neural network that performs a linear transformation on its input
/*
 *X: input matrix (batch_size, input_features)
 *W: weights matrix (output_features, input_features)
 *b: Bias vector (output_features)
 *Y: Output matrix (batch_size, output_features)
 *
 *The GPU implementation aims to parallelize these matrix operations operations
 */

__global__
void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out){ // Chain together the series of modules passed to the constructor
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        out[ind_out] = bias[col];

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            out[ind_out] += inp[ind_inp]*weights[ind_weights];
        }
    }
}


__global__
void linear_backward_gpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&inp[ind_inp], weights[ind_weights]*out[ind_out]);
        }
    }
}


__global__
void linear_update_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        atomicAdd(&bias[col], -lr*out[ind_out]);

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&weights[ind_weights], -lr*inp[ind_inp]*out[ind_out]);
        }
    }
}


Linear_GPU::Linear_GPU(int _bs, int _n_in, int _n_out, float _lr){
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in*n_out;
    sz_out = bs*n_out;
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;

    cudaMallocManaged(&weights, sz_weights*sizeof(float));
    cudaMallocManaged(&bias, n_out*sizeof(float));

    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);
}


void Linear_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out);
    cudaDeviceSynchronize();
}


void Linear_GPU::backward(){
    init_zero(inp, bs*n_in);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_backward_gpu<<<n_blocks, n_threads>>>(inp, cp_weights, out, bs, n_in, n_out);
    cudaDeviceSynchronize();

    cudaFree(cp_weights);
    cudaFree(out);
}


void Linear_GPU::update(){
    cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
    set_eq(cp_weights, weights, sz_weights);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out, lr);
    cudaDeviceSynchronize();
}

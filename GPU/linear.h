#ifndef LINEAR_GPU_H
#define LINEAR_GPU_H


#include "../utils/module.h" // This has the forward backward and update methods


class Linear_GPU: public Module{
    public:
        float *weights, *cp_weights, *bias, lr;     //Pointser to a float array that stores the weights of the linear layer. These are parameters the network learns
    //cp_weights this a pointer to a float array that stores a copy of the weights. This used during the update step
    //bias a pointer to a float array that stores the bias terms of the linear layer
    //lr the learning rate a scalar that controls the step size when updating the wights and biases. its hyperparameter of the training process
        int bs, n_in, n_out, sz_weights, n_block_rows, n_block_cols; //batch size, number of input features, total size of the weights array, last two are used to configure the cuda kernel launch (grid and bloack dimesions) for parallel computing

        Linear_GPU(int _bs, int _n_in, int _n_out, float _lr = 0.1f);  //Constructor parameters to set the batch size, number of input features, number of output features, and learning rate. The learning rate has a default value of 0.1f
        void forward(float *_inp, float *_out); //this function takes the pointer to the input data (_inp) and a pointer to the output data (_out) and performs the linear transformations
        void backward(); //computers the gradient of the loss with respect to the inputs
        void update(); //updates the weights and biases of the layer using computed gradients and the learning rate
};


#endif

#include "depthwise_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

/*
#########################################

深度可分离卷积层设计

#########################################
*/



int depthwise_convolutional_out_height(depthwise_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int depthwise_convolutional_out_width(depthwise_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}


static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}


#ifdef GPU
#ifdef CUDNN
void cudnn_depthwise_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, l->c, l->size, l->size);

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, l->c, l->size, l->size);
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif
    /*cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);*/
}
#endif
#endif

depthwise_convolutional_layer make_depthwise_convolutional_layer(int batch, int h, int w, int c,int size, int stride, int padding, ACTIVATION activation, int batch_normalize)
{
    int i;
	depthwise_convolutional_layer l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.n = c;
	l.c = c;

    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(l.n*size*size, sizeof(float));
    l.weight_updates = calloc(l.n*size*size, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));

    l.nweights = l.n*size*size;
    l.nbiases = l.n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
   //for(i = 0; i < c*size*size; ++i) l.weights[i] = 0.01*i;
    for(i = 0; i < l.n*l.size*l.size; ++i) l.weights[i] = scale*rand_normal();
    int out_w = depthwise_convolutional_out_width(l);
    int out_h = depthwise_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_depthwise_convolutional_layer;
    l.backward = backward_depthwise_convolutional_layer;
    l.update = update_depthwise_convolutional_layer;


    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));
        for(i = 0; i < c; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }


#ifdef GPU
    l.forward_gpu = forward_depthwise_convolutional_layer_gpu;
    l.backward_gpu = backward_depthwise_convolutional_layer_gpu;
    l.update_gpu = update_depthwise_convolutional_layer_gpu;

    if(gpu_index >= 0){


        l.weights_gpu = cuda_make_array(l.weights, c*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*size*size);

        l.biases_gpu = cuda_make_array(l.biases, c);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*c);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);



        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, c);
            l.variance_gpu = cuda_make_array(l.variance,c);

            l.rolling_mean_gpu = cuda_make_array(l.mean, c);
            l.rolling_variance_gpu = cuda_make_array(l.variance, c);

            l.mean_delta_gpu = cuda_make_array(l.mean, c);
            l.variance_delta_gpu = cuda_make_array(l.variance, c);

            l.scales_gpu = cuda_make_array(l.scales, c);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*c);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_depthwise_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "dw conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", c, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);



    return l;
}

void resize_depthwise_convolutional_layer(depthwise_convolutional_layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	int out_w = depthwise_convolutional_out_width(*l);
	int out_h = depthwise_convolutional_out_height(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));
	if (l->batch_normalize) {
		l->x = realloc(l->x, l->batch*l->outputs * sizeof(float));
		l->x_norm = realloc(l->x_norm, l->batch*l->outputs * sizeof(float));
	}

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

	if (l->batch_normalize) {
		cuda_free(l->x_gpu);
		cuda_free(l->x_norm_gpu);

		l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
		l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	}
#ifdef CUDNN
	cudnn_depthwise_convolutional_setup(l);
#endif
#endif
	l->workspace_size = get_workspace_size(*l);
}


void add_bias_depthwise(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias_depthwise(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias_depthwise(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    //int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);



   
    int k = l.size*l.size;
    int n = out_h*out_w;




    for(int b = 0; b < l.batch; ++b){
		for (int c=0;c<l.c;c++)
		{
			float *aoffset = l.weights+c*l.size*l.size;
			float *boffset = net.workspace;
			float *coffset = l.output+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
			float *intput_offset = net.input + c*l.h*l.w+ b*l.c*l.h*l.w;
			im2col_cpu(intput_offset, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);
		
		}
    }

/*
	for (int i = 0; i < l.batch*l.c*l.out_h*l.out_w; i++)
	{
		fprintf(stderr, "%f \t", l.output[i]);
	}
*/






    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

	int m = l.n;
    activate_array(l.output, m*n*l.batch, l.activation);//ï¿œï¿œï¿œîº¯ï¿œï¿œÇ°ï¿œòŽ«µï¿œ
/*
	for (int i = 0; i < l.batch*l.c*l.out_h*l.out_w; i++)
	{
		fprintf(stderr, "%f \t", l.output[i]);
	}*/

}

void backward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    //int i;
    int m = l.n;
    int n = l.size*l.size;
    int k = l.out_w*l.out_h;
    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }


	for (int b = 0; b < l.batch; ++b) {
		for (int c = 0; c<l.c; c++)
		{

			float *aoffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *boffset = net.workspace;
			float *coffset = l.weight_updates + c*l.size*l.size;


			float *im = net.input + c*l.h*l.w + b*l.c*l.h*l.w;


			im2col_cpu(im, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);

			if (net.delta) {
				aoffset = l.weights+ c*l.size*l.size;
				boffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
				coffset = net.workspace;

				gemm(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

				col2im_cpu(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta + c*l.h*l.w + b*l.n*l.h*l.w);
			}


		}
	}


/*
	for (int i = 0; i < l.c*l.size*l.size; i++)
	{
		fprintf(stderr, "weight_updates:%f \t", l.weight_updates[i]);
	}
*/



}

void update_depthwise_convolutional_layer(depthwise_convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}
void denormalize_depthwise_convolutional_layer(depthwise_convolutional_layer l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
		for (j = 0; j < l.size*l.size; ++j) {
			l.weights[i*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

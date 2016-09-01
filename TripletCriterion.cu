#include "common.h"

#define ANCHOR 0
#define POSITIVE 1
#define NEGATIVE 2

__global__ void triplet_dist_kernel(const int n, const float norm,
                                    const int nb_batch, const int length,
                                    float* x, float* y) {

  // Unnecessary for loop...
  // This is EXACTLY equivalent to `if (i < n)` statement
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {

    // Equivalent C for loop follows
    // for (int i = 0, i < nb_batch^2, i++)

    // (y, x) coordinate in dist / y
    int row = i / nb_batch;
    int col = i % nb_batch;

    // Pointers to two rows of input / x
    float *xrow = x + row*length;
    float *xcol = x + col*length;

    // Initialise scalar product summation
    float sum = 0.0f;

    // Compute sum[(component difference) ^ norm]
    for (int j = 0; j < length; j++) {
      if (norm == 1.0f) {
        sum += fabsf(xrow[j] - xcol[j]);
      } else {
        sum += powf(xrow[j] - xcol[j], norm);
      }
    }

    // Compute norm-root of the sum
    if (norm == 1.0f) {
      y[row*nb_batch + col] = sum;
    } else {
      y[row*nb_batch + col] = powf(sum, 1.0/norm);
    }
  }
}

__global__ void triplet_loss_semi_kernel(const int n, const int nb_batch, const int length,
                                         const float alpha, const float* x, const float* d,
                                         const float* l, float* y, float* z) {
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {

    // find anchor embedding
    int idx_a = i;

    // find positive embedding
    int idx_p = i;
    float val_p = 0.0f;
    for (int j = 0; j < nb_batch; j++) {
      if ((l[j] == l[i]) & (val_p < d[i*nb_batch + j])) {
        idx_p = j;
        val_p = d[i*nb_batch + j];
      }
    }

    // find negative embedding
    int idx_n = i;
    float val_n = FLT_MAX;
    for (int j = 0; j < nb_batch; j++) {
      if ((l[j] != l[i]) & (val_p < d[i*nb_batch + j]) & (val_n > d[i*nb_batch + j])) {
        idx_n = j;
        val_n = d[i*nb_batch + j];
      }
    }

    // keep track of embedding indices
    y[i*3 + ANCHOR]   = idx_a;
    y[i*3 + POSITIVE] = idx_p;
    y[i*3 + NEGATIVE] = idx_n;

    // loss = max((a - p)^2 - (a - n)^2 + alpha, 0)
    float sum = 0.0f;
    for (int j = 0; j < length; j++)
      sum += powf(x[idx_a*length + j] - x[idx_p*length + j], 2);
    for (int j = 0; j < length; j++)
      sum -= powf(x[idx_a*length + j] - x[idx_n*length + j], 2);

    z[i] = fmaxf(sum + alpha, 0.0f);
  }
}

__global__ void triplet_loss_semi_allpairs_kernel(const int n, const int nb_batch, const int length,
                                                  const int nb_blocks, const int samples,
                                                  const float alpha, const float* x, const float* d,
                                                  const float* l, float* y, float* z) {
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
    // pick each element from positive diagonal block in distance matrix
    int row = i % (nb_blocks*samples);
    int col = i / (nb_blocks*samples) + (row / samples)*samples;
    int dst = (i / (nb_blocks*samples))*nb_batch + i % (nb_blocks*samples);
    if (col >= row)
      col++;

    // find anchor embedding
    int idx_a = row;

    // find positive embedding
    int idx_p = col;
    float val_p = d[idx_a*nb_batch + idx_p];

    // find negative embedding
    int idx_n = idx_a;
    float val_n = FLT_MAX;
    for (int j = nb_blocks*samples; j < nb_batch; j++) {
      if ((l[j] != l[idx_a]) & (val_p < d[idx_a*nb_batch + j]) & (val_n > d[idx_a*nb_batch + j])) {
        idx_n = j;
        val_n = d[idx_a*nb_batch + j];
      }
    }

    // keep track of embedding indices
    y[dst*3 + ANCHOR]   = idx_a;
    y[dst*3 + POSITIVE] = idx_p;
    y[dst*3 + NEGATIVE] = idx_n;

    // loss = max((a - p)^2 - (a - n)^2 + alpha, 0)
    float sum = 0.0f;
    for (int j = 0; j < length; j++)
      sum += powf(x[idx_a*length + j] - x[idx_p*length + j], 2);
    for (int j = 0; j < length; j++)
      sum -= powf(x[idx_a*length + j] - x[idx_n*length + j], 2);
    z[dst] = fmaxf(sum + alpha, 0.0f);
  }
}

static int triplet_TripletCriterion_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);

  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *label = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float norm = luaT_getfieldchecknumber(L, 1, "norm");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  int samples = luaT_getfieldchecknumber(L, 1, "samples");
  int nb_blocks = luaT_getfieldchecknumber(L, 1, "blocks");
  THCudaTensor *dist = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "dist", "torch.CudaTensor");
  THCudaTensor *emb = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "embeddings", "torch.CudaTensor");
  THCudaTensor *loss = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "loss", "torch.CudaTensor");

  long nb_batch = input->size[0];
  long length   = input->size[1];


  // prepare place holder
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resize2d(state, dist, nb_batch, nb_batch);
  if (samples == 1) {
    THCudaTensor_resize2d(state, emb, nb_batch, 3);
    THCudaTensor_resize1d(state, loss, nb_batch);
  } else {
    THCudaTensor_resize2d(state, emb, nb_batch*(samples-1), 3);
    THCudaTensor_resize1d(state, loss, nb_batch*(samples-1));
    THCudaTensor_zero(state, emb);
    THCudaTensor_zero(state, loss);
  }


  // queue kernel (dist matrix)
  long num_threads = nb_batch*nb_batch;
  triplet_dist_kernel <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
    num_threads, norm, nb_batch, length,
    THCudaTensor_data(state, input),
    THCudaTensor_data(state, dist)
  );


  // queue kernel (find embeddings)
  if (samples > 1) {
    num_threads = nb_blocks*samples*(samples-1);
    triplet_loss_semi_allpairs_kernel <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
      num_threads, nb_batch, length, nb_blocks, samples, alpha,
      THCudaTensor_data(state, input),
      THCudaTensor_data(state, dist),
      THCudaTensor_data(state, label),
      THCudaTensor_data(state, emb),
      THCudaTensor_data(state, loss)
    );
  } else {
    num_threads = nb_batch;
    triplet_loss_semi_kernel <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
      num_threads, nb_batch, length, alpha,
      THCudaTensor_data(state, input),
      THCudaTensor_data(state, dist),
      THCudaTensor_data(state, label),
      THCudaTensor_data(state, emb),
      THCudaTensor_data(state, loss)
    );
  }

  // close
  THCudaTensor_free(state, input);
  return 1;
}

__global__ void triplet_prop_kernel(const int n, const int nb_pairs, const int length,
                                    const float* x, const float* emb, const float *loss, float* y) {
  for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
    int row = i / length;
    int col = i % length;

    if (loss[row] > 0) {
      y[i] = (x[((int)emb[3*row+2])*length + col] - x[((int)emb[3*row+1])*length + col])*2.0/((float)nb_pairs);
    } else {
      y[i] = 0;
    }
  }
}

static int triplet_TripletCriterion_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);

  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *emb = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "embeddings", "torch.CudaTensor");
  THCudaTensor *loss = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "loss", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nb_pairs = loss->size[0];
  long length   = input->size[1];

  THCudaTensor_resize2d(state, gradInput, nb_pairs, length);

  // queue kernel
  long num_threads = nb_pairs*length;
  triplet_prop_kernel <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
    num_threads, nb_pairs, length,
    THCudaTensor_data(state, input),
    THCudaTensor_data(state, emb),
    THCudaTensor_data(state, loss),
    THCudaTensor_data(state, gradInput)
  );

  return 1;
}

static const struct luaL_Reg triplet_TripletCriterion__ [] = {
  {"TripletCriterion_updateOutput", triplet_TripletCriterion_updateOutput},
  {"TripletCriterion_updateGradInput", triplet_TripletCriterion_updateGradInput},
  {NULL, NULL}
};

static void triplet_TripletCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, triplet_TripletCriterion__, "nn");
  lua_pop(L,1);
}

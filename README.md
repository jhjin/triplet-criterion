# Triplet Criterion On-the-fly

A loss function based on the distances between anchor, positive and negative embeddings used in 
"FaceNet: A Unified Embedding for Face Recognition and Clustering" http://arxiv.org/abs/1503.03832 .
The module finds positive and negative embeddings within a current mini-batch on-the-fly,
so it does not require additional space to save embeddings.
This is basically a simpler version of https://github.com/Atcold/torch-TripletEmbedding .


## Install

Install this module via luarocks

```
luarocks install https://raw.githubusercontent.com/jhjin/triplet-criterion/master/rocks/triplet-scm-1.rockspec
```


## Usage

The loss function can be used in the same way as other criterions except few parameters as follows.
Please see here https://github.com/jhjin/imagenet-multiGPU.torch/blob/face-id/model.lua#L39 for more details.

```lua
require('triplet')
local loss = nn.TripletCriterion(samples, blocks, norm, margin) 
```

+ `samples` : the number of faces sampled from each identity in a batch
+ `blocks` : the number of identities in a batch (`samples` x `blocks` < `batchSize`)
+ `norm` : Lp-norm for distances between embeddings (default 2)
+ `margin` : a hypersphere margin between anchor-positive and anchor-negative pairs (default 0.2)

In a mini-batch, samples from the same identity should be prepared in a consecutive ordering by thier batch index.
In the case of 2 `samples` and 3 `blocks` with a `batchSize` of 8, for example, the batch should be prepared in

| Batch index | Identity                |
|-------------|-------------------------|
| 1           | Person A                |
| 2           | Person A                |
| 3           | Person B                |
| 4           | Person B                |
| 5           | Person C                |
| 6           | Person C                |
| 7           | Person randomly sampled |
| 8           | Person randomly sampled |

From the example, anchor and positive embeddings are selected from
the first `samples` x `blocks` region (batch index 1~6)
while negative embeddings are selected from
the rest (`batchSize` - `samples` x `blocks`) of the region (batch index 7,8).

However, the sampling method is already taken into account in the code
https://github.com/jhjin/imagenet-multiGPU.torch/blob/face-id/dataset.lua#L326
Hence no change is required if you plan to use the training script.


## Training

Entire training script can be found in the `face-id` branch of the repository.
https://github.com/jhjin/imagenet-multiGPU.torch/tree/face-id
A large size of batch is preferred in order to let the training converge to a higher score/accuracy.

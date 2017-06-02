
## Ceci n'est pas un benchmark

Deep learning has been extremely successful on a few classes of data/machine learning
problems such as involving images, speech and text (supervised learning) and games (reinforcement
learning).

In this repo we'll look at the performance of the most commonly used deep learning tools 
*with high-level API* from R/Python (keras on tensorflow and theano backends, mxnet, neon etc.) 
running on EC2 machines with GPUs (P2 instances with NVIDIA Tesla K80 GPUs)
using the most common network architectures on basic datasets of the classes mentioned above.

This author's interest is mainly in "traditional" machine learning problems such as
fraud detection, credit scoring or churn, and it seems that on that kind of data/problems
deep learning is not as successful and 
[it provides lower accuracy](https://github.com/szilard/benchm-ml#deep-neural-networks) 
than random forests or gradient boosting machines. 
Unfortunately most of the hype surrounding deep learning and "artificial intelligence" does not
acknowledge this reality.



### Conv-nets (CNN) 

Image recognition on the MNIST dataset. 2x conv+pooling, dropout, 2x fully connected layers,
trained with SGD for 10 epochs.
Code and detailed results [here](cnn-mnist).


#### By Tools

On p2.xlarge (1 GPU):

Tool               | Time (s) | vs Best |  vs TF  |GPU (%) | CPUs  | CPU1 (%) | Error rate
-------------------|----------|---------|---------|--------|-------|----------|----------
neon               |   25     |         |  0.7x   |  57    |  1    |   100    |   0.9%
Keras (tensorflow) |   37     |**1.5x** |         | 66     |  4    |   18     |   0.8%
R Keras (TF)       |   55     |**2.2x** |  1.5x   | 58     |  4    |   25     |   1.0%
Keras (CNTK)       |   58     |**2.3x** |  1.6x   | 80     |  1    |   100    |   0.8%
Keras (theano)     |   130    | **5x**  |  3.5x   | 97     |  1    |   100    |   0.9%
mxnet              |   50     | **2x**  |  1.3x   | 94     |  4    |   34     |   1.0%


#### By GPU number/types

Tensorflow backend:

Instance   | GPUs |   GPU     | Time (s) | GPU (%) |  GPUs
-----------|------|-----------|----------|---------|--------
p2.xlarge  |  1   | Tesla K80 |   37     |  66     |  1
p2.8xlarge |  8   | Tesla K80 |   36     | 67 (1)  |  1/8
g2.2xlarge |  1   | GRID K520 |   56     |  70     |  1

**P2** GPU is **1.5x** faster than **G2**. 

Tensorflow uses only 1 GPU even on the multi-GPU server (maybe because of small dataset/model).

mxnet can be set to use multiple GPUs, but then it runs slower (probably because of small dataset/model).

Theano obtained multi-GPU support only recently and that feature has not been added to Keras yet.

Neon disabled multi-GPU support (except on their cloud hosted version).


#### GPU vs CPU

Tensorflow backend:

Device       | Time (s)  | vs GPU
-------------|-----------|---------
GPU (P2)     |   37      |   1x
CPU 4 cores  |  326      |   9x
CPU 32 cores |  130      |  3.5x

"If it's not running on the GPU, it's crap" - Scott Le Grand [[ref](http://datascience.la/dsstne-a-new-deep-learning-framework-for-large-sparse-datasets/)]



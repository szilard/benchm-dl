
## Ceci n'est pas un benchmark

Deep learning has been extremely successful on a few classes of data/machine learning
problems such as involving images, speech and text (supervised learning) and games (reinforcement
learning).

In this repo we'll look at the performance of the most commonly used deep learning tools 
*with high-level API* (keras on tensorflow and theano backends, mxnet etc.) 
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

Image recognition on the MNIST dataset.
Code and detailed results [here](cnn-mnist).

On p2.xlarge (1 GPU):

Tool               | Time (s) | GPU (%) | CPUs  | CPU1 (%) | Accuracy
-------------------|----------|---------|-------|----------|----------
Keras (tensorflow) |   37     |  66     |  4    |   18     |   0.8%
Keras (theano)     |   146    |  86     |  1    |   100    |   0.9%

Theano backend is 4x slower and uses only one CPU core.

TODO: Fix `CNMeM is disabled, cuDNN 5103` for theano.

Tensorflow backend:

Instance   | GPUs |   GPU     | Time (s) | GPU (%) |  GPUs
-----------|------|-----------|----------|---------|--------
p2.xlarge  |  1   | Tesla K80 |   37     |  66     |  1
p2.8xlarge |  8   | Tesla K80 |   36     | 67 (1)  |  1/8
g2.2xlarge |  1   | GRID K520 |   56     |  70     |  1

P2 GPU is 1.5x faster than G2. 

Tensorflow uses only 1 GPU even on the multi-GPU server (on this small dataset/model).




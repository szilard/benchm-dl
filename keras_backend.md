# Keras Backend Benchmark
Inspired by [Max Woolf's benchmark](http://minimaxir.com/2017/06/keras-cntk/), the performance of 3 different backends (Theano, TensorFlow, and CNTK) of Keras with 3 different GPUs (K80, Titan X, and 1080 Ti) across various neural network tasks are compared.

For the performance of TensorFlow and CNTK with K80, the numbers reported at [Max Woolf's benchmark](http://minimaxir.com/2017/06/keras-cntk/) are used.

## Conclusion
The accuracies of Theano, TensorFlow and CNTK backends are similar across all benchmark tests, while speeds vary a lot. Theano is significantly (up to 50 times) slower than TensorFlow and CNTK. Between TensorFlow and CNTK, CNTK is a lot (about 2 to 4 times) faster than TensorFlow for LSTM (Bidirectional LSTM on IMDb Data and Text Generation via LSTM), while speeds for other type of neural networks are close to each other.

Among K80, Titan X and 1080 Ti GPUs, 1080 Ti is the fastest and K80 is the slowest. Theano is significantly (up to 14 times) faster on 1080 Ti than on Titan X, while the improvements for TensorFlow and CNTK are moderate.

## Results
### Bidirectional LSTM on IMDb Data
CNTK is significantly faster than TensorFlow and Theano.

#### Accuracy
Validation accuray after 4 epochs

Backend				|	K80		|  Titan X	|	 1080 Ti
--------------------|-----------|---------------|-----------------
Theano				| 			| 0.8310		| 0.8364 
TensorFlow			| 0.8343	|**0.8327**		| 0.8313
CNTK				|**0.8354**	| 0.8325		|**0.8388**

#### Speed
Average time in seconds for 4 epochs

Backend				|	K80			|  Titan X	|	 1080 Ti
--------------------|---------------|---------------|-----------------
Theano				| 				| 310.3	(3.1x)	| 148.5 (1.6x)
TensorFlow			| 276.6	(1.8x)	| 295.3	(3.0x)	| 245.5 (2.6x)
CNTK				|**152.4**		|**99.5**		|**94.3**

### Fasttext on IMDb Data
TensorFlow and CNTK are significantly faster than Theano, while CNTK is slightly faster than TensorFlow.

#### Accuracy
Validation accuracy after 5 epochs

Backend				|	K80		|  Titan X	|	 1080 Ti
--------------------|-----------|---------------|-----------------
Theano				| 			|**0.8856**		|**0.8856**
TensorFlow			|**0.9068**	|**0.8856**		|**0.8856**
CNTK				| 0.9064	| 0.8854		| 0.8854

#### Speed
Average time in seconds for 5 epochs

Backend				|	K80			|  Titan X	|	 1080 Ti
--------------------|---------------|---------------|-----------------
Theano				| 				| 10.9 (3.2x)	| 9.5 (3.5x)
TensorFlow			|**58.3**		|  6.0 (1.8x)	| 4.5 (1.7x)
CNTK				| 69.5 (1.2x)	|**3.4**		|**2.7**

### MLP on MNIST Data
TensorFlow and CNTK are significantly faster than Theano, while CNTK is slightly faster than TensorFlow.

#### Accuracy
Validation accuracy after 20 epochs

Backend				|	K80		|  Titan X	|	 1080 Ti
--------------------|-----------|---------------|-----------------
Theano				| 			|**0.9843**		|**0.9843**
TensorFlow			|**0.9839**	|0.9804			|0.9840
CNTK				| 0.9796	|0.9835			|0.9830

#### Speed
Average time in seconds for 20 epochs

Backend				|	K80			|  Titan X	|	 1080 Ti
--------------------|---------------|---------------|-----------------
Theano				| 				|20.0 (14.3x)	| 3.6 (2.8x)
TensorFlow			|3.4 (1.2x)		|2.8 (2.0x)		| 1.9 (1.5x)
CNTK				|**2.8**		|**1.4**		|**1.3**

### CNN on MNIST Data
TensorFlow and CNTK are significantly faster than Theano while TensorFlow is slightly faster than CNTK.

#### Accuracy
Validation accuracy after 12 epochs

Backend				|	K80		|  Titan X	|	 1080 Ti
--------------------|-----------|---------------|-----------------
Theano				| 			|0.9905			|0.9894
TensorFlow			|**0.9916**	|**0.9909**		|0.9899
CNTK				|0.9892		|0.9908			|**0.9910**

#### Speed
Average time in seconds for 12 epochs

Backend				|	K80			|  Titan X	|	 1080 Ti
--------------------|---------------|---------------|-----------------
Theano				| 				|332.2 (53.6x)	|17.3 (3.7x)
TensorFlow			|**11.1**		|**6.2**		|**4.7**
CNTK				|15.9 (1.4x)	|7.0 (1.1x)		|5.0 (1.1x)

### CNN on CIFAR-10
TensorFlow and CNTK are significantly faster than Theano while TensorFlow is slightly faster than CNTK.

#### Accuracy
Validation accuracy after 20 epochs

Backend				|	K80		|  Titan X	|	 1080 Ti
--------------------|-----------|---------------|-----------------
Theano				| 			|**0.7446**		|**0.7531**
TensorFlow			|**0.7453**	|0.7366			|0.7366
CNTK				|0.7400		|0.7226			|0.7410

#### Speed
Average time in seconds for 20 epochs

Backend				|	K80			|  Titan X	|	 1080 Ti
--------------------|---------------|---------------|-----------------
Theano				| 				|630.3 (24.5x)	|40.3 (2.8x)
TensorFlow			|**39.2**		|**17.0**		|**14.2**
CNTK				|40.3 			|25.7 (1.5x)	|24.2 (1.7x)

### Text Generation via LSTM
CNTK is significantly faster than TensorFlow and Theano.

#### Loss
Validation loss after 10 epochs

Backend				|	K80		|  Titan X	|	 1080 Ti
--------------------|-----------|---------------|-----------------
Theano				| 			|1.4268			|1.3988
TensorFlow			|1.4060		|1.4138			|1.3976
CNTK				|**1.4047**	|1.3980			|**1.3956**

#### Speed
Average time in seconds for 10 epochs

Backend				|	K80			|  Titan X	|	 1080 Ti
--------------------|---------------|---------------|-----------------
Theano				| 				|548.3 (13.2x)	|83.1 (2.5x)
TensorFlow			|87.6 (1.9x)	|162.2 (3.9x)	|107.6 (3.3x)
CNTK				|**46.4**		|**41.4**		|**32.7**

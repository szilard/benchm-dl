

## adapted from https://github.com/NervanaSystems/neon/blob/master/examples/mnist_mlp.py
##          and https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_conv.py


from __future__ import print_function
from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Conv, Dropout, Pooling
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Softmax, CrossEntropyBinary, Misclassification
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
import time


parser = NeonArgparser(__doc__)
args = parser.parse_args()
print(args)

dataset = MNIST(path=args.data_dir)
train_set = dataset.train_iter
test_set = dataset.valid_iter


init_norm = Gaussian(loc=0.0, scale=0.1)

layers = [Conv((4, 4, 32), init=init_norm, activation=Rectlin()),
          Pooling((2, 2)),
          Conv((3, 3, 16), init=init_norm, activation=Rectlin()),
          Pooling((2, 2)),
          Dropout(keep=0.8),
          Affine(nout=128, init=init_norm, activation=Rectlin()),
          Affine(nout=64, init=init_norm, activation=Rectlin()),
          Affine(nout=10, init=init_norm, activation=Softmax())]      
mlp = Model(layers=layers) 

cost = GeneralizedCost(costfunc=CrossEntropyBinary())
optimizer = GradientDescentMomentum(learning_rate=0.05, momentum_coef=0.9, wdecay=1e-5)

callbacks = Callbacks(mlp, eval_set=train_set, **args.callback_args)


start = time.time()
mlp.fit(train_set, optimizer=optimizer, num_epochs=10, cost=cost, callbacks=callbacks)
end = time.time()
print('Train time:', end - start, 'sec')


error_rate = mlp.eval(test_set, metric=Misclassification())
print('Error rate:', 100*error_rate[0], '%')



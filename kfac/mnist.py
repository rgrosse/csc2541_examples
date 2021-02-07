import numpy as np
import tensorflow_datasets as tfds

import autoencoders



## def MNISTArchitecture():
##     net_init, net_apply = named_serial(
##         ('enc1z', Dense(1000, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('enc1a', elementwise(nn.sigmoid)),
##         ('enc2z', Dense(500, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('enc2a', elementwise(nn.sigmoid)),
##         ('enc3z', Dense(250, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('enc3a', elementwise(nn.sigmoid)),
##         ('code', Dense(30, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec1z', Dense(250, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec1a', elementwise(nn.sigmoid)),
##         ('dec2z', Dense(500, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec2a', elementwise(nn.sigmoid)),
##         ('dec3z', Dense(1000, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec3a', elementwise(nn.sigmoid)),
##         ('out', Dense(784, W_init=sparse_init(), b_init=initializers.zeros)),
##     )
##     param_info = (('in', 'enc1z'),
##                   ('enc1a', 'enc2z'),
##                   ('enc2a', 'enc3z'),
##                   ('enc3a', 'code'),
##                   ('code', 'dec1z'),
##                   ('dec1a', 'dec2z'),
##                   ('dec2a', 'dec3z'),
##                   ('dec3a', 'out')
##                  )
##     in_shape=(-1, 784)
##     flatten, unflatten = get_flatten_fns(net_init, in_shape)
##     return Architecture(net_init, net_apply, in_shape, flatten, unflatten, param_info)

def get_architecture():
    layer_sizes = [('enc1', 1000),
                   ('enc2', 500),
                   ('enc3', 250),
                   ('code', 30),
                   ('dec1', 250),
                   ('dec2', 500),
                   ('dec3', 1000)]

    return autoencoders.get_architecture(784, layer_sizes)


def get_config():
    return autoencoders.default_config()




def run():
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    X_train = train_data['image'].reshape((-1, 784)).astype(np.float32) / 255
    X_test = test_data['image'].reshape((-1, 784)).astype(np.float32) / 255

    config = get_config()
    arch = get_architecture()

    autoencoders.run_training(X_train, X_test, arch, config)

if __name__ == '__main__':
    run()



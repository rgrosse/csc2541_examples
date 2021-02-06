from jax import numpy as np, nn
from jax.experimental.stax import Dense, elementwise
from jax.nn import initializers
import numpy as onp
import tensorflow_datasets as tfds
import time

import kfac
from kfac_util import Architecture, BernoulliModel, get_flatten_fns, named_serial, sparse_init



def MNISTArchitecture():
    net_init, net_apply = named_serial(
        ('enc1z', Dense(1000, W_init=sparse_init(), b_init=initializers.zeros)),
        ('enc1a', elementwise(nn.sigmoid)),
        ('enc2z', Dense(500, W_init=sparse_init(), b_init=initializers.zeros)),
        ('enc2a', elementwise(nn.sigmoid)),
        ('enc3z', Dense(250, W_init=sparse_init(), b_init=initializers.zeros)),
        ('enc3a', elementwise(nn.sigmoid)),
        ('code', Dense(30, W_init=sparse_init(), b_init=initializers.zeros)),
        ('dec1z', Dense(250, W_init=sparse_init(), b_init=initializers.zeros)),
        ('dec1a', elementwise(nn.sigmoid)),
        ('dec2z', Dense(500, W_init=sparse_init(), b_init=initializers.zeros)),
        ('dec2a', elementwise(nn.sigmoid)),
        ('dec3z', Dense(1000, W_init=sparse_init(), b_init=initializers.zeros)),
        ('dec3a', elementwise(nn.sigmoid)),
        ('out', Dense(784, W_init=sparse_init(), b_init=initializers.zeros)),
    )
    param_info = (('in', 'enc1z'),
                  ('enc1a', 'enc2z'),
                  ('enc2a', 'enc3z'),
                  ('enc3a', 'code'),
                  ('code', 'dec1z'),
                  ('dec1a', 'dec2z'),
                  ('dec2a', 'dec3z'),
                  ('dec3a', 'out')
                 )
    in_shape=(-1, 784)
    flatten, unflatten = get_flatten_fns(net_init, in_shape)
    return Architecture(net_init, net_apply, in_shape, flatten, unflatten, param_info)


def mnist_config():
    config = {}
    config['max_iter'] = 20000
    config['initial_batch_size'] = 1000
    config['final_batch_size_iter'] = 500
    config['batch_size_granularity'] = 50
    config['chunk_size'] = 5000
    
    config['cov_update_interval'] = 1
    config['cov_batch_ratio'] = 1/8
    config['cov_timescale'] = 20
    
    config['eig_update_interval'] = 20

    config['lambda_update_interval'] = 5
    config['init_lambda'] = 150
    config['lambda_drop'] = 0.95
    config['lambda_boost'] = 1 / config['lambda_drop']
    config['lambda_min'] = 0
    config['lambda_max'] = onp.infty

    config['weight_cost'] = 1e-5

    config['gamma_update_interval'] = 20
    config['init_gamma'] = np.sqrt(config['init_lambda'] + config['weight_cost'])
    config['gamma_drop'] = np.sqrt(config['lambda_drop'])
    config['gamma_boost'] = 1 / config['gamma_drop']
    config['gamma_max'] = 1
    config['gamma_min'] = np.sqrt(config['weight_cost'])
    
    config['param_timescale'] = 100
    
    return config




def run():
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    X_train = train_data['image'].reshape((-1, 784)).astype(np.float32) / 255
    X_test = test_data['image'].reshape((-1, 784)).astype(np.float32) / 255

    config = mnist_config()
    arch = MNISTArchitecture()
    init_state = kfac.kfac_init(arch, BernoulliModel, X_train, X_train, config)


    state = init_state
    for i in range(20000):
        t0 = time.time()
        state = kfac.kfac_iter(state, arch, BernoulliModel, X_train, X_train, config)

        print('Step', i)
        print('Time:', time.time() - t0)
        print('Alpha:', state['coeffs'][0])
        if i > 0:
            print('Beta:', state['coeffs'][1])
        print('Quadratic decrease:', state['quad_dec'])

        if i % 20 == 0:
            cost = kfac.compute_cost(arch, BernoulliModel, state['w'], X_train, X_train, 
                                     config['weight_cost'], config['chunk_size'])
            print('Training objective:', cost)
            cost = kfac.compute_cost(arch, BernoulliModel, state['w_avg'], X_train, X_train, 
                                     config['weight_cost'], config['chunk_size'])
            print('Training objective (averaged):', cost)

            cost = kfac.compute_cost(arch, BernoulliModel, state['w'], X_test, X_test, 
                                     config['weight_cost'], config['chunk_size'])
            print('Test objective:', cost)
            cost = kfac.compute_cost(arch, BernoulliModel, state['w_avg'], X_test, X_test, 
                                     config['weight_cost'], config['chunk_size'])
            print('Test objective (averaged):', cost)

        if i % config['lambda_update_interval'] == 0:
            print('New lambda:', state['lambda'])
        if i % config['gamma_update_interval'] == 0:
            print('New gamma:', state['gamma'])
        print()

if __name__ == '__main__':
    run()



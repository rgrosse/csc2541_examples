from jax import nn, numpy as np
from jax.experimental import stax
import numpy as onp
import time

import kfac
import kfac_util


def get_architecture(input_size, layer_sizes):
    """Construct a sigmoid MLP autoencoder architecture with the given layer sizes.
    The code layer, given by the name 'code', is linear."""
    
    layers = []
    param_info = []
    act_name = 'in'
    for name, lsize in layer_sizes:
        if name == 'code':
            # Code layer is special because it's linear
            param_info.append((act_name, name))
            act_name = name

            layers.append((name, stax.Dense(
                lsize, W_init=kfac_util.sparse_init(), b_init=nn.initializers.zeros)))
        else:
            preact_name = name + 'z'
            param_info.append((act_name, preact_name))
            act_name = name + 'a'

            layers.append((preact_name, stax.Dense(
                lsize, W_init=kfac_util.sparse_init(), b_init=nn.initializers.zeros)))
            layers.append((act_name, stax.elementwise(nn.sigmoid)))

    layers.append(('out', stax.Dense(
        input_size, W_init=kfac_util.sparse_init(), b_init=nn.initializers.zeros)))

    param_info.append((act_name, 'out'))
    param_info = tuple(param_info)

    net_init, net_apply = kfac_util.named_serial(*layers)

    in_shape = (-1, input_size)
    flatten, unflatten = kfac_util.get_flatten_fns(net_init, in_shape)

    return kfac_util.Architecture(net_init, net_apply, in_shape, flatten, unflatten, param_info)


def default_config():
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
    config['init_gamma'] = onp.sqrt(config['init_lambda'] + config['weight_cost'])
    config['gamma_drop'] = onp.sqrt(config['lambda_drop'])
    config['gamma_boost'] = 1 / config['gamma_drop']
    config['gamma_max'] = 1
    config['gamma_min'] = onp.sqrt(config['weight_cost'])
    
    config['param_timescale'] = 100
    
    return config

def squared_error(logits, T):
    """Compute the squared error. For consistency with James's code, don't
    rescale by 0.5."""
    y = nn.sigmoid(logits)
    return np.sum((y-T)**2)

def run_training(X_train, X_test, arch, config):
    nll_fn = kfac_util.BernoulliModel.nll_fn
    state = kfac.kfac_init(arch, kfac_util.BernoulliModel, X_train, X_train, config)
    for i in range(config['max_iter']):
        t0 = time.time()
        state = kfac.kfac_iter(state, arch, kfac_util.BernoulliModel, X_train, X_train, config)

        print('Step', i)
        print('Time:', time.time() - t0)
        print('Alpha:', state['coeffs'][0])
        if i > 0:
            print('Beta:', state['coeffs'][1])
        print('Quadratic decrease:', state['quad_dec'])

        if i % 20 == 0:
            print()
            cost = kfac.compute_cost(arch, nll_fn, state['w'], X_train, X_train, 
                config['weight_cost'], config['chunk_size'])
            print('Training objective:', cost)
            cost = kfac.compute_cost(
                arch, nll_fn, state['w_avg'], X_train, X_train, 
                config['weight_cost'], config['chunk_size'])
            print('Training objective (averaged):', cost)

            cost = kfac.compute_cost(arch, nll_fn, state['w'], X_test, X_test, 
                config['weight_cost'], config['chunk_size'])
            print('Test objective:', cost)
            cost = kfac.compute_cost(
                arch, nll_fn, state['w_avg'], X_test, X_test, 
                config['weight_cost'], config['chunk_size'])
            print('Test objective (averaged):', cost)

            print()
            cost = kfac.compute_cost(arch, squared_error, state['w'], X_train, X_train, 
                0., config['chunk_size'])
            print('Training error:', cost)
            cost = kfac.compute_cost(arch, squared_error, state['w_avg'], X_train, X_train, 
                0., config['chunk_size'])
            print('Training error (averaged):', cost)

            cost = kfac.compute_cost(arch, squared_error, state['w'], X_test, X_test, 
                0., config['chunk_size'])
            print('Test error:', cost)
            cost = kfac.compute_cost(arch, squared_error, state['w_avg'], X_test, X_test, 
                0., config['chunk_size'])
            print('Test error (averaged):', cost)
            print()
            

        if i % config['lambda_update_interval'] == 0:
            print('New lambda:', state['lambda'])
        if i % config['gamma_update_interval'] == 0:
            print('New gamma:', state['gamma'])
        print()



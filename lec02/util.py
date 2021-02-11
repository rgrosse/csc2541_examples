from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, jvp, vjp, jit, numpy as np, random
from jax.experimental.stax import serial, Dense, Relu, Tanh
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.nn import relu
from matplotlib import pyplot as plt
import numpy as onp
import scipy.sparse
from collections import namedtuple



def named_serial(*layers):
    # based on jax.experimental.stax.serial
    nlayers = len(layers)
    names, fns = zip(*layers)
    init_fns, apply_fns = zip(*fns)
    output_name = names[-1]

    def init_fn(rng, input_shape):
        params = {}
        for name, init_fn in zip(names, init_fns):
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fn(layer_rng, input_shape)
            params[name] = param
        return input_shape, params

    def apply_fn(params, inputs, ret=None, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        result = {}
        for fun, name, rng in zip(apply_fns, names, rngs):
            inputs = result[name] = fun(params[name], inputs, rng=rng, **kwargs)

        if ret is None:
            return inputs
        elif ret == 'all':
            return result
        else:
            return result[ret]

    return init_fn, apply_fn

def make_float64(params):
    return tree_map(lambda x: x.astype(np.float64), params)

def get_flatten_fns(init_fn, in_shape, float64=True):
    rng = random.PRNGKey(0)
    _, dummy_params = init_fn(rng, in_shape)
    if float64:
        dummy_params = make_float64(dummy_params)
    _, unflatten = ravel_pytree(dummy_params)
    def flatten(p):
        return ravel_pytree(p)[0]
    return flatten, unflatten






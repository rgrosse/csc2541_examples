from collections import namedtuple
from jax import grad, numpy as np, random, jvp, nn
from jax.flatten_util import ravel_pytree
from jax.ops import index_update
from jax.tree_util import tree_map



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

    def apply_fn(params, inputs, add_to={}, ret_all=False, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        result = {'in': inputs}
        for fun, name, rng in zip(apply_fns, names, rngs):
            inputs = fun(params[name], inputs, rng=rng, **kwargs)
            if name in add_to:
                inputs = inputs + add_to[name]
            if ret_all:
                result[name] = inputs

        if ret_all:
            return inputs, result
        else:
            return inputs

    return init_fn, apply_fn

Architecture = namedtuple('Architecture', ['net_init', 'net_apply', 'in_shape',
                                           'flatten', 'unflatten', 'param_info'])

OutputModel = namedtuple('OutputModel', ['nll_fn', 'sample_grads_fn'])


def bernoulli_nll(logits, T):
    """Compute the sum (not the mean) of the losses on a batch."""
    log_p = -np.logaddexp(0, -logits)
    log_1_minus_p = -np.logaddexp(0, logits)
    return -np.sum(T * log_p + (1-T) * log_1_minus_p)

def bernoulli_sample_grads(logits, key):
    """Sample a vector whose covariance is the output layer metric (i.e. Fisher information)."""
    Y = nn.sigmoid(logits)
    T = random.bernoulli(key, Y)
    return Y - T

BernoulliModel = OutputModel(bernoulli_nll, bernoulli_sample_grads)



def make_float64(params):
    return tree_map(lambda x: x.astype(np.float64), params)

def get_flatten_fns(init_fn, in_shape, float64=False):
    rng = random.PRNGKey(0)
    _, dummy_params = init_fn(rng, in_shape)
    if float64:
        dummy_params = make_float64(dummy_params)
    _, unflatten = ravel_pytree(dummy_params)
    def flatten(p):
        return ravel_pytree(p)[0]
    return flatten, unflatten

def hvp(J, w, v):
    return jvp(grad(J), (w,), (v,))[1]

def get_ema_param(timescale):
    return 1 - 1 / timescale


def sparse_init(num_conn=15, stdev=1.):
    def init(rng, shape):
        k1, k2 = random.split(rng)
        in_dim, out_dim = shape
        num_conn_ = np.minimum(num_conn, in_dim)
        W = np.zeros(shape)
        row_idxs = np.outer(np.arange(in_dim), np.ones(out_dim)).astype(np.uint32)
        row_idxs = random.shuffle(k1, row_idxs)[:num_conn_, :].ravel()
        col_idxs = np.outer(np.ones(num_conn_), np.arange(out_dim)).astype(np.uint32).ravel()
        vals = random.normal(k2, shape=(num_conn_*out_dim,)) * stdev
        return index_update(W, (row_idxs, col_idxs), vals)
    return init


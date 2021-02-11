from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, jvp, jit, numpy as np, random
from jax.experimental.stax import Dense, Tanh
from matplotlib import pyplot as plt
import numpy as onp
from collections import namedtuple

from core import gnhvp, approx_solve
from util import named_serial, make_float64, get_flatten_fns


#########################################
## Implementation of response Jacobian ##
#########################################

def mixed_second_mvp(J_param, w, t, R_t):
    grad_cost = grad(J_param, 0)      # gradient w.r.t. w
    grad_cost_t = lambda t: grad_cost(w, t)
    return jvp(grad_cost_t, (t,), (R_t,))[1]     # forward-over-reverse

def dampen(mvp, lam):
    def new_mvp(x):
        return mvp(x) + lam*x
    return new_mvp

def approx_solve_H(f_param, L_param, w, phi, Rg_w, lam, niter):
    mvp = lambda v: gnhvp(lambda w: f_param(w, phi),
                          lambda y: L_param(y, phi), w, v)
    mvp_damp = dampen(mvp, lam)
    return approx_solve(mvp_damp, Rg_w, niter)

def response_jacobian_vector_product(f_param, L_param, w, phi, R_phi, lam, niter):
    def J_param(w, phi):
        return L_param(f_param(w, phi), phi)
    Rg_w = mixed_second_mvp(J_param, w, phi, R_phi)
    return approx_solve_H(f_param, L_param, w, phi, -Rg_w, lam, niter)


########################################
## 1-D Regression example             ##
########################################


Architecture = namedtuple('Architecture', ['net_init', 'net_apply', 'in_shape',
                                           'flatten', 'unflatten'])

def ToyMLP():
    net_init, net_apply = named_serial(
        ('z1', Dense(256)),
        ('h1', Tanh),
        ('z2', Dense(256)),
        ('h2', Tanh),
        ('y', Dense(1)))
    in_shape = (-1, 1)
    flatten, unflatten = get_flatten_fns(net_init, in_shape)
    return Architecture(net_init, net_apply, in_shape, flatten, unflatten)


def f_net(arch, w, x):
    x_in = x.reshape((-1, 1))
    return arch.net_apply(arch.unflatten(w), x_in).ravel()
def L(y, t):
    return 0.5 * np.sum((y-t)**2)

def make_parameterized_cost(arch, x, L):
    """Make a cost function parameterized by a vector phi. Here, the cost
    is squared error, and phi represents the targets."""
    def f_param(w, phi):
        return f_net(arch, w, x)
    def L_param(y, phi):
        return 0.5 * np.sum((y - phi)**2)
    return f_param, L_param


def generate_toy_data2():
    x1 = onp.random.uniform(-5, -2, size=50)
    fx1 = onp.sin(2*x1) - 2
    x2 = onp.random.uniform(2, 5, size=50)
    fx2 = onp.sin(2*x2) + 2
    x3 = np.array([0.])
    fx3 = np.array([-1.])
    x = onp.concatenate([x1, x2, x3])
    fx = onp.concatenate([fx1, fx2, fx3])
    t = onp.random.normal(fx, 0.5)
    return x, t

def train_toy_network():
    onp.random.seed(0)

    x, t = generate_toy_data2()
    x *= 0.2
    arch = ToyMLP()
    rng = random.PRNGKey(0)
    ALPHA = 1e-1

    out_shape, net_params = arch.net_init(rng, arch.in_shape)
    w_init = make_float64(arch.flatten(net_params))

    def train_obj(w):
        net_params = arch.unflatten(w)
        x_in = x.reshape((-1, 1))
        y = arch.net_apply(net_params, x_in).ravel() 

        return 0.5 * np.mean((y-t)**2)

    grad_train_obj = jit(grad(train_obj))

    w_curr = w_init.copy()
    for i in range(10000):
        w_curr -= ALPHA * grad_train_obj(w_curr)
    w_opt = w_curr.copy()

    plt.figure()
    plt.plot(x, t, 'bx')

    x_in = np.linspace(-1, 1, 100).reshape((-1, 1))
    y = arch.net_apply(arch.unflatten(w_opt), x_in)
    plt.plot(x_in.ravel(), y, 'r-')

    return x, t, w_opt

def make_figures(x, t, w_opt, idx):
    """Generate the sensitivity analysis figures. The argument idx is the index
    of the training example to perturb. The indices used in the figure are 0
    and 100 (the outlier)."""
    LAM = 1e-3
    OFFSET = 5
    NITER_VALS = [1, 2, 5, 10, 20, 50]

    arch = ToyMLP()

    R_t = onp.zeros(t.shape, dtype=onp.float64)
    R_t[idx] = OFFSET

    plt.figure()
    plt.plot(x, t, 'bx', alpha=0.5)
    plt.ylim(-4, 10)
    plt.xticks([])
    plt.yticks([])

    x_in = np.linspace(-1, 1, 100).reshape((-1, 1))
    y = arch.net_apply(arch.unflatten(w_opt), x_in).ravel()
    plt.plot(x_in.ravel(), y, 'r-')

    plt.figure()
    plt.plot(x, t, 'bx', alpha=0.3)
    plt.plot(x[idx], t[idx]+OFFSET, 'rx', ms=10)
    plt.plot(x[idx], t[idx], 'gx', ms=10)
    plt.ylim(-4, 10)
    plt.xticks([])
    plt.yticks([])

    x_in = np.linspace(-1, 1, 100).reshape((-1, 1))
    y = arch.net_apply(arch.unflatten(w_opt), x_in).ravel()

    f_param, L_param = make_parameterized_cost(arch, x, L)

    y_list = [y]
    for niter in NITER_VALS:
        R_w = response_jacobian_vector_product(f_param, L_param, w_opt, t, R_t, LAM, niter)
        R_y = jvp(lambda w: f_net(arch, w, x_in), (w_opt,), (R_w,))[1]
        y_list.append(y + R_y)

    labels = ['0'] + [str(i) for i in NITER_VALS]
    for i, curr_y in enumerate(y_list):
        r = i / (len(y_list) - 1)
        plt.plot(x_in.ravel(), curr_y, color=(r, 1-r, 0), alpha=0.5, label=labels[i])

    plt.legend(loc='upper left')

def run():
    x, t, w_opt = train_toy_network()
    make_figures(x, t, w_opt, 0)
    make_figures(x, t, w_opt, 100)



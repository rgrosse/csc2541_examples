from jax import grad, jit, numpy as np, random, vjp, jvp
from jax.scipy.linalg import eigh
import numpy as onp

import kfac_util


def L2_penalty(arch, w):
    # FIXME: don't regularize the biases
    return 0.5 * np.sum(w**2)

def get_batch_size(step, ndata, config):
    """Exponentially increasing batch size schedule."""
    step = np.floor(step/config['batch_size_granularity']) * config['batch_size_granularity']
    pwr = onp.minimum(step / (config['final_batch_size_iter']-1), 1.)
    return onp.floor(config['initial_batch_size'] * (ndata / config['initial_batch_size'])**pwr).astype(onp.uint32)

def get_sample_batch_size(batch_size, config):
    """Batch size to use for sampling the activation statistics."""
    return onp.ceil(config['cov_batch_ratio'] * batch_size).astype(onp.uint32)

def get_chunks(batch_size, chunk_size):
    """Iterator that breaks a range into smaller chunks. Useful for simulating
    larger batches than can fit on the GPU."""
    start = 0
    
    while start < batch_size:
        end = min(start+chunk_size, batch_size)
        yield slice(start, end)
        start = end


def make_instrumented_vjp(apply_fn, params, inputs):
    """Returns a function which takes in the output layer gradients and returns a dict
    containing the gradients for all the intermediate layers."""
    dummy_input = np.zeros((2,) + inputs.shape[1:])
    _, dummy_activations = apply_fn(params, dummy_input, ret_all=True)
    
    batch_size = inputs.shape[0]
    add_to = {name: np.zeros((batch_size,) + dummy_activations[name].shape[1:])
              for name in dummy_activations}
    apply_wrap = lambda a: apply_fn(params, inputs, a, ret_all=True)
    primals_out, vjp_fn, activations = vjp(apply_wrap, add_to, has_aux=True)
    return primals_out, vjp_fn, activations


def estimate_covariances_chunk(apply_fn, param_info, output_model, net_params, X_chunk, rng):
    """Compute the empirical covariances on a chunk of data."""
    logits, vjp_fn, activations = make_instrumented_vjp(apply_fn, net_params, X_chunk)
    key, rng = random.split(rng)
    output_grads = output_model.sample_grads_fn(logits, key)
    act_grads = vjp_fn(output_grads)[0]

    A = {}
    G = {}
    for in_name, out_name in param_info:
        a = activations[in_name]
        a_hom = np.hstack([a, np.ones((a.shape[0], 1))])
        A[in_name] = a_hom.T @ a_hom

        ds = act_grads[out_name]
        G[out_name] = ds.T @ ds

    return A, G

estimate_covariances_chunk = jit(estimate_covariances_chunk, static_argnums=(0,1,2))


def estimate_covariances(arch, output_model, w, X, rng, chunk_size):
    """Compute the empirical covariances on a batch of data."""
    batch_size = X.shape[0]
    net_params = arch.unflatten(w)
    A_sum = {in_name: 0. for in_name, out_name in arch.param_info}
    G_sum = {out_name: 0. for in_name, out_name in arch.param_info}
    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk = X[chunk_idxs,:]
        key, rng = random.split(rng)
        
        A_curr, G_curr = estimate_covariances_chunk(
            arch.net_apply, arch.param_info, output_model, net_params, X_chunk, key)
        A_sum = {name: A_sum[name] + A_curr[name] for name in A_sum}
        G_sum = {name: G_sum[name] + G_curr[name] for name in G_sum}
            
    A_mean = {name: A_sum[name] / batch_size for name in A_sum}
    G_mean = {name: G_sum[name] / batch_size for name in G_sum}
    
    return A_mean, G_mean

def update_covariances(A, G, arch, output_model, w, X, rng, cov_timescale, chunk_size):
    """Exponential moving average of the covariances."""
    A, G = dict(A), dict(G)
    curr_A, curr_G = estimate_covariances(arch, output_model, w, X, rng, chunk_size)
    ema_param = kfac_util.get_ema_param(cov_timescale)
    for k in A.keys():
        A[k] = ema_param * A[k] + (1-ema_param) * curr_A[k]
    for k in G.keys():
        G[k] = ema_param * G[k] + (1-ema_param) * curr_G[k]
    return A, G

def compute_pi(A, G):
    return np.sqrt((np.trace(A) * G.shape[0]) / (A.shape[0] * np.trace(G)))
    
def compute_inverses(arch, A, G, gamma):
    A_inv, G_inv = {}, {}
    for in_name, out_name in arch.param_info:
        pi = compute_pi(A[in_name], G[out_name])
        
        A_damp = gamma * pi
        A_inv[in_name] = np.linalg.inv(A[in_name] + A_damp * np.eye(A.shape[0]))
        
        G_damp = gamma / pi
        G_inv[out_name] = np.linalg.inv(G[out_name] + G_damp * np.eye(G.shape[0]))
        
    return A_inv, G_inv

def compute_eigs(arch, A, G):
    A_eig, G_eig, pi = {}, {}, {}
    for in_name, out_name in arch.param_info:
        A_eig[in_name] = eigh(A[in_name])
        G_eig[out_name] = eigh(G[out_name])
        pi[out_name] = compute_pi(A[in_name], G[out_name])
    return A_eig, G_eig, pi

def nll_cost(apply_fn, nll_fn, unflatten_fn, w, X, T):
    logits = apply_fn(unflatten_fn(w), X)
    return nll_fn(logits, T)

nll_cost = jit(nll_cost, static_argnums=(0, 1, 2))
grad_nll_cost = jit(grad(nll_cost, 3), static_argnums=(0, 1, 2))
    
def compute_cost(arch, output_model, w, X, T, weight_cost, chunk_size):
    batch_size = X.shape[0]
    total = 0

    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]
        total += nll_cost(arch.net_apply, output_model.nll_fn, arch.unflatten,
                          w, X_chunk, T_chunk)
        
    return total / batch_size + weight_cost * L2_penalty(arch, w)

def compute_gradient(arch, output_model, w, X, T, weight_cost, chunk_size):
    batch_size = X.shape[0]
    grad_w = 0
    
    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]
        
        grad_w += grad_nll_cost(arch.net_apply, output_model.nll_fn, arch.unflatten,
                                w, X_chunk, T_chunk)
        
    grad_w /= batch_size
    grad_w += weight_cost * grad(L2_penalty, 1)(arch, w)
    return grad_w

def compute_natgrad_from_inverses(arch, grad_w, A_inv, G_inv):
    param_grad = arch.unflatten(grad_w)
    natgrad = {}
    for in_name, out_name in arch.param_info:
        grad_W, grad_b = param_grad[out_name]
        grad_Wb = np.vstack([grad_W, grad_b.reshape((1, -1))])
        
        natgrad_Wb = A_inv[in_name] @ grad_Wb @ G_inv[out_name]
        
        natgrad_W, natgrad_b = natgrad_Wb[:-1, :], natgrad_Wb[-1, :]
        natgrad[out_name] = (natgrad_W, natgrad_b)
    return arch.flatten(natgrad)

def compute_natgrad_from_eigs_helper(param_info, param_grad, A_eig, G_eig, pi, gamma):
    natgrad = {}
    for in_name, out_name in param_info:
        grad_W, grad_b = param_grad[out_name]
        grad_Wb = np.vstack([grad_W, grad_b.reshape((1, -1))])
        
        A_d, A_Q = A_eig[in_name]
        G_d, G_Q = G_eig[out_name]
        
        # rotate into Kronecker eigenbasis
        grad_rot = A_Q.T @ grad_Wb @ G_Q
        
        # add damping and divide
        denom = np.outer(A_d + gamma * pi[out_name],
                         G_d + gamma / pi[out_name])
        natgrad_rot = grad_rot / denom
        
        # rotate back to the original basis
        natgrad_Wb = A_Q @ natgrad_rot @ G_Q.T
        
        natgrad_W, natgrad_b = natgrad_Wb[:-1, :], natgrad_Wb[-1, :]
        natgrad[out_name] = (natgrad_W, natgrad_b)
    return natgrad

compute_natgrad_from_eigs_helper = jit(compute_natgrad_from_eigs_helper, static_argnums=(0,))

def compute_natgrad_from_eigs(arch, grad_w, A_eig, G_eig, pi, gamma):
    param_grad = arch.unflatten(grad_w)
    natgrad = compute_natgrad_from_eigs_helper(
        arch.param_info, param_grad, A_eig, G_eig, pi, gamma)
    return arch.flatten(natgrad)

def compute_A_chunk(apply_fn, nll_fn, unflatten_fn, w, X, T, dirs, grad_w):
    ndir = len(dirs)
    predict_wrap = lambda w: apply_fn(unflatten_fn(w), X)
    
    RY, RgY = [], []
    for v in dirs:
        Y, RY_ = jvp(predict_wrap, (w,), (v,))
        nll_wrap = lambda Y: nll_fn(Y, T)
        RgY_ = kfac_util.hvp(nll_wrap, Y, RY_)
        RY.append(RY_)
        RgY.append(RgY_)

    A = np.array([[onp.sum(RY[i] * RgY[j])
                   for j in range(ndir)]
                  for i in range(ndir)])

    return A

compute_A_chunk = jit(compute_A_chunk, static_argnums=(0, 1, 2))


def compute_step_coeffs(arch, output_model, w, X, T, dirs, grad_w,
                        weight_cost, lmbda, chunk_size):
    """Compute the coefficients alpha and beta which minimize the quadratic
    approximation to the cost in the update:
    
            new_update = sum of coeffs[i] * dirs[i]

    Note that, unlike the rest of the K-FAC algorithm, this function assumes
    the loss function is negative log-likelihood for an exponential family.
    (This is because it relies on the Fisher information matrix approximating
    the Hessian of the NLL.)
    """
    ndir = len(dirs)

    # First, compute the "function space" portion of the quadratic approximation.
    # This is based on the Gauss-Newton approximation to the NLL, or equivalently,
    # the Fisher information matrix.
    
    A_func = onp.zeros((ndir, ndir))
    batch_size = X.shape[0]
    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]

        A_func += compute_A_chunk(arch.net_apply, output_model.nll_fn, arch.unflatten,
                                  w, X_chunk, T_chunk, dirs, grad_w)
    
    A_func /= batch_size
    
    # Now compute the weight space terms, which include both the Hessian of the
    # L2 regularizer and the damping term. This is almost a multiple of the
    # identity matrix, except that the L2 penalty only applies to weights, not
    # biases. Hence, we need to apply a mask to zero out the entries corresponding
    # to biases. This can be done using a Hessian-vector product with the L2
    # regularizer, which has the added benefit that the solution generalizes
    # to non-uniform L2 regularizers as well.
    
    wrap = lambda w: L2_penalty(arch, w)
    Hv = [kfac_util.hvp(wrap, w, v) for v in dirs]
    A_L2 = onp.array([[weight_cost * Hv[i] @ dirs[j]
                       for i in range(ndir)]
                      for j in range(ndir)])
    A_prox = onp.array([[lmbda * dirs[i] @ dirs[j]
                         for i in range(ndir)]
                        for j in range(ndir)])
    A = A_func + A_L2 + A_prox
    
    # The linear term is much simpler: it's just the dot product with the gradient.
    b = onp.array([v @ grad_w for v in dirs])
    
    # Minimize the quadratic approximation by solving the linear system.
    coeffs = onp.linalg.solve(A, -b)
    
    # The decrease in the quadratic objective is used to adapt lambda.
    quad_decrease = -0.5 * coeffs @ A @ coeffs - b @ coeffs
    
    return coeffs, quad_decrease

def compute_update(coeffs, dirs):
    ans = 0
    for coeff, v in zip(coeffs, dirs):
        ans = ans + coeff * v
    return ans
    
def update_gamma(state, arch, output_model, X, T, config):
    curr_gamma = state['gamma']
    gamma_less = onp.maximum(
        curr_gamma * config['gamma_drop']**config['gamma_update_interval'],
        config['gamma_min'])
    gamma_more = onp.minimum(
        curr_gamma * config['gamma_boost']**config['gamma_update_interval'],
        config['gamma_min'])
    gammas = [gamma_less, curr_gamma, gamma_more]
    
    grad_w = compute_gradient(
        arch, output_model, state['w'], X, T, config['weight_cost'],
        config['chunk_size'])
    
    results = []
    for gamma in gammas:
        natgrad_w = compute_natgrad_from_eigs(
            arch, grad_w, state['A_eig'], state['G_eig'], state['pi'], gamma)
        
        prev_update = state['update']
        coeffs, _ = compute_step_coeffs(
            arch, output_model, state['w'], X, T, [-natgrad_w, prev_update],
            grad_w, config['weight_cost'], state['lambda'], config['chunk_size'])
        update = compute_update(coeffs, [-natgrad_w, prev_update])
        new_w = state['w'] + update
        
        results.append(compute_cost(
            arch, output_model, new_w, X, T, config['weight_cost'],
            config['chunk_size']))
        
    best_idx = onp.argmin(results)
    return gammas[best_idx]

def update_lambda(arch, output_model, lmbda, old_w, new_w, X, T, quad_dec, config):
    old_cost = compute_cost(
        arch, output_model, old_w, X, T, config['weight_cost'], config['chunk_size'])
    new_cost = compute_cost(
        arch, output_model, new_w, X, T, config['weight_cost'], config['chunk_size'])
    rho = (old_cost - new_cost) / quad_dec
    
    if np.isnan(rho) or rho < 0.25:
        new_lambda = np.minimum(
            lmbda * config['lambda_boost']**config['lambda_update_interval'],
            config['lambda_max'])
    elif rho > 0.75:
        new_lambda = np.maximum(
            lmbda * config['lambda_drop']**config['lambda_update_interval'],
            config['lambda_min'])
    else:
        new_lambda = lmbda
    
    return new_lambda, rho
    

def kfac_init(arch, output_model, X_train, T_train, config, random_seed=0):
    state = {}
    
    state['step'] = 0
    state['rng'] = random.PRNGKey(random_seed)
    
    state['gamma'] = config['init_gamma']
    state['lambda'] = config['init_lambda']
    
    key, state['rng'] = random.split(state['rng'])
    _, params = arch.net_init(key, X_train.shape)
    state['w'] = arch.flatten(params)
    state['w_avg'] = state['w']
    
    key, state['rng'] = random.split(state['rng'])
    state['A'], state['G'] = estimate_covariances(
        arch, output_model, state['w'], X_train, key, config['chunk_size'])
    
    state['A_eig'], state['G_eig'], state['pi'] = compute_eigs(
        arch, state['A'], state['G'])
    
    return state

        
def kfac_iter(state, arch, output_model, X_train, T_train, config):
    old_state = state
    state = dict(state)  # shallow copy
    
    state['step'] += 1
    
    ndata = X_train.shape[0]
    batch_size = get_batch_size(state['step'], ndata, config)
    
    # Sample with replacement
    key, state['rng'] = random.split(state['rng'])
    idxs = random.permutation(key, np.arange(ndata))[:batch_size]
    X_batch, T_batch = X_train[idxs, :], T_train[idxs, :]

    # Update statistics by running backprop on the sampled targets
    if state['step'] % config['cov_update_interval'] == 0:
        batch_size_samp = get_sample_batch_size(batch_size, config)
        X_samp = X_batch[:batch_size_samp, :]
        state['A'], state['G'] = update_covariances(
            state['A'], state['G'], arch, output_model, state['w'], X_samp, state['rng'],
            config['cov_timescale'], config['chunk_size'])
        
    # Update the inverses
    if state['step'] % config['eig_update_interval'] == 0:
        state['A_eig'], state['G_eig'], state['pi'] = compute_eigs(
            arch, state['A'], state['G'])
        
    # Update gamma
    if state['step'] % config['gamma_update_interval'] == 0:
        state['gamma'] = update_gamma(state, arch, output_model, X_batch, T_batch, config)

    # Compute the gradient and approximate natural gradient
    grad_w = compute_gradient(
        arch, output_model, state['w'], X_batch, T_batch,
        config['weight_cost'], config['chunk_size'])
    natgrad_w = compute_natgrad_from_eigs(
        arch, grad_w, state['A_eig'], state['G_eig'], state['pi'], state['gamma'])

    # Determine the step size parameters using MVPs
    if 'update' in state:
        prev_update = state['update']
        dirs = [-natgrad_w, prev_update]
    else:
        dirs = [-natgrad_w]
    state['coeffs'], state['quad_dec'] = compute_step_coeffs(
        arch, output_model, state['w'], X_batch, T_batch, dirs,
        grad_w, config['weight_cost'], state['lambda'], config['chunk_size'])
    state['update'] = compute_update(state['coeffs'], dirs)
    state['w'] = state['w'] + state['update']
    
    # Update lambda
    if state['step'] % config['lambda_update_interval'] == 0:
        state['lambda'], state['rho'] = update_lambda(
            arch, output_model, state['lambda'], old_state['w'], state['w'], X_batch,
            T_batch, state['quad_dec'], config)
        
    # Iterate averaging
    ema_param = kfac_util.get_ema_param(config['param_timescale'])
    state['w_avg'] = ema_param * state['w_avg'] + (1-ema_param) * state['w']
    
    return state




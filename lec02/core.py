from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, jvp, vjp
import scipy.sparse


def hvp(J, w, v):
    return jvp(grad(J), (w,), (v,))[1]

def gnhvp(f, L, w, v):
    y, R_y = jvp(f, (w,), (v,))
    R_gy = hvp(L, y, R_y)
    _, f_vjp = vjp(f, w)
    return f_vjp(R_gy)[0]

def approx_solve(A_mvp, b, niter):
    dim = b.size
    A_linop = scipy.sparse.linalg.LinearOperator((dim,dim), matvec=A_mvp)
    res = scipy.sparse.linalg.cg(A_linop, b, maxiter=niter)
    return res[0]




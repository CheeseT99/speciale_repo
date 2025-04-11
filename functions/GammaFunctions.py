import numpy as np
from numpy import log
from scipy.special import gammaln
from scipy.special import multigammaln
import numba as nb
from numba  import njit

@njit
def gammaln_nr(z):
    """Numerical Recipes 6.1"""
    #Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array([
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    .339946499848118887e-4, .465236289270485756e-4,
    -.983744753048795646e-4, .158088703224912494e-3,
    -.210264441724104883e-3, .217439618115212643e-3,
    -.164318106536763890e-3, .844182239838527433e-4,
    -.261908384015814087e-4, .368991826595316234e-5])

    out=np.empty(z.shape[0])


    for i in range(z.shape[0]):
      y = z[i]
      tmp = z[i] + 5.24218750000000000
      tmp = (z[i] + 0.5) * np.log(tmp) - tmp
      ser = 0.999999999999997092

      n = coefs.shape[0]
      for j in range(n):
          y = y + 1.
          ser = ser + coefs[j] / y

      out[i] = tmp + log(2.5066282746310005 * ser / z[i])
    return out
@njit
def gammaln_nr_p(z):
    """Numerical Recipes 6.1"""
    #Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array([
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    .339946499848118887e-4, .465236289270485756e-4,
    -.983744753048795646e-4, .158088703224912494e-3,
    -.210264441724104883e-3, .217439618115212643e-3,
    -.164318106536763890e-3, .844182239838527433e-4,
    -.261908384015814087e-4, .368991826595316234e-5])

    out=np.empty(z.shape[0])


    for i in nb.prange(z.shape[0]):
      y = z[i]
      tmp = z[i] + 5.24218750000000000
      tmp = (z[i] + 0.5) * np.log(tmp) - tmp
      ser = 0.999999999999997092

      n = coefs.shape[0]
      for j in range(n):
          y = y + 1.
          ser = ser + coefs[j] / y

      out[i] = tmp + log(2.5066282746310005 * ser / z[i])
    return out
@njit
def multigammalnNumba(a, d):
    # Original code was from scipy.special import multigammaln

    r"""Returns the log of multivariate gamma, also sometimes called the
    generalized gamma.

    Parameters
    ----------
    a : int
        The multivariate gamma is computed for each item of `a`.
    d : int
        The dimension of the space of integration.

    Returns
    -------
    res : float
        The value of the log multivariate gamma at the given points `a`.

    Notes
    -----
    The formal definition of the multivariate gamma of dimension d for a real
    `a` is

    .. math::

        \Gamma_d(a) = \int_{A>0} e^{-tr(A)} |A|^{a - (d+1)/2} dA

    with the condition :math:`a > (d-1)/2`, and :math:`A > 0` being the set of
    all the positive definite matrices of dimension `d`.  Note that `a` is a
    scalar: the integrand only is multivariate, the argument is not (the
    function is defined over a subset of the real set).

    This can be proven to be equal to the much friendlier equation

    .. math::

        \Gamma_d(a) = \pi^{d(d-1)/4} \prod_{i=1}^{d} \Gamma(a - (i-1)/2).

    References
    ----------
    R. J. Muirhead, Aspects of multivariate statistical theory (Wiley Series in
    probability and mathematical statistics).

    """


    if 2*a <= 1.0*(d - 1):
        with nb.objmode():
            print("a= %f" %(a))
            print("d= %i" %(d))

        raise ValueError("condition a (%f) > 0.5 * (d-1) (%f) not met") # + str('a 0.5 * (d-1)'))
    j = np.arange(1,d+1)
    res = (d * (d-1) * 0.25) * np.log(np.pi)
    res += np.sum(gammaln_nr(a - (j - 1.)/2))
    return res


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time
    import sys

    a=858.5
    a=50.5
    d=14
    res_BM =multigammaln(a, d)
    print(res_BM)
    res_Numba = multigammalnNumba(a, d)
    print(res_Numba)

    assert (np.allclose(res_BM,res_Numba))

    sys.exit()

#     n_trials = 8
#     scipy_times = np.zeros(n_trials)
#     fastats_times = np.zeros(n_trials)
#     fastats_times_p = np.zeros(n_trials)
#
#     for i in range(n_trials):
#         zs = np.linspace(0.001, 100, 10**i) # evaluate gammaln over this range
#
#     # dont take first timing - this is just compilation
#     start = time.time()
#     arr_1=gammaln_nr(zs)
#     end = time.time()
#
#     start = time.time()
#     arr_1=gammaln_nr(zs)
#     end = time.time()
#     fastats_times[i] = end - start
#
#     start = time.time()
#     arr_3=gammaln_nr_p(zs)
#     end = time.time()
#     fastats_times_p[i] = end - start
#     start = time.time()
#
#     start = time.time()
#     arr_3=gammaln_nr_p(zs)
#     end = time.time()
#     fastats_times_p[i] = end - start
#     start = time.time()
#
#     arr_2=gammaln(zs)
#     end = time.time()
#     scipy_times[i] = end - start
#     print(np.allclose(arr_1,arr_2))
#     print(np.allclose(arr_1,arr_3))
#
# fig, ax = plt.subplots(figsize=(12,8))
# plt.plot(np.logspace(0, n_trials-1, n_trials), fastats_times, label="numba");
# plt.plot(np.logspace(0, n_trials-1, n_trials), fastats_times_p, label="numba_parallel");
# plt.plot(np.logspace(0, n_trials-1, n_trials), scipy_times, label="scipy");
# ax.set(xscale="log");
# ax.set_xlabel("Array Size", fontsize=15);
# ax.set_ylabel("Execution Time (s)", fontsize=15);
# ax.set_title("Execution Time of Log Gamma");
# plt.legend()
# fig.show()
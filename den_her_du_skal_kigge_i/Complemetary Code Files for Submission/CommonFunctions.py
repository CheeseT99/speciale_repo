import numpy as np
from scipy.special import gamma, factorial
import numpy.linalg as LA
import scipy.linalg as SPLA
import cvxopt

# This function returns the all combinations of exclusion or inclusion of N variables. 2^N options.
def constructAllCombinationsMatrix(N):

    TotComb = pow(2 , N)

    A = np.zeros((TotComb , N))

    for c in np.arange(0 , TotComb):
        for v in np.arange(0 , N):
            A[c,v] = (c & pow(2 , v)) / pow(2 , v)

    return A

# This function returns the m row of the all combinations of exclusion or inclusion of N variables. 2^N options.
# The m row of the matrix in the fuction constructAllCombinationsMatrix.  This is done in order to save memory space
# when N is very big.
def retreiveRowFromAllCombinationsMatrix(m , N):

    A = np.ascontiguousarray(np.zeros((N,), dtype=int))

    c = m
    for v in np.arange(0, N):
        p2v  = pow(2, v)
        A[v] = (c & p2v) / p2v

    return A

# ========== End of function retreiveRowFromAllCombinationsMatrix ==========

def retreiveRowFromAllCombinationsMatrixNumba(m , N):

    A = np.ascontiguousarray(np.zeros((N,), dtype=np.int64))

    c = m
    for v in np.arange(0, N):
        p2v  = pow(2, v)
        A[v] = (c & p2v) / p2v

    return A

# ========== End of function retreiveRowFromAllCombinationsMatrixNumba ==========

# this function is similar to np.tile. Duplicating a numpy array v, k times.
def tileNumba(v, k):
    m=len(v)
    y = np.ascontiguousarray(np.zeros(m*k, dtype=np.int64))

    for i in np.arange(0,k):
        y[i*m:(i+1)*m]=v

    return y

# ========== End of function tileNumba ==========

# this function is similar to np.tile. Duplicating a numpy array v, k times.
def MeanAlongAxisNumba(a,axis):

    assert axis==0 or axis==1
    mean = np.ascontiguousarray(np.zeros((a.shape[1-axis],1), dtype=np.float64))
    if axis == 1:
        a=np.transpose(a)

    L = a.shape[0]
    for i in np.arange(0,L):
        mean[:,0] += a[i,:]

    mean /= L

    return mean

# ========== End of function MeanAlongFirstIndex ==========

def myInverseMatrix(A, Error_str):

    try:
        AInv = LA.pinv(A)
    except np.linalg.LinAlgError as e:
        print('Error %s ' + Error_str)
        string = 'Trying normal inverse ... '
        try:
            AInv = LA.inv(A)
            print(string + 'LA.inv worked')
        except:
            string += 'Failed trying scipy pinvh'
            try:
                AInv = SPLA.pinvh(A)
                print(string + 'SPLA.pinvh worked')
            except:
                print(string + 'Failed ')

    return AInv


def printFactorsProbabilities(factorsNames, factorsProbability):
    print('Probabilities of factors')
    print(factorsNames)
    print(factorsProbability)

    print('Sorted Probabilities of factors')
    I = np.argsort(-factorsProbability)
    print(factorsNames[I])
    print(factorsProbability[I])

    return

def printPredictorsProbabilities(predictorsNames, predictorsProbability):
    print('Probabilities of Predictors')
    print(predictorsNames)
    print(predictorsProbability)

    print('Sorted Probabilities of Predictors')
    I = np.argsort(-predictorsProbability)
    print(predictorsNames[I])
    print(predictorsProbability[I])

    return


def printFactorsAndPredictorsProbabilities(factorsNames, factorsProbability, predictorsNames, predictorsProbability):

    printFactorsProbabilities(factorsNames, factorsProbability)

    printPredictorsProbabilities(predictorsNames, predictorsProbability)

    return


# A wrap to solve a quadrtic programing taken from:
# https://scaron.info/blog/quadratic-programming-in-python.html
# https://cvxopt.org/userguide/coneprog.html
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

# Compute logarithm multivariate Gamma function.
# Gamma_p(x) = pi^(p(p-1)/4) prod_(j=1)^p Gamma(x+(1-j)/2)
# log Gamma_p(x) = p(p-1)/4 log pi + sum_(j=1)^p log Gamma(x+(1-j)/2)
# d=p;
# Written by Michael Chen (sth4nth@gmail.com). Originaly in Matlab.
# downloaded from:
# https://github.com/areslp/matlab/blob/master/vbgm/logmvgamma.m

# def y = logmvgamma(x,d)
#
#     s = size(x);
#     x = reshape(x,1,prod(s));
#     x = bsxfun(@plus,repmat(x,d,1),(1-(1:d)')/2);
#     y = d*(d-1)/4*log(pi)+sum(gammaln(x),1);
#     y = reshape(y,s);
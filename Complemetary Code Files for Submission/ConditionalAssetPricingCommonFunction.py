import numpy as np
import numpy.linalg as LA
import sys
from scipy.special import multigammaln

from GammaFunctions import multigammalnNumba

def findOtherFactors(KMax, factorsIndicesIncludedInModel):
    otherFactors = np.zeros(KMax-len(factorsIndicesIncludedInModel), dtype=np.int64)
    nFactorsInmodel = len(factorsIndicesIncludedInModel)
    j = 0
    ii = 0
    for i in np.arange(0,KMax):
        if j < nFactorsInmodel and i == factorsIndicesIncludedInModel[j]:
            j += 1
        else:
            otherFactors[ii] = i
            ii += 1

    return otherFactors

# ========== End of function findOtherFactors ==========

# This function calculates the log marginal likelihood given the regression estimates.
def calclogMarginalLikelihoods(N, K, M, T, T0, Tstar, S0, Sr, SrR, Sf0, Sf):
    key_use_Numba = 3 # 0- No Numba  1 - Numba or >1 - calculate both compare and return no numba if 0

    if key_use_Numba > 0:
        logMarginalLikelihoodNumba = -T * (N + K) / 2 * np.log(np.pi) \
                                + (K * (M + 1) + N * (1 + M + K + K * M)) / 2 * np.log(T0 / Tstar) \
                                + multigammalnNumba((Tstar - (K + 1) * M - 1) / 2, N) \
                                - multigammalnNumba((T0 - (K + 1) * M - 1) / 2, N) \
                                + multigammalnNumba((Tstar + N - M - 1) / 2, K) \
                                - multigammalnNumba((T0 + N - M - 1) / 2, K) \
                                + (T0 - (K + 1) * M - 1) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                - (Tstar - (K + 1) * M - 1) / 2 * (np.log(LA.det(Sr / Tstar)) + len(Sr) * np.log(Tstar)) \
                                + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                - (Tstar + N - M - 1) / 2 * (np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

        logMarginalLikelihoodRNumba = -T * (N + K) / 2 * np.log(np.pi) \
                                 + (K * (M + 1) + N * (K + K * M)) / 2 * np.log(T0 / Tstar) \
                                 + multigammalnNumba((Tstar - K * M) / 2, N) \
                                 - multigammalnNumba((T0 - K * M) / 2, N) \
                                 + multigammalnNumba((Tstar + N - M - 1) / 2, K) \
                                 - multigammalnNumba((T0 + N - M - 1) / 2, K) \
                                 + (T0 - K * M) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                 - (Tstar - K * M) / 2 * (np.log(LA.det(SrR / Tstar)) + len(SrR) * np.log(Tstar)) \
                                 + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                 - (Tstar + N - M - 1) / 2 * (np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

    if key_use_Numba==0 or key_use_Numba > 1:
        logMarginalLikelihood = -T * (N + K) / 2 * np.log(np.pi) \
                                + (K * (M + 1) + N * (1 + M + K + K * M)) / 2 * np.log(T0 / Tstar) \
                                + multigammaln((Tstar - (K + 1) * M - 1) / 2, N) \
                                - multigammaln((T0 - (K + 1) * M - 1) / 2, N) \
                                + multigammaln((Tstar + N - M - 1) / 2, K) \
                                - multigammaln((T0 + N - M - 1) / 2, K) \
                                + (T0 - (K + 1) * M - 1) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                - (Tstar - (K + 1) * M - 1) / 2 * (np.log(LA.det(Sr / Tstar)) + len(Sr) * np.log(Tstar)) \
                                + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                - (Tstar + N - M - 1) / 2 * (np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

        logMarginalLikelihoodR = -T * (N + K) / 2 * np.log(np.pi) \
                                 + (K * (M + 1) + N * (K + K * M)) / 2 * np.log(T0 / Tstar) \
                                 + multigammaln((Tstar - K * M) / 2, N) \
                                 - multigammaln((T0 - K * M) / 2, N) \
                                 + multigammaln((Tstar + N - M - 1) / 2, K) \
                                 - multigammaln((T0 + N - M - 1) / 2, K) \
                                 + (T0 - K * M) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                 - (Tstar - K * M) / 2 * (np.log(LA.det(SrR / Tstar)) + len(SrR) * np.log(Tstar)) \
                                 + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                 - (Tstar + N - M - 1) / 2 * (np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

    if key_use_Numba > 1:

        if not np.allclose(logMarginalLikelihoodNumba, logMarginalLikelihood):
            print('Mis match in calclogMarginalLikelihoods === Unrestricted')
            a=[]
            a.append(multigammaln((Tstar - (K + 1) * M - 1) / 2, N))
            a.append(- multigammaln((T0 - (K + 1) * M - 1) / 2, N))
            a.append(multigammaln((Tstar + N - M - 1) / 2, K))
            a.append(- multigammaln((T0 + N - M - 1) / 2, K))

            b=[]
            b.append(multigammalnNumba((Tstar - (K + 1) * M - 1) / 2, N))
            b.append(- multigammalnNumba((T0 - (K + 1) * M - 1) / 2, N))
            b.append(multigammalnNumba((Tstar + N - M - 1) / 2, K))
            b.append(- multigammalnNumba((T0 + N - M - 1) / 2, K))

            print(a)
            print(b)
            arg = (Tstar - (K + 1) * M - 1) / 2
            print(arg)
            print(N)
            print(multigammaln(arg,N))
            print(multigammalnNumba(arg, N))

            sys.exit()

        if not np.allclose(logMarginalLikelihoodRNumba, logMarginalLikelihoodR):
            print('Mis match in calclogMarginalLikelihoods === Restricted')
            print(multigammaln((Tstar - K * M) / 2, N))
            print(multigammalnNumba((Tstar - K * M) / 2, N))

            sys.exit()

    if key_use_Numba > 0:
        return logMarginalLikelihoodNumba, logMarginalLikelihoodRNumba
    else:
        return logMarginalLikelihood, logMarginalLikelihoodR

# ========== End of function calclogMarginalLikelihoods ==========
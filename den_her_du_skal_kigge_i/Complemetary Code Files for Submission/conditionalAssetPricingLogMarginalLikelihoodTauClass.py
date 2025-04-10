# This function calculates the log marginal likelihood of the conditional
# asset pricing models.(appendix B in the paper)
import sys
import pandas as pd
import numpy as np
import copy
import numpy.linalg as LA
from scipy.special import multigammaln
import pickle
from enum import Enum
import matplotlib as mpl
# use the following line when running without X window.
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib._color_data as mcd
import scipy.linalg as SPLA
import os
from cvxopt import matrix, solvers
import statsmodels.api as sm
import scipy

from CommonFunctions import constructAllCombinationsMatrix, retreiveRowFromAllCombinationsMatrix, \
    retreiveRowFromAllCombinationsMatrixNumba, printFactorsAndPredictorsProbabilities, tileNumba, \
    MeanAlongAxisNumba, cvxopt_solve_qp
from GammaFunctions import multigammalnNumba
from ConditionalAssetPricingCommonFunction import calclogMarginalLikelihoods, findOtherFactors
import tictoc
import writeProfLatexTable as tex

keyPrint        = False
keyPrintResults = True
keyDebug        = False
key_Avoid_duplicate_factors = False
key_Avoid_duplicate_predictors = False
key_use_combination_matrix  = False

def retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax):
    combinationsRowFromMatrix = retreiveRowFromAllCombinationsMatrixNumba(model, KMax+MMax)
    factorsIncludedInModel    = combinationsRowFromMatrix[0: KMax]
    predictorsIncludedInModel = combinationsRowFromMatrix[KMax:]

    return factorsIncludedInModel, predictorsIncludedInModel

# ========== End of retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix ==========
# ======================================================================================================================
def setUncontionalFilter(KMax, MMax):
    nModelsMax = pow(2, KMax + MMax)
    filter = np.zeros((nModelsMax,), dtype=np.float64)
    for model in np.arange(0, nModelsMax):
        factorsIncludedInModel, predictorsIncludedInModel = \
            retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax)

        if np.sum(predictorsIncludedInModel) == 0:
            filter[model] = 1
        else:
            filter[model] = np.inf

    return filter

# ========== End of setUncontionalFilter ==========
# ======================================================================================================================
def modelsProbabilities(MarginalLikelihoodU,MarginalLikelihoodR, models_factors_included, KMax, MMax):
    nModelsMax = pow(2, KMax + MMax)
    nModels = models_factors_included.shape[0]
    assert KMax == models_factors_included.shape[1]
    # The columns are unrestricted conditional, unrestricted unconditional, restricted conditional, restricted unconditional.
    modelsProbabilities = np.zeros((nModels,4),np.float64)
    modelsProbabilitiesCounter = np.zeros((nModels,4),np.int64)

    for model in np.arange(0, nModelsMax):
        factorsIncludedInModel, predictorsIncludedInModel = \
            retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax)

        for i in np.arange(0,nModels):
            if np.allclose(models_factors_included[i,:],factorsIncludedInModel):
                modelsProbabilities[i,0] += MarginalLikelihoodU[model]
                modelsProbabilities[i,2] += MarginalLikelihoodR[model]
                modelsProbabilitiesCounter[i,0] +=1
                modelsProbabilitiesCounter[i,2] +=1

                if np.sum(predictorsIncludedInModel) == 0:
                    modelsProbabilities[i,1] += MarginalLikelihoodU[model]
                    modelsProbabilities[i,3] += MarginalLikelihoodR[model]
                    modelsProbabilitiesCounter[i,1] +=1
                    modelsProbabilitiesCounter[i,3] +=1

    assert np.sum(modelsProbabilitiesCounter[:,1]) == nModels and np.sum(modelsProbabilitiesCounter[:,3]) == nModels
    assert np.sum(modelsProbabilitiesCounter[:,0]) == nModels*pow(2,MMax) and \
        np.sum(modelsProbabilitiesCounter[:,2]) == nModels*pow(2,MMax)

    return modelsProbabilities

# ========== End of setUncontionalFilter ==========
# ======================================================================================================================
def calculateFactorsAndPredictorsProbabilities(MarginalLikelihood, KMax , MMax):
    factorsProbability    = np.zeros((KMax,), dtype=np.float64)
    predictorsProbability = np.zeros((MMax,), dtype=np.float64)
    combinationsRowFromMatrix = np.zeros((KMax + MMax,) , dtype=np.int64)

    nModelsMax = pow(2, KMax+MMax)
    for model in np.arange(0, nModelsMax):
        combinationsRowFromMatrix = retreiveRowFromAllCombinationsMatrixNumba(model, KMax + MMax)

        factorsProbability += MarginalLikelihood[model] * combinationsRowFromMatrix[0:KMax]
        predictorsProbability += MarginalLikelihood[model] * combinationsRowFromMatrix[KMax:]

    return factorsProbability, predictorsProbability

# ========== End of calculateFactorsAndPredictorsProbabilities ==========
# ======================================================================================================================
def calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihood, ntopModels, KMax, MMax,        \
                                                                 factorsNames, predictorsNames,                        \
                                                                 factorsInModel=None, predictorsInModel=None, keyPrintResults=True):
    MarginalLikelihood = np.exp(logMarginalLikelihood - max(logMarginalLikelihood)) # [np.argwhere(logMarginalLikelihood != -np.inf)]))

    MarginalLikelihood = MarginalLikelihood / np.sum(MarginalLikelihood)

    KMaxPlusMMax = KMax + MMax

    #nModelsMax = pow(2 , KMaxPlusMMax)

    local_key_use_combination_matrix = (not type(factorsInModel) == type(None))

    if local_key_use_combination_matrix:
        factorsProbability = MarginalLikelihood @ factorsInModel          # Probability of a factor to appear in all models.
        predictorsProbability = MarginalLikelihood @ predictorsInModel # Probability of a predictor to appear in all models.
    else:
        #factorsProbability = np.zeros((KMax,)    , dtype=float)
        #predictorsProbability = np.zeros((MMax,) , dtype=float)
        #for model in np.arange(0, nModelsMax):
        #    factorsIncludedInModel, predictorsIncludedInModel = \
        #        retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax)
        #    factorsProbability += MarginalLikelihood[model] * factorsIncludedInModel
        #    predictorsProbability += MarginalLikelihood[model] * predictorsIncludedInModel

        factorsProbability , predictorsProbability = calculateFactorsAndPredictorsProbabilities(MarginalLikelihood, KMax, MMax)
        #print('**** assert in calculateMarginalLikelihoodAndFactorsPredictorsProbabilities **** ')
        #assert np.allclose(factorsProbabilityNumba, factorsProbability) and np.allclose(predictorsProbabilityNumba, predictorsProbability)

    if keyPrintResults:
        I = np.argsort(-MarginalLikelihood) # Adding minus in order to sort in such a way the the first index is the maximum.

        print('Top %d probable models' %(ntopModels))
        print(MarginalLikelihood[I[0: ntopModels]])                             # First ntopModels most probable models.
        print('Top probable models factors')
        for i in np.arange(0, ntopModels):
            model = I[i]
            if local_key_use_combination_matrix:
                factorsIncludedInModel  = factorsInModel[model, :]
            else:
                [factorsIncludedInModel, predictorsIncludedInModel] = \
                    retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax)

            print(np.argwhere(factorsIncludedInModel == 1).flatten())  # First ntopModels most probable models.

        print('Top probable models predictors')
        for i in np.arange(0, ntopModels):
            model = I[i]
            if local_key_use_combination_matrix:
                predictorsIncludedInModel = predictorsInModel[model,:]
            else:
                [factorsIncludedInModel, predictorsIncludedInModel] = \
                    retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax)

            print(np.argwhere(predictorsIncludedInModel == 1).flatten())       # First ntopModels most probable models.

        print('Probabilities of factors')
        print(factorsNames)
        print(factorsProbability)

        print('Sorted Probabilities of factors')
        I = np.argsort(-factorsProbability)
        print(factorsNames[I])
        print(factorsProbability[I])

        print('Probabilities of Predictors')
        print(predictorsNames)
        print(predictorsProbability)

    return [MarginalLikelihood, factorsProbability, predictorsProbability]

# ========== End of calculateMarginalLikelihoodAndFactorsPredictorsProbabilities ==========
# ======================================================================================================================
def conditionalAssetPricingLogMarginalLikelihoodTau(rr, ff, zz, significantPredictors, Tau):

    # Constants.
    ntopModels = 10
    keyConditionalAPM = int(len(significantPredictors)>0)

    print("calculating both unrestricted and restricted models")
    print("key_Avoid_duplicate_factors= %d, key_Avoid_duplicate_predictors= %d, key_use_combination_matrix= %d " \
            % (key_Avoid_duplicate_factors, key_Avoid_duplicate_predictors, key_use_combination_matrix))
    print("Tau= %f "  %( Tau))

    # Moving on to calculating the marginal likelihood.
    # Trying to work with nupy arrays instead of pandas dataframe.
    factorsNames = ff.columns.drop('Date')
    FOrig = copy.deepcopy(ff.loc[:,factorsNames].values)
    KMax = len(factorsNames)

    # Market squared Sharpe ratio and maximum difference between the upper bound Sharpe ratio and the market.
    # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
    SR2Mkt  = pow(np.mean(ff.loc[keyConditionalAPM:,'Mkt-RF'].values) /
                  np.std(ff.loc[keyConditionalAPM:,'Mkt-RF'].values) , 2)
    Tau2m1SR2Mkt = (pow(Tau , 2) - 1) * SR2Mkt

    predictorsNames = zz.columns.drop('Date')
    discardPredictors = np.array(list(set(np.arange(0,len(predictorsNames))) - set(significantPredictors)))
    if len(discardPredictors) > 0:
        ZOrig = copy.deepcopy(zz.drop(columns=predictorsNames[discardPredictors]))
    else:
        ZOrig = copy.deepcopy(zz)

    predictorsNames = ZOrig.columns.drop('Date')
    ZOrig = ZOrig.loc[:,predictorsNames].values

    #
    print('ZOrig mean and std')
    print(np.mean(ZOrig,0))
    print(np.std(ZOrig, 0))

    MMax  = len(predictorsNames)

    KMaxPlusMMax = KMax + MMax

    ROrig = copy.deepcopy(rr)
    if not ROrig.empty:
        T = ROrig.shape[0]; T -= keyConditionalAPM                                # Minus one because of predictability.
        testAssetsPortfoliosNames = ROrig.columns.drop('Date')
        ROrig = ROrig.values
    else:
        T = ff.shape[0]; T -= keyConditionalAPM

    print('KMax= %d, MMax= %d, T= %d, keyConditionalAPM=%d' %(KMax , MMax, T, keyConditionalAPM))

    # Calculate Omega full here in order to save the calculations in the loop. Watch the t+1 in F for the interactions
    # between the predictors and factors.
    OmegaOrig = np.zeros((T, KMax * MMax), dtype=float); OmegaOrig.fill(np.nan)
    for t in np.arange(0, T):
        OmegaOrig[t, :] = np.kron(np.identity(KMax), ZOrig[t, :].reshape(-1, 1)) @ FOrig[t+1, :]

    assert not np.isnan(OmegaOrig).any()
    # In order to omit duplication of following factors' pairs:
    if keyPrint:
        # ME < --> SMB factor  # 8 and 2
        print(factorsNames[8-1] + ' <--> ' + factorsNames[2-1])
        # RMW < --> ROE    factor  # 4 and 10
        print(factorsNames[4-1] + ' <--> ' + factorsNames[10-1])
        # CMA < --> IA factor  # 5 and 9
        print(factorsNames[5-1] + ' <--> ' + factorsNames[9-1])

    nModelsMax = pow(2, KMax + MMax)

    if key_use_combination_matrix:
        allCombinationsMatrix = constructAllCombinationsMatrix(KMax + MMax)

        factorsInModel        = allCombinationsMatrix[: , 0 : KMax]
        predictorsInModel     = allCombinationsMatrix[: , KMax :]

        assert factorsInModel.shape == (nModelsMax, KMax) and predictorsInModel.shape == (nModelsMax, MMax)
    else:
        factorsInModel = None
        predictorsInModel = None

    # Variables initialization.
    logMarginalLikelihood  = np.zeros((nModelsMax,), dtype=float); logMarginalLikelihood.fill(-np.inf)
    # Placeholder for the restricted models.
    logMarginalLikelihoodR = np.zeros((nModelsMax,), dtype=float); logMarginalLikelihoodR.fill(-np.inf)

    nTooSmallT0  = 0
    nLegitModels = 0
    T0Total = 0; T0Max = 0; T0Min = np.inf

    AllFactorsSet    = set(np.arange(0 , KMax))
    iotaT = np.ones((T,1), dtype=float)

    totalTime = 0.0
    nprintFraction = pow(10, 1 + (KMaxPlusMMax > 20))
    tictoc.tic()
    mStart = 0
    if keyDebug:
        mStart = int(nModelsMax*3/4*.98)
        mStart = 102050547

    print('First model is %i. Total numbers of models is %i. nprintFraction= %d' %(mStart , nModelsMax, nprintFraction))
    for model in np.arange(mStart , nModelsMax):
        if model % np.floor(nModelsMax/nprintFraction) == 0:
            totalTime += tictoc.toc(False)
            print('Done %3d %% of total work at %12.2f sec. model= %11i, nModelsMax= %11i'
                  %(100*model/nModelsMax , totalTime, model, nModelsMax))
            # Dump
            with open('local_dump.pkl','bw') as file:
                pickle.dump(
                    [model, nModelsMax, KMax, MMax, factorsNames, predictorsNames, \
                     significantPredictors, Tau, \
                     nTooSmallT0 , nLegitModels , T0Max, T0Total, logMarginalLikelihood, logMarginalLikelihoodR],file)


        if not key_use_combination_matrix:
            combinationsRowFromMatrix        = retreiveRowFromAllCombinationsMatrixNumba(model, KMaxPlusMMax)
            factorsIndicesIncludedInModel    = np.argwhere(combinationsRowFromMatrix[0 : KMax] == 1).flatten()
            predictorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[KMax :]   == 1).flatten()
        else:
            factorsIndicesIncludedInModel = np.argwhere(factorsInModel[model,:] == 1).flatten()
            predictorsIndicesIncludedInModel = np.argwhere(predictorsInModel[model,:] == 1).flatten()

        otherFactors = np.array(list(AllFactorsSet - set(factorsIndicesIncludedInModel)) , dtype=int)

        # MKT is not in model assign a 0 probability. Total number of combinations is 2^(KMax-1)*2^MMax.
        # However in case of all factors in the test assets continue as linear predictive regression.
        # Total number of combinations is 2^MMax.
        if not (1-1 in factorsIndicesIncludedInModel) and len(factorsIndicesIncludedInModel) != 0:
            #logMarginalLikelihood[model] = -np.inf
            continue

        if key_Avoid_duplicate_predictors:
            # Add restriction when there is a linear dependency between the predictors.
            # Not including the following three predictors together: dp, ep, de.
            # predictors 1, 3, 4 out of the 8 combinations of including the
            # predictors only 5 are independent:
            # 1 - none of them, 3 - only one predictor and 1 - one pair out of the three
            # possible pairs.
            # and the tbl, lty, tms. predictors 8, 9, 11

            if (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel and 4-1 in predictorsIndicesIncludedInModel) or \
                (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel) or \
                (1 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel) or \
                (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel) or \
                (9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel):

                # logMarginalLikelihood[model] = -np.inf

                continue

        # Avoid models with duplicate factors.
        if key_Avoid_duplicate_factors:
            if (2 - 1 in factorsIndicesIncludedInModel and 8 - 1 in factorsIndicesIncludedInModel) or    \
                (4 - 1 in factorsIndicesIncludedInModel and 10 - 1 in factorsIndicesIncludedInModel) or  \
                (5 - 1 in factorsIndicesIncludedInModel and 9 - 1 in factorsIndicesIncludedInModel):

                # logMarginalLikelihood[model] = -np.inf

                if keyPrint:
                    print('duplicate factors!!! factorsIndicesIncludedInModel= ')
                    print(factorsIndicesIncludedInModel)

                continue

        # Each model has different N K R and F. Remove the temporary values at the end of each cycle.
        # del N, K, F, R

        if not ROrig.empty:
            R = np.concatenate((ROrig[keyConditionalAPM : , :] , \
                            FOrig[keyConditionalAPM : , otherFactors]),axis=1)
        else:
            R = FOrig[keyConditionalAPM : , otherFactors]

        N = R.shape[1]

        # No test assets. 2^MMax combinations.
        if N == 0:
            #logMarginalLikelihood[model] = -np.inf
            continue

        nLegitModels += 1

        F = FOrig[keyConditionalAPM : , factorsIndicesIncludedInModel]
        K = F.shape[1]
        Z = ZOrig[0 : T, predictorsIndicesIncludedInModel]
        M = Z.shape[1]

        if M == 0:
            Z = np.empty((T, 0), dtype=float)

        assert R.shape[0] == T and F.shape[0] == T and Z.shape[0] == T

        OmegaIndecies = np.repeat(factorsIndicesIncludedInModel * MMax, M) + np.tile(predictorsIndicesIncludedInModel, K)
        X = np.concatenate((iotaT, Z), axis=1)

        if keyDebug:
            Omega = np.zeros((T, K * M), dtype=float)
            for t in np.arange(0 , T):
                Omega[t,:]=np.kron(np.identity(K) , Z[t,:].reshape(-1,1)) @ F[t,:]

            assert np.allclose( Omega ,  OmegaOrig[:,OmegaIndecies])

        Omega = OmegaOrig[:,OmegaIndecies]

        W = np.concatenate((np.concatenate((X , F), axis=1) , Omega) , axis= 1)

        RMean = np.mean(R, 0).reshape(-1,1)
        FMean = np.mean(F, 0).reshape(-1,1)
        ZMean = np.mean(Z, 0).reshape(-1,1)

        Vf = np.cov(F, rowvar=False, bias=True).reshape(K , K)

        # Hypothetical sample quantities.

        beta0  = LA.pinv(np.transpose(F) @ F ) @ np.transpose(F) @ R

        if keyDebug:
            beta01 = LA.lstsq(F , R, rcond=None)[0]                            # Should be the same as beta0, just checking.
            assert np.allclose(beta0, beta01)

            Af0 = LA.pinv(np.transpose(iotaT) @ iotaT) @ np.transpose(iotaT) @ F        # Should equal FMean, just checking.
            assert np.allclose(Af0, np.transpose(FMean))

        Af0     = np.concatenate((FMean , np.zeros((K , M), dtype=float)) , axis=1)
        XtX     = np.transpose(X) @ X
        XtXInv  = LA.pinv(XtX)
        WtW     = np.transpose(W) @ W
        #WtWInv  = LA.lstsq(WtW , np.identity(1 + K, dtype=float), rcond=None)[0]

        try:
            WtWInv = LA.pinv(WtW)
        except np.linalg.LinAlgError as e:
            print('Error %s in model number %i' % (e, model))
            print(factorsIndicesIncludedInModel)
            print(predictorsIndicesIncludedInModel)
            string = 'Trying normal inverse ... '
            try:
                WtWInv = LA.inv(WtW)
                print(string + 'LA.inv worked')
            except:
                string += 'Failed trying scipy pinvh'
                try:
                    WtWInv = SPLA.pinvh(WtW)
                    print(string + 'SPLA.pinvh worked')
                except:
                    print(string + 'Failed ')


        Qw      = np.identity(T, dtype=float) - W @ WtWInv @ np.transpose(W)
        #SigmaRR = (np.transpose(R) @ Qw @ R) / (T - M - K - K*M - 1) / pow(100,2) # changing from percentage to numbers.

        # Since the distribution is known the maximum likelihood estimator is without the bias correction.
        SigmaRR = (np.transpose(R) @ Qw @ R) / (T) / pow(100 , 2)  # changing from percentage to numbers.

        SR2     = np.transpose(FMean) @ LA.pinv(Vf) @ FMean                               # factors Sharpe ratio square.

        T0      = int(np.round((N * (1 + SR2 + M*(1+SR2)))/Tau2m1SR2Mkt))
        # MultiGamma(p(a)) is defined for 2*a > p - 1
        T0LowerBound = max(N + (K + 1)*M, K + M - N)
        if T0 <= T0LowerBound:
            nTooSmallT0 += 1
            if keyPrint:
                print('T0 ( %i ) too small so increasing it to minmum acceptable value ( %i )' %(T0,T0LowerBound + 1))
            T0 = T0LowerBound + 1

        FtF  = np.transpose(F) @ F
        RtR  = np.transpose(R) @ R
        XtF  = np.transpose(X) @ F
        WtR  = np.transpose(W) @ R

        S0    = T0 / T * ( RtR - np.transpose(beta0) @ (y @ beta0))
        Sf0   = T0 * Vf

        Tstar   = T0 + T
        FMeanFMeanZMeant = np.concatenate((FMean, FMean @ np.transpose(ZMean)), axis=1)

        AfTilda = T/Tstar*XtXInv @ ( XtF +                                                              \
                                    T0*np.transpose(FMeanFMeanZMeant))
        Sf = Tstar * (Vf + FMean @ np.transpose(FMean))                                                                \
            -T/Tstar * (T0 * FMeanFMeanZMeant + np.transpose(XtF) ) @     \
             XtXInv @                                                                                                  \
            (T0 * np.transpose(FMeanFMeanZMeant) + XtF)

#        if keyRestricted !=1:
        # Calculating the unrestricted log marginal likelihood.
        phi0 = np.transpose(np.concatenate((
               np.concatenate((np.zeros((N , M+1), dtype=float), np.transpose(beta0)), axis=1) , \
               np.zeros((N, K * M), dtype=float)) , axis=1))

        phiTilda = T / Tstar * WtWInv @ ( WtR +  T0/T * WtW @ phi0)

        RminusWphiTilda = R - W @ phiTilda

        Sr       = S0 + np.transpose(RminusWphiTilda) @ (RminusWphiTilda) + \
                       T0 / T * np.transpose(phiTilda - phi0) @ WtW @ (phiTilda - phi0)
        if keyDebug:
            phiHat = WtWInv @ WtR
            SrHat = np.transpose(R - W @ phiHat) @ (R - W @ phiHat)
            SrNew = S0 + SrHat + np.transpose(phiHat) @ WtW @ phiHat + \
                    T0 / T * np.transpose(phi0) @ WtW @ phi0 - Tstar / T * np.transpose(phiTilda) @ WtW @ phiTilda

            SrNew1 = Tstar / T * (RtR - np.transpose(phiTilda) @ WtW @ phiTilda)

            assert np.allclose(SrNew, Sr) and np.allclose(Sr, SrNew1)

        logMarginalLikelihood[model] =  -T * (N + K) / 2 * np.log(np.pi)                                           \
                                        + ( K*( M + 1 ) + N*(1 + M + K + K*M ) ) / 2 * np.log(T0/Tstar)            \
                                        + multigammaln(( Tstar - ( K + 1 )*M - 1 )/2,N)                            \
                                        - multigammaln(( T0    - ( K + 1 )*M - 1 )/2,N)                            \
                                        + multigammaln(( Tstar + N - M - 1 )/2,K)                                  \
                                        - multigammaln(( T0    + N - M - 1 )/2,K)                                  \
                                        +( T0    - (K + 1) * M - 1 ) / 2 * (np.log(LA.det(S0/T0))    + len(S0)  * np.log(T0))     \
                                        -( Tstar - (K + 1) * M - 1 ) / 2 * (np.log(LA.det(Sr/Tstar)) + len(Sr)  * np.log(Tstar))  \
                                        +( T0    + N - M - 1 ) / 2       * (np.log(LA.det(Sf0/T0))   + len(Sf0) * np.log(T0))     \
                                        -( Tstar + N - M - 1)  / 2       * (np.log(LA.det(Sf/Tstar)) + len(Sf)  * np.log(Tstar))

        #else:
        # Calculating the restricted log marginal likelihood.
        phi0R = np.transpose(np.concatenate((np.transpose(beta0) , np.zeros((N,K*M),dtype=float)), axis=1))
        WR    = np.concatenate((F , Omega) , axis=1)
        WRtWR = np.transpose(WR) @ WR
        try:
            WRtWRInv = LA.pinv(WRtWR)
        except np.linalg.LinAlgError as e:
            print('Error %s in model number %i' %(e, model))
            # print(WRtWR[0,:])
            print(factorsIndicesIncludedInModel)
            print(predictorsIndicesIncludedInModel)
            # logMarginalLikelihoodR[model] = -np.inf
            string = 'Trying normal inverse ... '
            try:
                WRtWRInv = LA.inv(WRtWR)
                print(string + 'LA.inv worked')
            except:
                string += 'Failed trying scipy pinvh'
                try:
                    WRtWRInv = SPLA.pinvh(WRtWR)
                    print(string + 'SPLA.pinvh worked')
                except:
                    print(string + 'Failed assaining NINF')
                    continue


        phiTildaR = T / Tstar * WRtWRInv @ (np.transpose(WR) @ R +  T0/T * WRtWR @ phi0R)

        SrR       = S0 + np.transpose(R - WR @ phiTildaR) @ (R - WR @ phiTildaR) + \
                        T0 / T * np.transpose(phiTildaR - phi0R) @ WRtWR @ (phiTildaR - phi0R)

        if keyDebug:
            phiHatR = WRtWRInv @ np.transpose(WR) @ R
            SrRNew1 = Tstar / T * (RtR - np.transpose(phiTildaR) @ WRtWR @ phiTildaR)
            assert np.allclose(SrR, SrRNew1)


        logMarginalLikelihoodR[model] = -T * (N + K) / 2 * np.log(np.pi)                                            \
                                        + (K * (M + 1) + N * (K + K * M)) / 2 * np.log(T0 / Tstar)                  \
                                        + multigammaln((Tstar - K * M) / 2, N)                                      \
                                        - multigammaln((T0 - K * M) / 2, N)                                         \
                                        + multigammaln((Tstar + N - M - 1) / 2, K)                                  \
                                        - multigammaln((T0 + N - M - 1) / 2, K)                                     \
                                        + (T0 - K * M)        / 2 * (np.log(LA.det(S0 / T0))    + len(S0)  * np.log(T0))    \
                                        - (Tstar - K * M)     / 2 * (np.log(LA.det(SrR / Tstar)) + len(SrR)  * np.log(Tstar)) \
                                        + (T0 + N - M - 1)    / 2 * (np.log(LA.det(Sf0 / T0))   + len(Sf0) * np.log(T0))    \
                                        - (Tstar + N - M - 1) / 2 * (np.log(LA.det(Sf / Tstar)) + len(Sf)  * np.log(Tstar))

        T0Total = T0Total + T0
        T0Max   = max(T0Max, T0)
        T0Min   = min(T0Min, T0)
        del N, K, F, R, Z, M

    tictoc.toc()

    T0IncreasedFraction = nTooSmallT0 / nLegitModels
    T0Avg = T0Total / nLegitModels

    print('All combinations= %d Total number of legit models= %d 2ND count %d '             \
          %(nModelsMax, np.count_nonzero(logMarginalLikelihood != -np.inf), nLegitModels))

    print('# of times T0 was increased= %d T0 Average= %f T0 Max= %f T0 Min= %f' \
          %(nTooSmallT0, T0Avg, T0Max, T0Min))

    print('**** Unrestricted models ****')
    [MarginalLikelihood, factorsProbability, predictorsProbability] = \
        calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihood, ntopModels, KMax, MMax,    \
                                                    factorsNames=factorsNames, predictorsNames=predictorsNames,        \
                                                    factorsInModel=factorsInModel, predictorsInModel=predictorsInModel, keyPrintResults=True)

    print('**** Restricted model ****')
    [MarginalLikelihoodR, factorsProbabilityR, predictorsProbabilityR] = \
        calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihoodR, ntopModels, KMax, MMax,   \
                                                    factorsNames=factorsNames, predictorsNames=predictorsNames,        \
                                                    factorsInModel=factorsInModel, predictorsInModel=predictorsInModel, keyPrintResults=True)

    return (logMarginalLikelihood, factorsNames, factorsProbability, predictorsNames, predictorsProbability, \
            T0IncreasedFraction, T0Max, T0Min, T0Avg, logMarginalLikelihoodR, factorsProbabilityR, predictorsProbabilityR)

# ========== End of conditionalAssetPricingLogMarginalLikelihoodTau ==========
# ======================================================================================================================
class Model:
    def __init__(self, rr, ff, zz, significantPredictors, Tau, indexEndOfEstimation=None, key_demean_predictors=False):

        # Constants.
        #ntopModels = 10
        # indexEndOfEstimation=246
        #print("calculating both unrestricted and restricted models")
        #print("key_Avoid_duplicate_factors= %d, key_Avoid_duplicate_predictors= %d, key_use_combination_matrix= %d " \
        #    % (key_Avoid_duplicate_factors, key_Avoid_duplicate_predictors, key_use_combination_matrix))

        self.Tau = Tau
        print("Tau= %f " % (self.Tau))
        self.keyConditionalCAPM = int(len(significantPredictors)>0)

        if self.keyConditionalCAPM:
            assert len(ff) == len(zz)
        if len(rr) != 0:
            assert len(rr)==len(ff)

        if indexEndOfEstimation==None:
            indexEndOfEstimation=len(ff)-1

        print(ff.loc[indexEndOfEstimation, 'Date'])
        print(zz.loc[indexEndOfEstimation, 'Date'])
        if len(rr) != 0:
            print(rr.loc[indexEndOfEstimation, :])

        # Trying to work with numpy arrays instead of pandas dataframe.
        # In pandas Dataframe x.loc[0:n] are the first n+1 elements.
        self.factorsNames   = ff.columns.drop('Date')
        self.rMktEstimation = ff.loc[self.keyConditionalCAPM : indexEndOfEstimation,'Mkt-RF'].values
        self.FEstimation    = np.ascontiguousarray( \
                        copy.deepcopy(ff.loc[self.keyConditionalCAPM : indexEndOfEstimation, self.factorsNames].values))
        self.FTest          = np.ascontiguousarray( \
                        copy.deepcopy(ff.loc[indexEndOfEstimation + 1 :, self.factorsNames].values))
        self.KMax = len(self.factorsNames)

        # Market squared Sharpe ratio and maximum difference between the upper bound Sharpe ratio and the market.
        # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
        self.SR2MktEstimation  = pow(np.mean(self.rMktEstimation) / np.std(self.rMktEstimation) , 2)
        print("Market SR^2 in the estimation period= %f" %(self.SR2MktEstimation))

        predictorsNames = zz.columns.drop('Date')
        discardPredictors = np.array(list(set(np.arange(0,len(predictorsNames))) - set(significantPredictors)))
        if len(discardPredictors) > 0:
            ZOrig = copy.deepcopy(zz.drop(columns=predictorsNames[discardPredictors]))
        else:
            ZOrig = copy.deepcopy(zz)

        del predictorsNames
        # Aliening Z, R, F in term of the indecies.
        self.predictorsNames = ZOrig.columns.drop('Date')
        self.ZEstimation = np.ascontiguousarray(copy.deepcopy( \
                        ZOrig.loc[: indexEndOfEstimation -1   , self.predictorsNames].values))
        self.ZTest       = np.ascontiguousarray(copy.deepcopy( \
                        ZOrig.loc[indexEndOfEstimation  : len(ZOrig) -2 , self.predictorsNames].values))

        # Demeaning the predictors in the estimation sample.
        print('ZEstimation mean and std')
        self.ZEstimationMean = np.mean(self.ZEstimation,0)
        self.ZEstimationStd  = np.std(self.ZEstimation, 0)
        print(self.ZEstimationMean)
        print(self.ZEstimationStd )
        print('ZTest mean and std')
        print(np.mean(self.ZTest, 0))
        print(np.std(self.ZTest, 0))
        if key_demean_predictors:
            self.ZEstimation[:,:] = (self.ZEstimation[:,:] - self.ZEstimationMean) / self.ZEstimationStd
            self.ZTest[:,:]       = (self.ZTest[:,:] - self.ZEstimationMean) / self.ZEstimationStd
            print("After demeaning")
            print('ZEstimation mean and std')
            print(np.mean(self.ZEstimation, 0))
            print(np.std(self.ZEstimation, 0))
            print('ZTest mean and std')
            print(np.mean(self.ZTest, 0))
            print(np.std(self.ZTest, 0))

        del ZOrig

        self.MMax  = len(self.predictorsNames)

        self.T = self.FEstimation.shape[0]
        if not rr.empty:
            self.testAssetsPortfoliosNames = rr.columns.drop('Date')
            self.REstimation = np.ascontiguousarray(copy.deepcopy( \
                        rr.loc[self.keyConditionalCAPM : indexEndOfEstimation, self.testAssetsPortfoliosNames].values))
            self.RTest       = np.ascontiguousarray(copy.deepcopy( \
                        rr.loc[indexEndOfEstimation + 1 :, testAssetsPortfoliosNames].values))
        else:
            self.REstimation = np.zeros((0,0),dtype=np.float64)
            self.RTest = np.zeros((0,0),dtype=np.float64)

        print("REstimation.shape= " + str(self.REstimation.shape))
        print("RTest.shape= " + str(self.RTest.shape))
        print("FEstimation.shape= " + str(self.FEstimation.shape))
        print("FTest.shape= " + str(self.FTest.shape))
        print("ZEstimation.shape= " + str(self.ZEstimation.shape))
        print("ZTest.shape= " + str(self.ZTest.shape))

        if self.keyConditionalCAPM:
            assert (len(self.FEstimation) == len(self.ZEstimation)) and (len(self.FTest) == len(self.ZTest))
        if self.REstimation.shape[1] > 0:
            assert (len(self.REstimation)==len(self.FEstimation)) and (len(self.RTest)==len(self.FTest))

        print('KMax= %d, MMax= %d, T= %d, keyConditionalAPM=%d' %(self.KMax , self.MMax, self.T, self.keyConditionalCAPM))

        # Calculate Omega full here in order to save the calculations in the loop.
        # F and Z are aliened so they have the same index t for the interactions
        # between the predictors and factors.
        self.OmegaOrigEstimation = np.zeros((self.T, self.KMax * self.MMax), dtype=float)
        self.OmegaOrigEstimation.fill(np.nan)
        for t in np.arange(0, self.T):
            self.OmegaOrigEstimation[t, :] = np.kron(np.identity(self.KMax), self.ZEstimation[t, :].reshape(-1, 1)) @ \
                                             self.FEstimation[t, :]

        assert not np.isnan(self.OmegaOrigEstimation).any()
        # In order to omit duplication of following factors' pairs:
        if keyPrint:
            # ME < --> SMB factor  # 8 and 2
            print(self.factorsNames[8-1] + ' <--> ' + self.factorsNames[2-1])
            # RMW < --> ROE    factor  # 4 and 10
            print(self.factorsNames[4-1] + ' <--> ' + self.factorsNames[10-1])
            # CMA < --> IA factor  # 5 and 9
            print(self.factorsNames[5-1] + ' <--> ' + self.factorsNames[9-1])

# ========== End of Constructor ==========

    def conditionalAssetPricingLogMarginalLikelihoodTau(self):
        # Constants.
        ntopModels = 10

        (logMarginalLikelihood, logMarginalLikelihoodR, T0IncreasedFraction, T0Max, T0Min, T0Avg)= \
            conditionalAssetPricingLogMarginalLikelihoodTauNew(self.REstimation, self.FEstimation, self.ZEstimation, \
                                                           self.OmegaOrigEstimation, self.Tau, self.SR2MktEstimation)

        print('**** Unrestricted models ****')
        [MarginalLikelihood, factorsProbability, predictorsProbability] = \
            calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihood, ntopModels, self.KMax,\
                                    self.MMax, factorsNames=self.factorsNames, predictorsNames=self.predictorsNames, \
                                    factorsInModel=None, predictorsInModel=None, keyPrintResults=True)

        print('**** Restricted model ****')
        [MarginalLikelihoodR, factorsProbabilityR, predictorsProbabilityR] = \
            calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihoodR, ntopModels, self.KMax,\
                                    self.MMax, factorsNames=self.factorsNames, predictorsNames=self.predictorsNames, \
                                    factorsInModel=None, predictorsInModel=None, keyPrintResults=True)

        # factorsProbability=None ; predictorsProbability=None ; factorsProbabilityR=None ; predictorsProbabilityR=None

        return (logMarginalLikelihood, self.factorsNames, factorsProbability, self.predictorsNames, \
                predictorsProbability, T0IncreasedFraction, T0Max, T0Min, T0Avg,  \
                logMarginalLikelihoodR, factorsProbabilityR, predictorsProbabilityR)

    # ========== End of method conditionalAssetPricingLogMarginalLikelihoodTau ==========

    def conditionalAssetPricingLogMarginalLikelihoodTauNumba(self):
        # Constants.
        ntopModels = 10

        print("***** conditionalAssetPricingLogMarginalLikelihoodTauNumba **** ")
        tictoc.tic()
        (logMarginalLikelihood, logMarginalLikelihoodR, T0IncreasedFraction, T0Max, T0Min, T0Avg, \
                T0_div_T0_plus_TAvg, T_div_T0_plus_TAvg, nLegitModels, nTooSmallT0) = \
                conditionalAssetPricingLogMarginalLikelihoodTauNumba(self.REstimation, self.FEstimation, self.ZEstimation, \
                                                               self.OmegaOrigEstimation, self.Tau, self.SR2MktEstimation)
        tictoc.toc()

        print('All combinations= %d Total number of legit models= %d 2ND count %d ' \
              % (pow(2, self.MMax + self.KMax), np.count_nonzero(logMarginalLikelihood != -np.inf), nLegitModels))

        print('# of times T0 was increased= %d T0 Average= %f T0 Max= %f T0 Min= %f' \
              % (nTooSmallT0, T0Avg, T0Max, T0Min))

        print('**** Unrestricted models ****')
        [MarginalLikelihood, factorsProbability, predictorsProbability] = \
            calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihood, ntopModels, self.KMax, \
                                    self.MMax, factorsNames=self.factorsNames, predictorsNames=self.predictorsNames, \
                                    factorsInModel=None, predictorsInModel=None, keyPrintResults=True)

        print('**** Restricted model ****')
        [MarginalLikelihoodR, factorsProbabilityR, predictorsProbabilityR] = \
            calculateMarginalLikelihoodAndFactorsPredictorsProbabilities(logMarginalLikelihoodR, ntopModels, self.KMax,\
                                    self.MMax, factorsNames=self.factorsNames, predictorsNames=self.predictorsNames, \
                                    factorsInModel=None, predictorsInModel=None, keyPrintResults=True)

        # factorsProbability=None
        # predictorsProbability=None
        # factorsProbabilityR=None
        # predictorsProbabilityR=None

        return (logMarginalLikelihood, self.factorsNames, factorsProbability, self.predictorsNames, \
                predictorsProbability, T0IncreasedFraction, T0Max, T0Min, T0Avg, T0_div_T0_plus_TAvg, T_div_T0_plus_TAvg,\
                logMarginalLikelihoodR, factorsProbabilityR, predictorsProbabilityR)
    # ========== End of method conditionalAssetPricingLogMarginalLikelihoodTauNumba ==========

    def conditionalAssetPricingOOSTauNumba(self, models_probabilities, nModelsInPrediction):
        # Constants.

        print("***** conditionalAssetPricingOOSTauNumba **** ")
        print('Sum of probabilities= %f' %(np.sum(models_probabilities)))
        print("Number of models to use in prediction= %i " %(nModelsInPrediction))
        if nModelsInPrediction > 0:
            nModelsMax = pow(2, self.MMax + self.KMax)
            I = np.argsort(-models_probabilities) % nModelsMax
            ModelsIndices = np.unique(I[0:nModelsInPrediction])
            print('Cumulative probabilities the %i models in ModelsIndices= %f' % (len(ModelsIndices), \
                   np.sum(models_probabilities[ModelsIndices])+np.sum(models_probabilities[ModelsIndices+nModelsMax])))

        tictoc.tic()
        (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS,\
            returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
            T0IncreasedFraction, T0Max, T0Min, T0Avg, nLegitModels, nTooSmallT0, cumulative_probability) = \
            conditionalAssetPricingOOSPredictionsTauNumba(self.REstimation, self.FEstimation, self.ZEstimation, \
                                            self.OmegaOrigEstimation, self.Tau, self.SR2MktEstimation, self.ZTest, \
                                            models_probabilities, nModelsInPrediction)
        tictoc.toc()

        print('All combinations= %d Total number of legit models= %d ' \
              % (pow(2, self.MMax + self.KMax), nLegitModels))

        print('Cumulative probability of models= %f' %cumulative_probability )

        return (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                    cumulative_probability)

    # ========== End of method conditionalAssetPricingOOSTauNumba ==========

    def conditionalAssetPricingSingleOOSTauNumba(self, models_probabilities, single_top_model):
        # Constants.
        nModelsInPrediction = -single_top_model

        print("***** conditionalAssetPricingSingleOOSTauNumba **** ")
        print('Sum of probabilities= %f' %(np.sum(models_probabilities)))
        print("Number of top model to use in prediction= %i" %(single_top_model))
        
        nModelsMax = pow(2, self.MMax + self.KMax)
        # Copy the models_probabilities array since we set it to zero in conditionalAssetPricingOOSPredictionsTauNumba.
        # Not very efficient....
        model_probabilities_temp = copy.deepcopy(models_probabilities)
        I = np.argsort(-model_probabilities_temp)
        model_index =  I[single_top_model-1]
        print(" %i top model to use in prediction is model number %i with probability %f" \
            %(single_top_model, model_index, model_probabilities_temp[model_index]))

        tictoc.tic()

        (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
            returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
            T0IncreasedFraction, T0Max, T0Min, T0Avg, nLegitModels, nTooSmallT0, cumulative_probability) = \
            conditionalAssetPricingOOSPredictionsTauNumba(self.REstimation, self.FEstimation, self.ZEstimation, \
                                            self.OmegaOrigEstimation, self.Tau, self.SR2MktEstimation, self.ZTest, \
                                            model_probabilities_temp, nModelsInPrediction)
        del model_probabilities_temp
        tictoc.toc()

        print('All combinations= %d Total number of legit models= %d ' \
              % (pow(2, self.MMax + self.KMax), nLegitModels))

        print('Cumulative probability of models= %f' %cumulative_probability )

        return (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                returns_IN, returns_square_IN, returns_interactions_IN,  covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                    cumulative_probability)

    # ========== End of method conditionalAssetPricingSingleOOSTauNumba ==========
    
    def conditionalAssetCalculateSpread(self, models_probabilities, nModelsInPrediction):
        # Constants.

        print("***** conditionalAssetCalculateSpread **** ")
        print('Sum of probabilities= %f' %(np.sum(models_probabilities)))
        print("Number of models to use in prediction= %i " %(nModelsInPrediction))
        if nModelsInPrediction > 0:
            nModelsMax = pow(2, self.MMax + self.KMax)
            I = np.argsort(-models_probabilities) % nModelsMax
            ModelsIndices = np.unique(I[0:nModelsInPrediction])
            print('Cumulative probabilities the %i models in ModelsIndices= %f' % (len(ModelsIndices), \
                   np.sum(models_probabilities[ModelsIndices])+np.sum(models_probabilities[ModelsIndices+nModelsMax])))

        tictoc.tic()

        (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
            weights_sum_OOS, weights_square_sum_OOS, weights_sum_equal_OOS, weights_square_sum_equal_OOS, \
            returns_sum_OOS, returns_square_sum_OOS, returns_sum_equal_OOS, returns_square_sum_equal_OOS, \
            Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, \
            returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
            weights_sum_IN, weights_square_sum_IN, weights_sum_equal_IN, weights_square_sum_equal_IN, \
            returns_sum_IN, returns_square_sum_IN, returns_sum_equal_IN, returns_square_sum_equal_IN, \
            Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, \
            Returns_terms_cumulative_probability, \
            T0IncreasedFraction, T0Max, T0Min, T0Avg, nLegitModels, nTooSmallT0, cumulative_probability, total_number_of_models) = \
            conditionalAssetCalculateSpreadNumba(self.REstimation, self.FEstimation, self.ZEstimation, \
                                            self.OmegaOrigEstimation, self.Tau, self.SR2MktEstimation, \
                                            self.RTest, self.FTest, self.ZTest, \
                                            models_probabilities, nModelsInPrediction)
        tictoc.toc()
        assert np.allclose(returns_OOS, np.sum(Returns_terms_weighted_OOS,axis=2))
        assert np.allclose(returns_IN, np.sum(Returns_terms_weighted_IN,axis=2))

        print('All combinations= %d Total number of legit models= %d ' \
              % (pow(2, self.MMax + self.KMax), nLegitModels))

        print('Cumulative probability of models= %f' %cumulative_probability )

        return (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                    weights_sum_OOS, weights_square_sum_OOS, weights_sum_equal_OOS, weights_square_sum_equal_OOS, \
                    returns_sum_OOS, returns_square_sum_OOS, returns_sum_equal_OOS, returns_square_sum_equal_OOS, \
                    Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, \
                    returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                    weights_sum_IN, weights_square_sum_IN, weights_sum_equal_IN, weights_square_sum_equal_IN, \
                    returns_sum_IN, returns_square_sum_IN, returns_sum_equal_IN, returns_square_sum_equal_IN, \
                    Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, \
                    Returns_terms_cumulative_probability, \
                    cumulative_probability, total_number_of_models)

    # ========== End of method conditionalAssetCalculateSpread ==========

#    (returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN,\
#                                                            returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, \
#                                                            gamma=gamma, only_top_model=True, dump_directory=dump_directory, num_top=top)

    # In this function the arguments covariance_matrix_in_sample and covariance_matrix_OOS are the V_t components in eq. 5 
    def AnalyseInSampleAndOOSPortfolioReturns(self, \
                                        returns_in_sample, returns_square_in_sample, returns_interactions_in_sample, \
                                        covariance_matrix_in_sample, covariance_matrix_no_ER_in_sample, \
                                        returns_OOS, returns_square_OOS, returns_interactions_OOS, \
                                        covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                                        gamma, only_top_model=False, dump_directory=None, num_top=None):

        T = len(self.FEstimation)
        TOOS = len(self.FTest)

        # Calculating the in-sample and out-of-sampel of the benchmark portfolios: 
        # CAPM, FF3, FF6, AQR6, 14 factors unconditional.
        if not only_top_model:
            # CAPM
            model_indecies = np.array([0])
            print("CAPM - %s " % (self.factorsNames[model_indecies]))
            R_CAPM_in_sample = self.FEstimation[:,model_indecies]
            R_CAPM = copy.deepcopy(R_CAPM_in_sample)
            if TOOS > 0:
                R_CAPM_OOS = self.FTest[:,model_indecies]
                R_CAPM = np.concatenate((R_CAPM, R_CAPM_OOS),axis=0)
            
            tau = 1.5
            print('T0 CAPM %f for tau=%f' %(Model.calculateT0(R_CAPM_in_sample.reshape(-1,1), R_CAPM_in_sample.reshape(-1,1), tau, self.KMax), tau))
            
            # FF3
            model_indecies = np.array([0,1,2])
            print("FF3 - %s"  % (self.factorsNames[model_indecies]))
            R_FF3_in_sample = Model.unconditional_model_portfolio_returns(self.FEstimation[:,model_indecies], self.FEstimation[:,model_indecies])
            R_FF3 = copy.deepcopy(R_FF3_in_sample)
            R_FF3_GMVP_in_sample = Model.GMVP_unconditional_model_portfolio_returns(self.FEstimation[:,model_indecies], self.FEstimation[:,model_indecies])
            R_FF3_GMVP = copy.deepcopy(R_FF3_GMVP_in_sample)
            if TOOS > 0:
                R_FF3_OOS = Model.unconditional_model_portfolio_returns(self.FTest[:,model_indecies], self.FEstimation[:,model_indecies])
                R_FF3 = np.concatenate((R_FF3, R_FF3_OOS),axis=0)
                R_FF3_GMVP_OOS = Model.GMVP_unconditional_model_portfolio_returns(self.FTest[:,model_indecies], self.FEstimation[:,model_indecies])
                R_FF3_GMVP = np.concatenate((R_FF3_GMVP, R_FF3_GMVP_OOS),axis=0)
            
            # FF3 regulation T.
            print("FF3 regulation T - %s"  %(self.factorsNames[model_indecies]))
            Mean = np.mean(self.FEstimation[:,model_indecies],axis=0)
            covariance_matrix_unconditional = np.cov(self.FEstimation[:,model_indecies],rowvar=False, bias=False)
            returns_in_sample_uncoditional = np.repeat(Mean.reshape(1,-1),T,axis=0)
            returns_square_in_sample_uncoditional = np.repeat(Mean.reshape(1,-1)**2,T,axis=0)
            R_FF3_regulation_T_in_sample = Model.diagonal_cov_regulation_T_portfolio_returns(self.FEstimation[:,model_indecies], \
                returns_in_sample_uncoditional, returns_square_in_sample_uncoditional, covariance_matrix_unconditional, gamma)
            R_FF3_regulation_T = copy.deepcopy(R_FF3_regulation_T_in_sample)
            if TOOS > 0:
                returns_OOS_uncoditional = np.repeat(Mean.reshape(1,-1),TOOS,axis=0)
                returns_square_OOS_uncoditional = np.repeat(Mean.reshape(1,-1)**2,TOOS,axis=0)
                R_FF3_regulation_T_OOS = Model.diagonal_cov_regulation_T_portfolio_returns(self.FTest[:,model_indecies], \
                    returns_OOS_uncoditional, returns_square_OOS_uncoditional, covariance_matrix_unconditional, gamma)
                R_FF3_regulation_T = np.concatenate((R_FF3_regulation_T, R_FF3_regulation_T_OOS),axis=0)

            if TOOS > 0:
                print("FF3 regulation T SR: in-sample %f OOS %f" \
                    %( Model.SharpeRatio(R_FF3_regulation_T_in_sample), Model.SharpeRatio(R_FF3_regulation_T_OOS)))
            else:
                print("FF3 regulation T SR: in-sample %f" %(  Model.SharpeRatio(R_FF3_regulation_T_in_sample)))

            print('T0 FF3 %f for tau=%f' %(Model.calculateT0(R_CAPM_in_sample.reshape(-1,1), self.FEstimation[:,model_indecies], tau, self.KMax), tau))

            # FF6
            model_indecies = np.array([0,1,2,3,4,5])
            print("FF6 - %s" %(self.factorsNames[model_indecies]))
            R_FF6_in_sample = Model.unconditional_model_portfolio_returns(self.FEstimation[:,model_indecies], self.FEstimation[:,model_indecies])
            R_FF6 = copy.deepcopy(R_FF6_in_sample)
            R_FF6_GMVP_in_sample = Model.GMVP_unconditional_model_portfolio_returns(self.FEstimation[:,model_indecies], self.FEstimation[:,model_indecies])
            R_FF6_GMVP = copy.deepcopy(R_FF6_GMVP_in_sample)
            if TOOS > 0:
                R_FF6_OOS = Model.unconditional_model_portfolio_returns(self.FTest[:,model_indecies], self.FEstimation[:,model_indecies])
                R_FF6 = np.concatenate((R_FF6, R_FF6_OOS),axis=0)
                R_FF6_GMVP_OOS = Model.GMVP_unconditional_model_portfolio_returns(self.FTest[:,model_indecies], self.FEstimation[:,model_indecies])
                R_FF6_GMVP = np.concatenate((R_FF6_GMVP, R_FF6_GMVP_OOS),axis=0)

            if TOOS > 0:
                print("FF6 SR: in-sample %f OOS %f" %( Model.SharpeRatio(R_FF6_in_sample), Model.SharpeRatio(R_FF6_OOS)))
            else:
                print("FF6 SR: in-sample %f" %( Model.SharpeRatio(R_FF6_in_sample)))
            
            # FF6 regulation T.
            print("FF6 regulation T - %s"  %(self.factorsNames[model_indecies]))
            Mean = np.mean(self.FEstimation[:,model_indecies],axis=0)
            covariance_matrix_unconditional = np.cov(self.FEstimation[:,model_indecies],rowvar=False, bias=False)
            returns_in_sample_uncoditional = np.repeat(Mean.reshape(1,-1),T,axis=0)
            returns_square_in_sample_uncoditional = np.repeat(Mean.reshape(1,-1)**2,T,axis=0)
            R_FF6_regulation_T_in_sample = Model.diagonal_cov_regulation_T_portfolio_returns(self.FEstimation[:,model_indecies], \
                returns_in_sample_uncoditional, returns_square_in_sample_uncoditional, covariance_matrix_unconditional, gamma)
            R_FF6_regulation_T = copy.deepcopy(R_FF6_regulation_T_in_sample)
            if TOOS > 0:
                returns_OOS_uncoditional = np.repeat(Mean.reshape(1,-1),TOOS,axis=0)
                returns_square_OOS_uncoditional = np.repeat(Mean.reshape(1,-1)**2,TOOS,axis=0)
                R_FF6_regulation_T_OOS = Model.diagonal_cov_regulation_T_portfolio_returns(self.FTest[:,model_indecies], \
                    returns_OOS_uncoditional, returns_square_OOS_uncoditional, covariance_matrix_unconditional, gamma)
                R_FF6_regulation_T = np.concatenate((R_FF6_regulation_T, R_FF6_regulation_T_OOS),axis=0)

            if TOOS > 0:
                print("FF6 regulation T SR: in-sample %f OOS %f" \
                    %( Model.SharpeRatio(R_FF6_regulation_T_in_sample), Model.SharpeRatio(R_FF6_regulation_T_OOS)))
            else:
                print("FF6 regulation T SR: in-sample %f" %(  Model.SharpeRatio(R_FF6_regulation_T_in_sample)))

            print('T0 FF6 %f for tau=%f' %(Model.calculateT0(R_CAPM_in_sample.reshape(-1,1), self.FEstimation[:,model_indecies], tau, self.KMax), tau))
            model_indecies = np.array([0,1,2,3,4])
            print('T0 FF5 %f for tau=%f' %(Model.calculateT0(R_CAPM_in_sample.reshape(-1,1), self.FEstimation[:,model_indecies], tau, self.KMax), tau))
            
            # AQR6
            model_indecies = np.array([0,1,2,5,8,9])
            print("AQR6 - %s" %(self.factorsNames[model_indecies]))
            R_AQR6_in_sample = Model.unconditional_model_portfolio_returns(self.FEstimation[:,model_indecies], self.FEstimation[:,model_indecies])
            R_AQR6 = copy.deepcopy(R_AQR6_in_sample)
            R_AQR6_GMVP_in_sample = Model.GMVP_unconditional_model_portfolio_returns(self.FEstimation[:,model_indecies], self.FEstimation[:,model_indecies])
            R_AQR6_GMVP = copy.deepcopy(R_AQR6_GMVP_in_sample)
            if TOOS > 0:
                R_AQR6_OOS = Model.unconditional_model_portfolio_returns(self.FTest[:,model_indecies], self.FEstimation[:,model_indecies])
                R_AQR6 = np.concatenate((R_AQR6, R_AQR6_OOS),axis=0)
                R_AQR6_GMVP_OOS = Model.GMVP_unconditional_model_portfolio_returns(self.FTest[:,model_indecies], self.FEstimation[:,model_indecies])
                R_AQR6_GMVP = np.concatenate((R_AQR6_GMVP, R_AQR6_GMVP_OOS),axis=0)

            if TOOS > 0:
                print("AQR6 SR: in-sample %f OOS %f" %( Model.SharpeRatio(R_AQR6_in_sample), Model.SharpeRatio(R_AQR6_OOS)))
            else:
                print("AQR6 SR: in-sample %f" %( Model.SharpeRatio(R_AQR6_in_sample)))

            print("AQR6 regulation T - %s"  %(self.factorsNames[model_indecies]))
            Mean = np.mean(self.FEstimation[:,model_indecies],axis=0)
            covariance_matrix_unconditional = np.cov(self.FEstimation[:,model_indecies],rowvar=False, bias=False)
            returns_in_sample_uncoditional = np.repeat(Mean.reshape(1,-1),T,axis=0)
            returns_square_in_sample_uncoditional = np.repeat(Mean.reshape(1,-1)**2,T,axis=0)
            R_AQR6_regulation_T_in_sample = Model.diagonal_cov_regulation_T_portfolio_returns(self.FEstimation[:,model_indecies], \
                returns_in_sample_uncoditional, returns_square_in_sample_uncoditional, covariance_matrix_unconditional, gamma)
            R_AQR6_regulation_T = copy.deepcopy(R_AQR6_regulation_T_in_sample)
            if TOOS > 0:
                returns_OOS_uncoditional = np.repeat(Mean.reshape(1,-1),TOOS,axis=0)
                returns_square_OOS_uncoditional = np.repeat(Mean.reshape(1,-1)**2,TOOS,axis=0)
                R_AQR6_regulation_T_OOS = Model.diagonal_cov_regulation_T_portfolio_returns(self.FTest[:,model_indecies], \
                    returns_OOS_uncoditional, returns_square_OOS_uncoditional, covariance_matrix_unconditional, gamma)
                
                R_AQR6_regulation_T = np.concatenate((R_AQR6_regulation_T, R_AQR6_regulation_T_OOS),axis=0)

            if TOOS > 0:
                print("AQR6 regulation T SR: in-sample %f OOS %f" \
                    %( Model.SharpeRatio(R_AQR6_regulation_T_in_sample), Model.SharpeRatio(R_AQR6_regulation_T_OOS)))
            else:
                print("AQR6 regulation T SR: in-sample %f" %(  Model.SharpeRatio(R_AQR6_regulation_T_in_sample)))

            print('T0 AQR66 %f for tau=%f' %(Model.calculateT0(R_CAPM_in_sample.reshape(-1,1), self.FEstimation[:,model_indecies], tau, self.KMax), tau))
                   
        # ===== End of not only_top_model ===== ; calculating the benchmark models.

        # Conditional with diagonal covariance matrix. V_t + diagonal (\Omega_t)
        (R_BMA_diagonal_in_sample, w_BMA_diagonal_in_sample, variance_matrix_ratio_diagonal_in_sample, \
            variance_matrix_ratio_contribution_diagonal_in_sample, \
            R_BMA_GMVP_diagonal_in_sample, w_BMA_GMVP_diagonal_in_sample) = \
            Model.BMA_diagonal_cov_portfolio_returns(self.FEstimation, returns_in_sample, returns_square_in_sample, \
                covariance_matrix_in_sample)
        variance_matrix_ratio_diagonal = copy.deepcopy(variance_matrix_ratio_diagonal_in_sample)
        variance_matrix_ratio_contribution_diagonal = copy.deepcopy(variance_matrix_ratio_contribution_diagonal_in_sample)
        R_BMA_diagonal = copy.deepcopy(R_BMA_diagonal_in_sample)
        w_BMA_diagonal = copy.deepcopy(w_BMA_diagonal_in_sample)
        R_BMA_GMVP_diagonal = copy.deepcopy(R_BMA_GMVP_diagonal_in_sample)
        w_BMA_GMVP_diagonal = copy.deepcopy(w_BMA_GMVP_diagonal_in_sample)
        if TOOS > 0:
            (R_BMA_diagonal_OOS, w_BMA_diagonal_OOS, variance_matrix_ratio_diagonal_OOS, \
                variance_matrix_ratio_contribution_diagonal_OOS, \
                R_BMA_GMVP_diagonal_OOS, w_BMA_GMVP_diagonal_OOS) = \
                Model.BMA_diagonal_cov_portfolio_returns(self.FTest, returns_OOS, returns_square_OOS, \
                    covariance_matrix_OOS)
            variance_matrix_ratio_diagonal = np.concatenate((variance_matrix_ratio_diagonal, variance_matrix_ratio_diagonal_OOS), axis=0)
            variance_matrix_ratio_contribution_diagonal = \
                np.concatenate((variance_matrix_ratio_contribution_diagonal, variance_matrix_ratio_contribution_diagonal_OOS), axis=0)
            R_BMA_diagonal =  np.concatenate((R_BMA_diagonal, R_BMA_diagonal_OOS),axis=0)
            w_BMA_diagonal =  np.concatenate((w_BMA_diagonal, w_BMA_diagonal_OOS),axis=0)
            R_BMA_GMVP_diagonal = np.concatenate((R_BMA_GMVP_diagonal, R_BMA_GMVP_diagonal_OOS),axis=0)
            w_BMA_GMVP_diagonal = np.concatenate((w_BMA_GMVP_diagonal, w_BMA_GMVP_diagonal_OOS),axis=0)

        # Regulation T with diagonal covariance matrix.
        R_BMA_diagonal_regulation_T_in_sample = Model.diagonal_cov_regulation_T_portfolio_returns(self.FEstimation, returns_in_sample, returns_square_in_sample, covariance_matrix_in_sample, gamma)
        R_BMA_diagonal_regulation_T = copy.deepcopy(R_BMA_diagonal_regulation_T_in_sample)
        if TOOS > 0:
            R_BMA_diagonal_regulation_T_OOS = Model.diagonal_cov_regulation_T_portfolio_returns(self.FTest, returns_OOS, returns_square_OOS, covariance_matrix_OOS, gamma)
            R_BMA_diagonal_regulation_T = np.concatenate((R_BMA_diagonal_regulation_T, R_BMA_diagonal_regulation_T_OOS), axis=0)

        # **** End **** Conditional with diagonal covariance matrix. V_t + diagonal (\Omega_t)

        # Covariance matrix consists of only the weighted average of the models covariance matrix V_t in equation 5. 
        # To zero the model disagreement conditional part the estimated square is set to the square of the estimate.
        (R_BMA_static_cov_in_sample, w_BMA_static_cov_in_sample, dummy, \
            dummy1, \
            R_BMA_GMVP_static_cov_in_sample, w_BMA_GMVP_static_cov_in_sample) = \
            Model.BMA_diagonal_cov_portfolio_returns(self.FEstimation, returns_in_sample, returns_in_sample**2, \
                covariance_matrix_in_sample)

        R_BMA_static_cov = copy.deepcopy(R_BMA_static_cov_in_sample)
        w_BMA_static_cov = copy.deepcopy(w_BMA_static_cov_in_sample)
        R_BMA_GMVP_static_cov = copy.deepcopy(R_BMA_GMVP_static_cov_in_sample)
        w_BMA_GMVP_static_cov = copy.deepcopy(w_BMA_GMVP_static_cov_in_sample)

        if TOOS > 0:           
            # Covariance matrix consists of only the weighted average of the models covariance matrix V_t in equation 5. 
            # To zero the model disagreement conditional part the estimated square is set to the square of the estimate.
            (R_BMA_static_cov_OOS, w_BMA_static_cov_OOS, dummy, \
                dummy1, \
                R_BMA_GMVP_static_cov_OOS, w_BMA_GMVP_static_cov_OOS) = \
                Model.BMA_diagonal_cov_portfolio_returns(self.FTest, returns_OOS, returns_OOS**2, \
                    covariance_matrix_OOS)

            R_BMA_static_cov = np.concatenate((R_BMA_static_cov, R_BMA_static_cov_OOS),axis=0)
            w_BMA_static_cov = np.concatenate((w_BMA_static_cov, w_BMA_static_cov_OOS),axis=0)
            R_BMA_GMVP_static_cov = np.concatenate((R_BMA_GMVP_static_cov, R_BMA_GMVP_static_cov_OOS),axis=0)
            w_BMA_GMVP_static_cov = np.concatenate((w_BMA_GMVP_static_cov, w_BMA_GMVP_static_cov_OOS),axis=0)

        # Regulation T with covariance matrix consists of only the weighted average of the models covariance matrix V_t in equation 5. 
        R_BMA_static_cov_regulation_T_in_sample = Model.diagonal_cov_regulation_T_portfolio_returns(self.FEstimation, returns_in_sample, returns_in_sample**2, covariance_matrix_in_sample, gamma)
        R_BMA_static_cov_regulation_T = copy.deepcopy(R_BMA_static_cov_regulation_T_in_sample)
        if TOOS > 0:
            R_BMA_static_cov_regulation_T_OOS = Model.diagonal_cov_regulation_T_portfolio_returns(self.FTest, returns_OOS, returns_OOS**2, covariance_matrix_OOS, gamma)
            R_BMA_static_cov_regulation_T = np.concatenate((R_BMA_static_cov_regulation_T, R_BMA_static_cov_regulation_T_OOS), axis=0)
        
        # **** End **** Covariance matrix consists of only the weighted average of the models covariance matrix V_t in equation 5. 

        # Covariance matrix consists of only the weighted average of the models covariance matrix V_t with out estimation risk in equation 5. 
        # To zero the model disagreement conditional part the estimated square is set to the square of the estimate.
        (R_BMA_cov_no_ER_in_sample, w_BMA_cov_no_ER_in_sample, dummy, \
            dummy1, \
            R_BMA_GMVP_cov_no_ER_in_sample, w_BMA_GMVP_cov_no_ER_in_sample) = \
            Model.BMA_diagonal_cov_portfolio_returns(self.FEstimation, returns_in_sample, returns_in_sample**2, \
                covariance_matrix_no_ER_in_sample)

        R_BMA_cov_no_ER = copy.deepcopy(R_BMA_cov_no_ER_in_sample)
        w_BMA_cov_no_ER = copy.deepcopy(w_BMA_cov_no_ER_in_sample)
        R_BMA_GMVP_cov_no_ER = copy.deepcopy(R_BMA_GMVP_cov_no_ER_in_sample)
        w_BMA_GMVP_cov_no_ER = copy.deepcopy(w_BMA_GMVP_cov_no_ER_in_sample)

        if TOOS > 0:           
            # Covariance matrix consists of only the weighted average of the models covariance matrix V_t with out estimation risk in equation 5. 
            # To zero the model disagreement conditional part the estimated square is set to the square of the estimate.
            (R_BMA_cov_no_ER_OOS, w_BMA_cov__no_ER_OOS, dummy, \
                dummy1, \
                R_BMA_GMVP_cov_no_ER_OOS, w_BMA_GMVP_cov_no_ER_OOS) = \
                Model.BMA_diagonal_cov_portfolio_returns(self.FTest, returns_OOS, returns_OOS**2, \
                    covariance_matrix_no_ER_OOS)

            R_BMA_cov_no_ER = np.concatenate((R_BMA_cov_no_ER, R_BMA_cov_no_ER_OOS),axis=0)
            w_BMA_cov_no_ER = np.concatenate((w_BMA_cov_no_ER, w_BMA_cov__no_ER_OOS),axis=0)
            R_BMA_GMVP_cov_no_ER = np.concatenate((R_BMA_GMVP_cov_no_ER, R_BMA_GMVP_cov_no_ER_OOS),axis=0)
            w_BMA_GMVP_cov_no_ER = np.concatenate((w_BMA_GMVP_cov_no_ER, w_BMA_GMVP_cov_no_ER_OOS),axis=0)

        # Regulation T with covariance matrix consists of only the weighted average of the models covariance matrix V_t in equation 5. 
        R_BMA_cov_no_ER_regulation_T_in_sample = Model.diagonal_cov_regulation_T_portfolio_returns(self.FEstimation, returns_in_sample, returns_in_sample**2, covariance_matrix_no_ER_in_sample, gamma)
        R_BMA_cov_no_ER_regulation_T = copy.deepcopy(R_BMA_cov_no_ER_regulation_T_in_sample)
        if TOOS > 0:
            R_BMA_cov_no_ER_regulation_T_OOS = Model.diagonal_cov_regulation_T_portfolio_returns(self.FTest, returns_OOS, returns_OOS**2, covariance_matrix_no_ER_OOS, gamma)
            R_BMA_cov_no_ER_regulation_T = np.concatenate((R_BMA_static_cov_regulation_T, R_BMA_cov_no_ER_regulation_T_OOS), axis=0)
        
        # **** End **** Covariance matrix consists of only the weighted average of the models covariance matrix V_t in equation 5. 

        if not only_top_model:
            # Conditional with full covariance matrix.
            (R_BMA_full_in_sample, w_BMA_full_in_sample, variance_matrix_ratio_full_in_sample, \
                variance_matrix_ratio_contribution_full_in_sample, \
                R_BMA_GMVP_full_in_sample, w_BMA_GMVP_full_in_sample, \
                cov_matrix_full_TS_avg_in_sample, omega_TS_avg_in_sample) = \
                Model.BMA_full_cov_portfolio_returns(self.FEstimation, returns_in_sample, returns_square_in_sample, \
                    returns_interactions_in_sample, covariance_matrix_in_sample)
            cov_matrix_full_obs_in_sample = np.cov(self.FEstimation, rowvar=False, bias=False).reshape(self.FEstimation.shape[1], self.FEstimation.shape[1])
            variance_matrix_ratio_full = copy.deepcopy(variance_matrix_ratio_full_in_sample)
            variance_matrix_ratio_contribution_full = copy.deepcopy(variance_matrix_ratio_contribution_full_in_sample)
            R_BMA_full =  copy.deepcopy(R_BMA_full_in_sample)
            w_BMA_full = copy.deepcopy(w_BMA_full_in_sample)
            R_BMA_GMVP_full = copy.deepcopy(R_BMA_GMVP_full_in_sample)
            w_BMA_GMVP_full = copy.deepcopy(w_BMA_GMVP_full_in_sample)
            if TOOS > 0:
                (R_BMA_full_OOS, w_BMA_full_OOS, variance_matrix_ratio_full_OOS, \
                    variance_matrix_ratio_contribution_full_OOS, \
                    R_BMA_GMVP_full_OOS, w_BMA_GMVP_full_OOS, \
                    cov_matrix_full_TS_avg_OOS, omega_TS_avg_OOS) = \
                    Model.BMA_full_cov_portfolio_returns(self.FTest, returns_OOS, returns_square_OOS, \
                        returns_interactions_OOS, covariance_matrix_OOS)
                cov_matrix_full_obs_OOS = np.cov(self.FTest, rowvar=False, bias=False).reshape(self.FTest.shape[1], self.FTest.shape[1])
                variance_matrix_ratio_full = np.concatenate((variance_matrix_ratio_full, variance_matrix_ratio_full_OOS), axis=0)
                variance_matrix_ratio_contribution_full = \
                    np.concatenate((variance_matrix_ratio_contribution_full, variance_matrix_ratio_contribution_full_OOS), axis=0)
                R_BMA_full =  np.concatenate((R_BMA_full, R_BMA_full_OOS),axis=0)
                w_BMA_full =  np.concatenate((w_BMA_full, w_BMA_full_OOS),axis=0)
                R_BMA_GMVP_full = np.concatenate((R_BMA_GMVP_full, R_BMA_GMVP_full_OOS),axis=0)
                w_BMA_GMVP_full = np.concatenate((w_BMA_GMVP_full, w_BMA_GMVP_full_OOS),axis=0)
            # Regulation T with full covariance matrix.
            R_BMA_full_regulation_T_in_sample = Model.full_cov_regulation_T_portfolio_returns(self.FEstimation, \
                                                        returns_in_sample, returns_square_in_sample, \
                                                        returns_interactions_in_sample, covariance_matrix_in_sample, gamma)
            R_BMA_full_regulation_T = copy.deepcopy(R_BMA_full_regulation_T_in_sample)
            if TOOS > 0:
                R_BMA_full_regulation_T_OOS = Model.full_cov_regulation_T_portfolio_returns(self.FTest, \
                                                        returns_OOS, returns_square_OOS, 
                                                        returns_interactions_OOS, covariance_matrix_OOS, gamma)
                R_BMA_full_regulation_T = np.concatenate((R_BMA_full_regulation_T, R_BMA_full_regulation_T_OOS),\
                                            axis=0)

        if not dump_directory is None and not only_top_model :
            np.savetxt(dump_directory+"variance_matrix_ratio_diagonal.csv", variance_matrix_ratio_diagonal, delimiter=",")
            DF = pd.DataFrame(variance_matrix_ratio_contribution_diagonal, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"variance_matrix_ratio_contribution_diagonal.csv", index=False)
            # ================    Full covariance matrix    ================
            np.savetxt(dump_directory+"variance_matrix_ratio_full.csv", variance_matrix_ratio_full, delimiter=",")
            DF = pd.DataFrame(variance_matrix_ratio_contribution_full, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"variance_matrix_ratio_contribution_full.csv", index=False)
            np.savetxt(dump_directory+"R_CAPM.csv", R_CAPM, delimiter=",")
            np.savetxt(dump_directory+"R_FF3.csv", R_FF3, delimiter=",")
            np.savetxt(dump_directory+"R_FF3_GMVP.csv", R_FF3_GMVP, delimiter=",")
            np.savetxt(dump_directory+"R_FF3_regulation_T.csv", R_FF3_regulation_T, delimiter=",")
            np.savetxt(dump_directory+"R_FF6.csv", R_FF6, delimiter=",")
            np.savetxt(dump_directory+"R_FF6_GMVP.csv", R_FF6_GMVP, delimiter=",")
            np.savetxt(dump_directory+"R_FF6_regulation_T.csv", R_FF6_regulation_T, delimiter=",")
            np.savetxt(dump_directory+"R_AQR6.csv", R_AQR6, delimiter=",")
            np.savetxt(dump_directory+"R_AQR6_GMVP.csv", R_AQR6_GMVP, delimiter=",")
            np.savetxt(dump_directory+"R_AQR6_regulation_T.csv", R_AQR6_regulation_T, delimiter=",")
            # ================    Diagonal covariance matrix    ================
            np.savetxt(dump_directory+"R_BMA_diagonal.csv", R_BMA_diagonal, delimiter=",")
            DF = pd.DataFrame(w_BMA_diagonal, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_diagonal.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_GMVP_diagonal.csv", R_BMA_GMVP_diagonal, delimiter=",")
            DF = pd.DataFrame(w_BMA_GMVP_diagonal, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_GMVP_diagonal.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_diagonal_regulation_T.csv", R_BMA_diagonal_regulation_T, delimiter=",")
            # ================ Static part covariance matrix ================
            np.savetxt(dump_directory+"R_BMA_static_cov.csv", R_BMA_static_cov , delimiter=",")
            DF = pd.DataFrame(w_BMA_static_cov, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_static_cov.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_GMVP_static_cov.csv", R_BMA_GMVP_static_cov , delimiter=",")
            DF = pd.DataFrame(w_BMA_GMVP_static_cov, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_GMVP_static_cov.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_static_cov_regulation_T.csv", R_BMA_static_cov_regulation_T, delimiter=",")
            # ================ Vt part covariance matrix without the estimation risk part ================
            np.savetxt(dump_directory+"R_BMA_cov_no_ER.csv", R_BMA_cov_no_ER , delimiter=",")
            DF = pd.DataFrame(w_BMA_cov_no_ER, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_cov_no_ER.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_GMVP_cov_no_ER.csv", R_BMA_GMVP_cov_no_ER , delimiter=",")
            DF = pd.DataFrame(w_BMA_GMVP_cov_no_ER, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_GMVP_cov_no_ER.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_cov_no_ER_regulation_T.csv", R_BMA_cov_no_ER_regulation_T, delimiter=",")
            # ================ Full covariance matrix ================
            np.savetxt(dump_directory+"R_BMA_full.csv", R_BMA_full, delimiter=",")
            # Leveraged retruns of the BMA full such that the volatility of the leveraged portfolio is
            # equal to the volatility of the MKT protfolio.
            leve = np.std(R_CAPM_in_sample)/np.std(R_BMA_full_in_sample)
            print ('Leverage= %f' %(leve))
            np.savetxt(dump_directory+"R_BMA_full_leveraged.csv", R_BMA_full*leve, delimiter=",")
            DF = pd.DataFrame(cov_matrix_full_TS_avg_in_sample, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"cov_matrix_full_TS_avg_in_sample.csv", index=False)
            DF = pd.DataFrame(omega_TS_avg_in_sample, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"omega_TS_avg_in_sample.csv", index=False)
            DF = pd.DataFrame(cov_matrix_full_obs_in_sample, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"cov_matrix_full_obs_in_sample.csv", index=False)
            if TOOS > 0:
                DF = pd.DataFrame(cov_matrix_full_TS_avg_OOS, columns=list(BMA.factorsNames))
                DF.to_csv(dump_directory+"cov_matrix_full_TS_avg_OOS.csv", index=False)
                DF = pd.DataFrame(omega_TS_avg_OOS, columns=list(BMA.factorsNames))
                DF.to_csv(dump_directory+"omega_TS_avg_OOS.csv", index=False)
                DF = pd.DataFrame(cov_matrix_full_obs_OOS, columns=list(BMA.factorsNames))
                DF.to_csv(dump_directory+"cov_matrix_full_obs_OOS.csv", index=False)

            DF = pd.DataFrame(w_BMA_full, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_full.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_GMVP_full.csv", R_BMA_GMVP_full, delimiter=",")
            DF = pd.DataFrame(w_BMA_GMVP_full, columns=list(BMA.factorsNames))
            DF.to_csv(dump_directory+"w_BMA_GMVP_full.csv", index=False)
            np.savetxt(dump_directory+"R_BMA_full_regulation_T.csv", R_BMA_full_regulation_T, delimiter=",")
        
        if not dump_directory is None and only_top_model and not num_top is None:
            np.savetxt(dump_directory+"R_conditional_top_" + str(num_top) + ".csv", R_BMA_diagonal, delimiter=",")
            np.savetxt(dump_directory+"R_conditional_GMVP_top_" + str(num_top) + ".csv",R_BMA_GMVP_diagonal, delimiter=",")
            np.savetxt(dump_directory+"R_regulation_T_top_" + str(num_top) + ".csv", R_BMA_diagonal_regulation_T, delimiter=",")

        # Pack SR results in a matrix for a table.
        nResults = 19
        if TOOS > 0:
            results = -np.ones((nResults, 2),dtype=np.float64); results.fill(np.NaN)
        else:
            results = -np.ones((nResults, 1),dtype=np.float64); results.fill(np.NaN)

        if not only_top_model:
            results[0,0] = Model.SharpeRatio(R_CAPM_in_sample)
            if TOOS > 0:
                results[0,1] = Model.SharpeRatio(R_CAPM_OOS)
            results[1,0] = Model.SharpeRatio(R_FF3_in_sample)
            if TOOS > 0:
                results[1,1] = Model.SharpeRatio(R_FF3_OOS)
            results[2,0] = Model.SharpeRatio(R_FF3_GMVP_in_sample)
            if TOOS > 0:
                results[2,1] = Model.SharpeRatio(R_FF3_GMVP_OOS)
            results[3,0] = Model.SharpeRatio(R_FF3_regulation_T_in_sample)
            if TOOS > 0:
                results[3,1] = Model.SharpeRatio(R_FF3_regulation_T_OOS)
            results[4,0] = Model.SharpeRatio(R_FF6_in_sample)
            if TOOS > 0:
                results[4,1] = Model.SharpeRatio(R_FF6_OOS)
            results[5,0] = Model.SharpeRatio(R_FF6_GMVP_in_sample)
            if TOOS > 0:
                results[5,1] = Model.SharpeRatio(R_FF6_GMVP_OOS)
            results[6,0] = Model.SharpeRatio(R_FF6_regulation_T_in_sample)
            if TOOS > 0:
                results[6,1] = Model.SharpeRatio(R_FF6_regulation_T_OOS)
            results[7,0] = Model.SharpeRatio(R_AQR6_in_sample)
            if TOOS > 0:
                results[7,1] = Model.SharpeRatio(R_AQR6_OOS)
            results[8,0] = Model.SharpeRatio(R_AQR6_GMVP_in_sample)
            if TOOS > 0:
                results[8,1] = Model.SharpeRatio(R_AQR6_GMVP_OOS)
            results[9,0] = Model.SharpeRatio(R_AQR6_regulation_T_in_sample)
            if TOOS > 0:
                results[9,1] = Model.SharpeRatio(R_AQR6_regulation_T_OOS)
            results[10,0] = Model.SharpeRatio(R_BMA_full_in_sample)
            if TOOS > 0:
                results[10,1] = Model.SharpeRatio(R_BMA_full_OOS)
            results[11,0] = Model.SharpeRatio(R_BMA_GMVP_full_in_sample)
            if TOOS > 0:
                results[11,1] = Model.SharpeRatio(R_BMA_GMVP_full_OOS)
            results[12,0] = Model.SharpeRatio(R_BMA_full_regulation_T_in_sample)
            if TOOS > 0:
                results[12,1] = Model.SharpeRatio(R_BMA_full_regulation_T_OOS)
        # ===== End of not only conditional =====
        results[13,0] = Model.SharpeRatio(R_BMA_diagonal_in_sample)
        if TOOS > 0:
            results[13,1] = Model.SharpeRatio(R_BMA_diagonal_OOS)
        results[14,0] = Model.SharpeRatio(R_BMA_GMVP_diagonal_in_sample)
        if TOOS > 0:
            results[14,1] = Model.SharpeRatio(R_BMA_GMVP_diagonal_OOS)
        results[15,0] = Model.SharpeRatio(R_BMA_diagonal_regulation_T_in_sample)
        if TOOS > 0:
            results[15,1] = Model.SharpeRatio(R_BMA_diagonal_regulation_T_OOS)
        results[16,0] = Model.SharpeRatio(R_BMA_static_cov_in_sample)
        if TOOS > 0:
            results[16,1] = Model.SharpeRatio(R_BMA_static_cov_OOS)
        results[17,0] = Model.SharpeRatio(R_BMA_GMVP_static_cov_in_sample)
        if TOOS > 0:
            results[17,1] = Model.SharpeRatio(R_BMA_GMVP_static_cov_OOS)
        results[18,0] = Model.SharpeRatio(R_BMA_static_cov_regulation_T_in_sample)
        if TOOS > 0:
            results[18,1] = Model.SharpeRatio(R_BMA_static_cov_regulation_T_OOS)

        return results

    # ========== End of Method AnalyseInSampleAndOOSPortfolioReturns ==========

    # This method recevies the MKT returns, facotrs returns, tau and KMax and calculartes T0.
    @staticmethod
    def calculateT0(R_MKT, F, tau, KMax):
            SRMKT = np.mean(R_MKT)/np.std(R_MKT)
            F_mean = np.mean(F, axis=0)
            k = F.shape[1]
            F_mean.reshape(k,1)
            
            V_F = np.cov(F, rowvar = False, bias = False).reshape(k, k)
            SR2Max = np.sum(np.transpose(F_mean) @ LA.pinv(V_F) @ F_mean)
            
            T0 = ( KMax - k ) *( 1 + SR2Max ) /( ( tau**2 - 1 ) * SRMKT**2 )
            
            return int(T0)

    # ========== End of method calculateT0 ==========
    
    # This method recevies the mothly returns and returns the annual Sharpe ratio.
    @staticmethod
    def SharpeRatio(r):        
        return np.mean(r)/np.sqrt(np.cov(r,rowvar=False, bias=False))*np.sqrt(12)

    # ========== End of Method SharpeRatio ==========

    @staticmethod
    def unconditional_model_portfolio_returns(r_observed, r_estimated):
        N = r_estimated.shape[1]
        assert r_estimated.shape[1] == r_observed.shape[1]

        F = r_estimated
        FMean = np.mean(F, axis = 0)
        Vf = np.cov(F, rowvar = False, bias = False).reshape(N, N)
        w_unconditional = LA.pinv(Vf) @ FMean
        w_unconditional = w_unconditional/ (np.abs(np.sum(w_unconditional)))
        R_unconditional_portfolio_returns = r_observed @ w_unconditional
        
        print("In unconditional_model_portfolio_returns: sum w_i= %f  sum|w_i|= %f " \
            %(np.sum(w_unconditional), np.sum(np.abs(w_unconditional))))
        
        return R_unconditional_portfolio_returns

    # ========== End of method unconditional_model_portfolio_returns ==========

    @staticmethod
    def GMVP_unconditional_model_portfolio_returns(r_observed, r_estimated):
        N = r_estimated.shape[1]
        assert r_estimated.shape[1] == r_observed.shape[1]
        iotaN = np.ones((N,1), dtype=np.float64)

        F = r_estimated
        FMean = np.mean(F, axis = 0)
        Vf = np.cov(F, rowvar = False, bias = False).reshape(N, N)
        Vf_inv = LA.pinv(Vf)
        w_GMVP = Vf_inv @ iotaN / (np.transpose(iotaN) @ Vf_inv @ iotaN)
        R_GMVP_portfolio_returns = r_observed @ w_GMVP
        
        print("In GMVP_unconditional_model_portfolio_returns: sum w_i= %f  sum|w_i|= %f " \
            %(np.sum(w_GMVP), np.sum(np.abs(w_GMVP))))
        
        return R_GMVP_portfolio_returns

    # ========== End of method GMVP_unconditional_model_portfolio_returns ==========


    @staticmethod
    def BMA_diagonal_cov_portfolio_returns(r_observed, r_estimated, r_square_estimated, variance_matrix):
        T = r_observed.shape[0]
        N = r_observed.shape[1]
        assert (r_observed.shape[1] == r_estimated.shape[1]) and (r_estimated.shape == r_square_estimated.shape)
        assert variance_matrix.shape == (T, N, N)
        r_portfolio = np.zeros((T,), dtype=np.float64)
        w_conditional = np.zeros((T,N), dtype=np.float64)
        variance_matrix_ratio = np.zeros((T,), dtype=np.float64)
        variance_matrix_ratio_contribution = np.zeros((T,N), dtype=np.float64)
        sum_w = np.zeros((T,), dtype=np.float64)
        sum_abs_w = np.zeros((T,), dtype=np.float64)
        sum_w_before = np.zeros((T,), dtype=np.float64)
        sum_abs_w_before = np.zeros((T,), dtype=np.float64)
        iotaN = np.ones((N,1), dtype=np.float64)
        r_GMVP_portfolio = np.zeros((T,), dtype=np.float64)
        w_GMVP = np.zeros((T,N), dtype=np.float64)

        for t in np.arange(0,T):
            # Static plus conditional ccovariance matrix.
            cov_matrix_t = variance_matrix[t,:,:] + np.diag(r_square_estimated[t,:] - r_estimated[t,:]**2)
            # GMVP portfolio returns.
            cov_matrix_t_inv = LA.pinv(cov_matrix_t)
            w_GMVP_t = cov_matrix_t_inv @ iotaN/ (np.transpose(iotaN) @ cov_matrix_t_inv @ iotaN)
            r_GMVP_portfolio[t] =  np.dot(w_GMVP_t.reshape(-1), r_observed[t,:])
            w_GMVP[t,:] = w_GMVP_t.reshape(-1,)
            del w_GMVP_t
            # END GMVP portfolio returns.
            w_conditional_t = LA.pinv(cov_matrix_t) @ r_estimated[t,:]
            sum_w_before[t] = np.sum(w_conditional_t)
            sum_abs_w_before[t] = np.sum(np.abs(w_conditional_t))

            w_conditional_t = w_conditional_t/ (np.abs(np.sum(w_conditional_t)))
            r_portfolio[t] =  np.dot(w_conditional_t[:], r_observed[t,:])
            w_conditional[t,:] = w_conditional_t
            sum_w[t] = np.sum(w_conditional_t)
            sum_abs_w[t] = np.sum(np.abs(w_conditional_t))
            del w_conditional_t
            det_variance_matrix_t = LA.det(variance_matrix[t,:,:])
            variance_matrix_ratio[t] = det_variance_matrix_t/LA.det(cov_matrix_t)
            omega_t = copy.deepcopy(cov_matrix_t - variance_matrix[t,:,:])
            for n in np.arange(0, N):
                cov_matrix_t = copy.deepcopy(variance_matrix[t,:,:])
                cov_matrix_t[n,n] += omega_t[n,n]
                variance_matrix_ratio_contribution[t,n] = LA.det(cov_matrix_t)/det_variance_matrix_t
                del cov_matrix_t

        print("In BMA_diagonal_cov_portfolio_returns: sum w_i_before: min, max, avg  %f %f %f ,sum w_i= %f" \
            %(np.min(sum_w_before), np.max(sum_w_before), np.mean(sum_w_before), np.mean(sum_w)))
        
        print("In BMA_diagonal_cov_portfolio_returns: min, max, avg, sum|w_i|= %f %f %f " \
            %(np.min(sum_abs_w), np.max(sum_abs_w), np.mean(sum_abs_w)))

        print("In BMA_diagonal_cov_portfolio_returns - variance_matrix_ratio: min, max, avg= %f %f %f " \
            %(np.min(variance_matrix_ratio), np.max(variance_matrix_ratio), np.mean(variance_matrix_ratio)))
        
        return (r_portfolio, w_conditional, variance_matrix_ratio, variance_matrix_ratio_contribution, \
            r_GMVP_portfolio, w_GMVP)

    # ========== End of method BMA_diagonal_cov_portfolio_returns ==========

    @staticmethod
    def diagonal_cov_regulation_T_portfolio_returns(r_observed, r_estimated, r_square_estimated, variance_matrix, gamma):
        T = r_observed.shape[0]
        N = r_observed.shape[1]
        assert (r_observed.shape[1] == r_estimated.shape[1]) and (r_estimated.shape == r_square_estimated.shape)
        assert variance_matrix.shape == (N, N) or variance_matrix.shape == (T, N, N)
        #if variance_matrix.shape == (N, N):
        #    temp_variance_matrix = copy.deepcopy(variance_matrix)
        #    variance_matrix = np.zeros((T,N,N), dtype=np.float64)
        #    for t in np.arange(0,T):
        #        variance_matrix[t,:,:] = temp_variance_matrix
        #else:
        #    assert variance_matrix.shape == (T, N, N)

        r_portfolio = np.zeros((T,), dtype=np.float64)
        A = np.concatenate((-np.eye(2*N), np.ones((1, 2*N), dtype=np.float64)),axis=0)
        B = np.concatenate((np.zeros(2*N),np.array([2])), axis=0)
        sum_w = np.zeros((T,), dtype=np.float64)
        sum_abs_w = np.zeros((T,), dtype=np.float64)
        for t in np.arange(0,T):
            if variance_matrix.shape == (T, N, N) or (variance_matrix.shape == (N, N) and t == 0):
                # Conditonal models.
                if variance_matrix.shape == (T, N, N):
                    cov_matrix_t = variance_matrix[t,:,:] + np.diag(r_square_estimated[t,:] - r_estimated[t,:]**2)
                # Unconditonal models.
                elif (variance_matrix.shape == (N, N) and t == 0):
                    cov_matrix_t = variance_matrix[:,:] + np.diag(r_square_estimated[t,:] - r_estimated[t,:]**2)

                cov_matrix_t = np.concatenate((cov_matrix_t, -cov_matrix_t), axis=1)
                cov_matrix_t = np.concatenate((cov_matrix_t, -cov_matrix_t), axis=0)
                cov_matrix_t *= gamma
                mu_t = np.concatenate((-r_estimated[t,:], r_estimated[t,:]), axis=0)

                x = cvxopt_solve_qp(cov_matrix_t, mu_t, A, B)

                w_conditional_t = x[0 : N] - x[N : ]
                
            r_portfolio[t] =  np.dot(w_conditional_t, r_observed[t,:])
            sum_w[t] = np.sum(w_conditional_t)
            sum_abs_w[t] = np.sum(np.abs(w_conditional_t))
    
        print("In diagonal_cov_regulation_T_portfolio_returns gamma= %f: sum w_i= %f  min, max, avg, sum|w_i|= %f %f %f " \
             %(gamma, np.mean(sum_w), np.min(sum_abs_w), np.max(sum_abs_w), np.mean(sum_abs_w)))

        return r_portfolio

    # ========== End of method diagonal_cov_regulation_T_portfolio_returns ==========   
    @staticmethod
    def BMA_full_cov_portfolio_returns(r_observed, r_estimated, r_square_estimated, \
                                        returns_interactions_estimated, variance_matrix):
        T = r_observed.shape[0]
        N = r_observed.shape[1]
        assert (r_observed.shape[1] == r_estimated.shape[1]) and (r_estimated.shape == r_square_estimated.shape)
        assert variance_matrix.shape == (T, N, N)
        r_portfolio = np.zeros((T,), dtype=np.float64)
        w_conditional = np.zeros((T,N), dtype=np.float64)
        variance_matrix_ratio = np.zeros((T,), dtype=np.float64)
        variance_matrix_ratio_contribution = np.zeros((T,N), dtype=np.float64)
        sum_w = np.zeros((T,), dtype=np.float64)
        sum_abs_w = np.zeros((T,), dtype=np.float64)
        sum_w_before = np.zeros((T,), dtype=np.float64)
        sum_abs_w_before = np.zeros((T,), dtype=np.float64)
        iotaN = np.ones((N,1), dtype=np.float64)
        r_GMVP_portfolio = np.zeros((T,), dtype=np.float64)
        w_GMVP = np.zeros((T,N), dtype=np.float64)
        cov_matrix_TS_avg = np.zeros((N,N), dtype=np.float64)
        omega_TS_avg = np.zeros((N,N), dtype=np.float64)

        for t in np.arange(0,T):
            assert np.allclose(np.diag(returns_interactions_estimated[t,:,:]),r_square_estimated[t,:])
            #cov_matrix_t = variance_matrix_static + np.diag(r_square_estimated[t,:] - r_estimated[t,:]**2)
            cov_matrix_t = variance_matrix[t,:,:] + \
                returns_interactions_estimated[t,:,:] - r_estimated[t,:].reshape(-1,1) @ r_estimated[t,:].reshape(1,-1)
            cov_matrix_TS_avg += cov_matrix_t
            # GMVP portfolio returns.
            cov_matrix_t_inv = LA.pinv(cov_matrix_t)
            w_GMVP_t = cov_matrix_t_inv @ iotaN/ (np.transpose(iotaN) @ cov_matrix_t_inv @ iotaN)
            r_GMVP_portfolio[t] =  np.dot(w_GMVP_t.reshape(-1), r_observed[t,:])
            w_GMVP[t,:] = w_GMVP_t.reshape(-1,)
            del w_GMVP_t
            # END GMVP portfolio returns.
            w_conditional_t = LA.pinv(cov_matrix_t) @ r_estimated[t,:]
            sum_w_before[t] = np.sum(w_conditional_t)
            sum_abs_w_before[t] = np.sum(np.abs(w_conditional_t))

            w_conditional_t = w_conditional_t/ (np.abs(np.sum(w_conditional_t)))
            r_portfolio[t] =  np.dot(w_conditional_t[:], r_observed[t,:])
            w_conditional[t,:] = w_conditional_t
            sum_w[t] = np.sum(w_conditional_t)
            sum_abs_w[t] = np.sum(np.abs(w_conditional_t))
            det_variance_matrix_t = LA.det(variance_matrix[t,:,:])
            variance_matrix_ratio[t] = det_variance_matrix_t/LA.det(cov_matrix_t)
            omega_t = copy.deepcopy(cov_matrix_t - variance_matrix[t,:,:])
            omega_TS_avg += omega_t
            for n in np.arange(0, N):
                cov_matrix_t = copy.deepcopy(variance_matrix[t,:,:])
                cov_matrix_t[n,n] += omega_t[n,n]
                variance_matrix_ratio_contribution[t,n] = LA.det(cov_matrix_t)/det_variance_matrix_t
                del cov_matrix_t

        cov_matrix_TS_avg /= T
        omega_TS_avg /= T

        print("In BMA_full_cov_portfolio_returns: sum w_i_before: min, max, avg  %f %f %f ,sum w_i= %f" \
            %(np.min(sum_w_before), np.max(sum_w_before), np.mean(sum_w_before), np.mean(sum_w)))
        
        print("In BMA_full_cov_portfolio_returns: min, max, avg, sum|w_i|= %f %f %f " \
            %(np.min(sum_abs_w), np.max(sum_abs_w), np.mean(sum_abs_w)))

        print("In BMA_full_cov_portfolio_returns - variance_matrix_ratio: min, max, avg= %f %f %f " \
            %(np.min(variance_matrix_ratio), np.max(variance_matrix_ratio), np.mean(variance_matrix_ratio)))
        
        return (r_portfolio, w_conditional, variance_matrix_ratio, variance_matrix_ratio_contribution, \
            r_GMVP_portfolio, w_GMVP, cov_matrix_TS_avg, omega_TS_avg)

    # ========== End of method BMA_full_cov_portfolio_returns ==========

    @staticmethod
    def full_cov_regulation_T_portfolio_returns(r_observed, r_estimated, r_square_estimated, \
                                                returns_interactions_estimated, variance_matrix, gamma):
        T = r_observed.shape[0]
        N = r_observed.shape[1]
        assert (r_observed.shape[1] == r_estimated.shape[1]) and (r_estimated.shape == r_square_estimated.shape)
        assert variance_matrix.shape == (T, N, N)

        r_portfolio = np.zeros((T,), dtype=np.float64)
        A = np.concatenate((-np.eye(2*N), np.ones((1, 2*N), dtype=np.float64)),axis=0)
        B = np.concatenate((np.zeros(2*N),np.array([2])), axis=0)
        sum_w = np.zeros((T,), dtype=np.float64)
        sum_abs_w = np.zeros((T,), dtype=np.float64)
        for t in np.arange(0,T):
            assert np.allclose(np.diag(returns_interactions_estimated[t,:,:]),r_square_estimated[t,:])
            #cov_matrix_t = variance_matrix_static + np.diag(r_square_estimated[t,:] - r_estimated[t,:]**2)
            cov_matrix_t = variance_matrix[t,:,:] + \
                returns_interactions_estimated[t,:,:] - r_estimated[t,:].reshape(-1,1) @ r_estimated[t,:].reshape(1,-1)

            cov_matrix_t = np.concatenate((cov_matrix_t, -cov_matrix_t), axis=1)
            cov_matrix_t = np.concatenate((cov_matrix_t, -cov_matrix_t), axis=0)
            cov_matrix_t *= gamma
            mu_t = np.concatenate((-r_estimated[t,:], r_estimated[t,:]), axis=0)

            x = cvxopt_solve_qp(cov_matrix_t, mu_t, A, B)

            w_conditional_t = x[0 : N] - x[N : ]
            r_portfolio[t] =  np.dot(w_conditional_t, r_observed[t,:])
            sum_w[t] = np.sum(w_conditional_t)
            sum_abs_w[t] = np.sum(np.abs(w_conditional_t))
    
        print("In full_cov_regulation_T_portfolio_returns gamma= %f: sum w_i= %f  min, max, avg, sum|w_i|= %f %f %f " \
             %(gamma, np.mean(sum_w), np.min(sum_abs_w), np.max(sum_abs_w), np.mean(sum_abs_w)))

        return r_portfolio

    # ========== End of method full_cov_regulation_T_portfolio_returns ==========   

# ========== End of Model Class ==========

# ======================================================================================================================
def conditionalAssetPricingLogMarginalLikelihoodTauNew(ROrig, FOrig, ZOrig, OmegaOrig, Tau, SR2Mkt):

    print("calculating both unrestricted and restricted models")
    print("key_Avoid_duplicate_factors= %d, key_Avoid_duplicate_predictors= %d, key_use_combination_matrix= %d " \
          % (key_Avoid_duplicate_factors, key_Avoid_duplicate_predictors, key_use_combination_matrix))
    print("Tau= %f " % (Tau))

    # Moving on to calculating the marginal likelihood.
    # Trying to work with nupy arrays instead of pandas dataframe.
    KMax = FOrig.shape[1]
    MMax = ZOrig.shape[1]
    KMaxPlusMMax = KMax + MMax
    T = FOrig.shape[0]

    # Market squared Sharpe ratio and maximum difference between the upper bound Sharpe ratio and the market.
    # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
    Tau2m1SR2Mkt = (pow(Tau, 2) - 1) * SR2Mkt

    nModelsMax = pow(2, KMax + MMax)

    # Variables initialization.
    logMarginalLikelihood = np.zeros((nModelsMax,), dtype=np.float64)
    logMarginalLikelihood.fill(-np.inf)
    # Placeholder for the restricted models.
    logMarginalLikelihoodR = np.zeros((nModelsMax,), dtype=np.float64)
    logMarginalLikelihoodR.fill(-np.inf)

    nTooSmallT0 = 0
    nLegitModels = 0
    T0Total = 0
    T0Max = 0
    T0Min = np.inf

    AllFactorsSet = set(np.arange(0, KMax))
    iotaT = np.ones((T, 1), dtype=float)

    totalTime = 0.0
    nprintFraction = pow(10, 1 + (KMaxPlusMMax > 20))
    tictoc.tic()
    mStart = 0
    if keyDebug:
        mStart = int(nModelsMax * 3 / 4 * .98)
        mStart = 102050547

    nModelsMax = 6
    print('First model is %i. Total numbers of models is %i. nprintFraction= %d' % (mStart, nModelsMax, nprintFraction))
    for model in np.arange(mStart, nModelsMax):
        if model % np.floor(nModelsMax / nprintFraction) == 0:
            totalTime += tictoc.toc(False)
            print('Done %3d %% of total work at %12.2f sec. model= %11i, nModelsMax= %11i'
                  % (100 * model / nModelsMax, totalTime, model, nModelsMax))
            # Dump
            if nModelsMax > 100000:
                with open('local_dump.pkl', 'bw') as file:
                    pickle.dump(
                        [model, nModelsMax, KMax, MMax, factorsNames, predictorsNames, \
                        significantPredictors, Tau, \
                        nTooSmallT0, nLegitModels, T0Max, T0Total, logMarginalLikelihood, logMarginalLikelihoodR], file)

        combinationsRowFromMatrix = retreiveRowFromAllCombinationsMatrixNumba(model, KMaxPlusMMax)
        factorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[0: KMax] == 1).flatten()
        predictorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[KMax:] == 1).flatten()

        otherFactors = np.array(list(AllFactorsSet - set(factorsIndicesIncludedInModel)), dtype=int)

        # MKT is not in model assign a 0 probability. Total number of combinations is 2^(KMax-1)*2^MMax.
        # However in case of all factors in the test assets continue as linear predictive regression.
        # Total number of combinations is 2^MMax.
        if not (1 - 1 in factorsIndicesIncludedInModel) and len(factorsIndicesIncludedInModel) != 0:
            # logMarginalLikelihood[model] = -np.inf
            continue

        if key_Avoid_duplicate_predictors:
            # Add restriction when there is a linear dependency between the predictors.
            # Not including the following three predictors together: dp, ep, de.
            # predictors 1, 3, 4 out of the 8 combinations of including the
            # predictors only 5 are independent:
            # 1 - none of them, 3 - only one predictor and 1 - one pair out of the three
            # possible pairs.
            # and the tbl, lty, tms. predictors 8, 9, 11

            if (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel) or \
                    (9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel):
                # logMarginalLikelihood[model] = -np.inf

                continue

        # Avoid models with duplicate factors.
        if key_Avoid_duplicate_factors:
            if (2 - 1 in factorsIndicesIncludedInModel and 8 - 1 in factorsIndicesIncludedInModel) or \
                    (4 - 1 in factorsIndicesIncludedInModel and 10 - 1 in factorsIndicesIncludedInModel) or \
                    (5 - 1 in factorsIndicesIncludedInModel and 9 - 1 in factorsIndicesIncludedInModel):

                # logMarginalLikelihood[model] = -np.inf

                if keyPrint:
                    print('duplicate factors!!! factorsIndicesIncludedInModel= ')
                    print(factorsIndicesIncludedInModel)

                continue

        # Each model has different N K R and F. Remove the temporary values at the end of each cycle.
        # del N, K, F, R

        if len(ROrig) == T:
            R = np.concatenate((ROrig[:, :], \
                                FOrig[:, otherFactors]), axis=1)
        else:
            R = FOrig[:, otherFactors]

        N = R.shape[1]

        # No test assets. 2^MMax combinations.
        if N == 0:
            # logMarginalLikelihood[model] = -np.inf
            continue

        nLegitModels += 1

        F = FOrig[:, factorsIndicesIncludedInModel]
        K = F.shape[1]
        Z = ZOrig[:, predictorsIndicesIncludedInModel]
        M = Z.shape[1]

        if M == 0:
            Z = np.empty((T, 0), dtype=float)

        assert R.shape[0] == T and F.shape[0] == T and Z.shape[0] == T

        OmegaIndecies = np.repeat(factorsIndicesIncludedInModel * MMax, M) + np.tile(predictorsIndicesIncludedInModel,
                                                                                     K)
        X = np.concatenate((iotaT, Z), axis=1)

        if keyDebug:
            Omega = np.zeros((T, K * M), dtype=float)
            for t in np.arange(0, T):
                Omega[t, :] = np.kron(np.identity(K), Z[t, :].reshape(-1, 1)) @ F[t, :]

            assert np.allclose(Omega, OmegaOrig[:, OmegaIndecies])

        Omega = OmegaOrig[:, OmegaIndecies]

        W = np.concatenate((np.concatenate((X, F), axis=1), Omega), axis=1)

        RMean = np.mean(R, 0).reshape(-1, 1)
        FMean = np.mean(F, 0).reshape(-1, 1)
        ZMean = np.mean(Z, 0).reshape(-1, 1)

        Vf = np.cov(F, rowvar=False, bias=True).reshape(K, K)

        # Hypothetical sample quantities.

        beta0 = LA.pinv(np.transpose(F) @ F) @ np.transpose(F) @ R

        if keyDebug:
            beta01 = LA.lstsq(F, R, rcond=None)[0]  # Should be the same as beta0, just checking.
            assert np.allclose(beta0, beta01)

            Af0 = LA.pinv(np.transpose(iotaT) @ iotaT) @ np.transpose(iotaT) @ F  # Should equal FMean, just checking.
            assert np.allclose(Af0, np.transpose(FMean))

        Af0 = np.concatenate((FMean, np.zeros((K, M), dtype=float)), axis=1)
        XtX = np.transpose(X) @ X
        XtXInv = LA.pinv(XtX)
        WtW = np.transpose(W) @ W
        # WtWInv  = LA.lstsq(WtW , np.identity(1 + K, dtype=float), rcond=None)[0]

        try:
            WtWInv = LA.pinv(WtW)
        except np.linalg.LinAlgError as e:
            print('Error %s in model number %i' % (e, model))
            print(factorsIndicesIncludedInModel)
            print(predictorsIndicesIncludedInModel)
            string = 'Trying normal inverse ... '
            try:
                WtWInv = LA.inv(WtW)
                print(string + 'LA.inv worked')
            except:
                string += 'Failed trying scipy pinvh'
                try:
                    WtWInv = SPLA.pinvh(WtW)
                    print(string + 'SPLA.pinvh worked')
                except:
                    print(string + 'Failed ')

        Qw = np.identity(T, dtype=float) - W @ WtWInv @ np.transpose(W)
        # SigmaRR = (np.transpose(R) @ Qw @ R) / (T - M - K - K*M - 1) / pow(100,2) # changing from percentage to numbers.

        # Since the distribution is known the maximum likelihood estimator is without the bias correction.
        SigmaRR = (np.transpose(R) @ Qw @ R) / (T) / pow(100, 2)  # changing from percentage to numbers.

        SR2 = np.transpose(FMean) @ LA.pinv(Vf) @ FMean  # factors Sharpe ratio square.

        T0 = int(np.round((N * (1 + SR2 + M * (1 + SR2))) / Tau2m1SR2Mkt))
        #print(N, K, M, T0, SR2)

        # MultiGamma(p(a)) is defined for 2*a > p - 1
        T0LowerBound = max(N + (K + 1) * M, K + M - N)
        if T0 <= T0LowerBound:
            nTooSmallT0 += 1
            if keyPrint:
                print('T0 ( %i ) too small so increasing it to minmum acceptable value ( %i )' % (T0, T0LowerBound + 1))
            T0 = T0LowerBound + 1

        FtF = np.transpose(F) @ F
        RtR = np.transpose(R) @ R
        XtF = np.transpose(X) @ F
        WtR = np.transpose(W) @ R

        S0 = T0 / T * (RtR - np.transpose(beta0) @ (FtF) @ beta0)
        Sf0 = T0 * Vf

        Tstar = T0 + T
        FMeanFMeanZMeant = np.concatenate((FMean, FMean @ np.transpose(ZMean)), axis=1)

        AfTilda = T / Tstar * XtXInv @ (XtF + \
                                        T0 * np.transpose(FMeanFMeanZMeant))
        Sf = Tstar * (Vf + FMean @ np.transpose(FMean)) \
             - T / Tstar * (T0 * FMeanFMeanZMeant + np.transpose(XtF)) @ \
             XtXInv @ \
             (T0 * np.transpose(FMeanFMeanZMeant) + XtF)

        #        if keyRestricted !=1:
        # Calculating the unrestricted log marginal likelihood.
        phi0 = np.transpose(np.concatenate((
            np.concatenate((np.zeros((N, M + 1), dtype=float), np.transpose(beta0)), axis=1), \
            np.zeros((N, K * M), dtype=float)), axis=1))

        phiTilda = T / Tstar * WtWInv @ (WtR + T0 / T * WtW @ phi0)

        RminusWphiTilda = R - W @ phiTilda

        Sr = S0 + np.transpose(RminusWphiTilda) @ (RminusWphiTilda) + \
             T0 / T * np.transpose(phiTilda - phi0) @ WtW @ (phiTilda - phi0)
        if keyDebug:
            phiHat = WtWInv @ WtR
            SrHat = np.transpose(R - W @ phiHat) @ (R - W @ phiHat)
            SrNew = S0 + SrHat + np.transpose(phiHat) @ WtW @ phiHat + \
                    T0 / T * np.transpose(phi0) @ WtW @ phi0 - Tstar / T * np.transpose(phiTilda) @ WtW @ phiTilda

            SrNew1 = Tstar / T * (RtR - np.transpose(phiTilda) @ WtW @ phiTilda)

            assert np.allclose(SrNew, Sr) and np.allclose(Sr, SrNew1)

        logMarginalLikelihood[model] = -T * (N + K) / 2 * np.log(np.pi) \
                                       + (K * (M + 1) + N * (1 + M + K + K * M)) / 2 * np.log(T0 / Tstar) \
                                       + multigammaln((Tstar - (K + 1) * M - 1) / 2, N) \
                                       - multigammaln((T0 - (K + 1) * M - 1) / 2, N) \
                                       + multigammaln((Tstar + N - M - 1) / 2, K) \
                                       - multigammaln((T0 + N - M - 1) / 2, K) \
                                       + (T0 - (K + 1) * M - 1) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                       - (Tstar - (K + 1) * M - 1) / 2 * (
                                                   np.log(LA.det(Sr / Tstar)) + len(Sr) * np.log(Tstar)) \
                                       + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                       - (Tstar + N - M - 1) / 2 * (
                                                   np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

        # else:
        # Calculating the restricted log marginal likelihood.
        phi0R = np.transpose(np.concatenate((np.transpose(beta0), np.zeros((N, K * M), dtype=float)), axis=1))
        WR = np.concatenate((F, Omega), axis=1)
        WRtWR = np.transpose(WR) @ WR
        try:
            WRtWRInv = LA.pinv(WRtWR)
        except np.linalg.LinAlgError as e:
            print('Error %s in model number %i' % (e, model))
            # print(WRtWR[0,:])
            print(factorsIndicesIncludedInModel)
            print(predictorsIndicesIncludedInModel)
            # logMarginalLikelihoodR[model] = -np.inf
            string = 'Trying normal inverse ... '
            try:
                WRtWRInv = LA.inv(WRtWR)
                print(string + 'LA.inv worked')
            except:
                string += 'Failed trying scipy pinvh'
                try:
                    WRtWRInv = SPLA.pinvh(WRtWR)
                    print(string + 'SPLA.pinvh worked')
                except:
                    print(string + 'Failed assaining NINF')
                    continue

        phiTildaR = T / Tstar * WRtWRInv @ (np.transpose(WR) @ R + T0 / T * WRtWR @ phi0R)

        SrR = S0 + np.transpose(R - WR @ phiTildaR) @ (R - WR @ phiTildaR) + \
              T0 / T * np.transpose(phiTildaR - phi0R) @ WRtWR @ (phiTildaR - phi0R)

        if keyDebug:
            phiHatR = WRtWRInv @ np.transpose(WR) @ R
            SrRNew1 = Tstar / T * (RtR - np.transpose(phiTildaR) @ WRtWR @ phiTildaR)
            assert np.allclose(SrR, SrRNew1)

        logMarginalLikelihoodR[model] = -T * (N + K) / 2 * np.log(np.pi) \
                                        + (K * (M + 1) + N * (K + K * M)) / 2 * np.log(T0 / Tstar) \
                                        + multigammaln((Tstar - K * M) / 2, N) \
                                        - multigammaln((T0 - K * M) / 2, N) \
                                        + multigammaln((Tstar + N - M - 1) / 2, K) \
                                        - multigammaln((T0 + N - M - 1) / 2, K) \
                                        + (T0 - K * M) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                        - (Tstar - K * M) / 2 * (np.log(LA.det(SrR / Tstar)) + len(SrR) * np.log(Tstar)) \
                                        + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                        - (Tstar + N - M - 1) / 2 * (
                                                    np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

        #[logMarginalLikelihood[model], logMarginalLikelihoodR[model]] = calclogMarginalLikelihood(N, K, M, T, T0, Tstar,
        #                                                                                          S0, Sr, SrR, Sf0, Sf)

        T0Total = T0Total + T0
        T0Max = max(T0Max, T0)
        T0Min = min(T0Min, T0)
        del N, K, F, R, Z, M

    tictoc.toc()

    T0IncreasedFraction = nTooSmallT0 / nLegitModels
    T0Avg = T0Total / nLegitModels

    print('All combinations= %d Total number of legit models= %d 2ND count %d ' \
          % (nModelsMax, np.count_nonzero(logMarginalLikelihood != -np.inf), nLegitModels))

    print('# of times T0 was increased= %d T0 Average= %f T0 Max= %f T0 Min= %f' \
          % (nTooSmallT0, T0Avg, T0Max, T0Min))

    return (logMarginalLikelihood, logMarginalLikelihoodR, T0IncreasedFraction, T0Max, T0Min, T0Avg)

# ========== End of function conditionalAssetPricingLogMarginalLikelihoodTauNew ==========
# ======================================================================================================================

def conditionalAssetPricingLogMarginalLikelihoodTauNumba(ROrig, FOrig, ZOrig, OmegaOrig, Tau, SR2Mkt):

    key_Avoid_duplicate_predictors = 1
    #print("calculating both unrestricted and restricted models")
    #print("key_Avoid_duplicate_factors= %d, key_Avoid_duplicate_predictors= %d, key_use_combination_matrix= %d " \
    #      % (key_Avoid_duplicate_factors, key_Avoid_duplicate_predictors, key_use_combination_matrix))
    #print("Tau= %f " % (Tau))

    # Moving on to calculating the marginal likelihood.
    # Trying to work with nupy arrays instead of pandas dataframe.
    KMax = FOrig.shape[1]
    MMax = ZOrig.shape[1]
    KMaxPlusMMax = KMax + MMax
    T = FOrig.shape[0]

    # Market squared Sharpe ratio and maximum difference between the upper bound Sharpe ratio and the market.
    # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
    Tau2m1SR2Mkt = (pow(Tau, 2) - 1) * SR2Mkt

    nModelsMax = pow(2, KMax + MMax)

    # Variables initialization.
    logMarginalLikelihood = np.zeros((nModelsMax,), dtype=np.float64)
    logMarginalLikelihood.fill(-np.inf)
    # Placeholder for the restricted models.
    logMarginalLikelihoodR = np.zeros((nModelsMax,), dtype=np.float64)
    logMarginalLikelihoodR.fill(-np.inf)

    nTooSmallT0 = 0
    nLegitModels = 0
    T0Total = 0
    T0Max = 0
    T0Min = np.inf
    T0_div_T0_plus_T = 0
    T_div_T0_plus_T = 0

    iotaT = np.ones((T, 1), dtype=np.float64)
    #
    # totalTime = 0.0
    nprintFraction = pow(10, 1 + (KMaxPlusMMax > 20))
    # tictoc.tic()
    mStart = 0

    for model in np.arange(mStart, nModelsMax):
        if model % max(np.floor(nModelsMax / nprintFraction),100) == 0:
            print("Model  " + str(model) + " out of " + str(nModelsMax))

        combinationsRowFromMatrix = retreiveRowFromAllCombinationsMatrixNumba(model, KMaxPlusMMax)
        factorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[0: KMax] == 1).flatten()
        predictorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[KMax:] == 1).flatten()


        otherFactors = findOtherFactors(KMax, factorsIndicesIncludedInModel)
        #print(otherFactors)

        # MKT is not in model assign a 0 probability. Total number of combinations is 2^(KMax-1)*2^MMax.
        # However in case of all factors in the test assets continue as linear predictive regression.
        # Total number of combinations is 2^MMax.
        if not (1 - 1 in factorsIndicesIncludedInModel) and len(factorsIndicesIncludedInModel) != 0:
    #         # logMarginalLikelihood[model] = -np.inf
            continue
    #
        if key_Avoid_duplicate_predictors:
            # Add restriction when there is a linear dependency between the predictors.
            # Not including the following three predictors together: dp, ep, de.
            # predictors 1, 3, 4 out of the 8 combinations of including the
            # predictors only 5 are independent:
            # 1 - none of them, 3 - only one predictor and 1 - one pair out of the three
            # possible pairs.
            # and the tbl, lty, tms. predictors 8, 9, 11

            if (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel) or \
                    (9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel):

                continue

        # Avoid models with duplicate factors.
        if key_Avoid_duplicate_factors:
            if (2 - 1 in factorsIndicesIncludedInModel and 8 - 1 in factorsIndicesIncludedInModel) or \
                    (4 - 1 in factorsIndicesIncludedInModel and 10 - 1 in factorsIndicesIncludedInModel) or \
                    (5 - 1 in factorsIndicesIncludedInModel and 9 - 1 in factorsIndicesIncludedInModel):

                if keyPrint:
                    print('duplicate factors!!! factorsIndicesIncludedInModel= ')
                    print(factorsIndicesIncludedInModel)

                continue

        # Each model has different N K R and F. Remove the temporary values at the end of each cycle.
        # del N, K, F, R

        if len(ROrig) == T:
            R = np.concatenate((ROrig[:, :], FOrig[:, otherFactors]), axis=1)
        else:
            R = FOrig[:, otherFactors]

        N = R.shape[1]

        # No test assets. 2^MMax combinations.
        if N == 0:
            continue

        nLegitModels += 1

        F = FOrig[:, factorsIndicesIncludedInModel]
        K = F.shape[1]
        Z = ZOrig[:, predictorsIndicesIncludedInModel]
        M = Z.shape[1]

        if M == 0:
            Z = np.empty((T, 0), dtype=np.float64)

        assert R.shape[0] == T and F.shape[0] == T and Z.shape[0] == T

        OmegaIndecies = np.repeat(factorsIndicesIncludedInModel * MMax, M) + \
                        tileNumba(predictorsIndicesIncludedInModel, K)
        X = np.concatenate((iotaT, Z), axis=1)

    #     if keyDebug:
    #         Omega = np.zeros((T, K * M), dtype=float)
    #         for t in np.arange(0, T):
    #             Omega[t, :] = np.kron(np.identity(K), Z[t, :].reshape(-1, 1)) @ F[t, :]
    #
    #         assert np.allclose(Omega, OmegaOrig[:, OmegaIndecies])
    #
        Omega = OmegaOrig[:, OmegaIndecies]
    #
        W = np.concatenate((np.concatenate((X, F), axis=1), Omega), axis=1)

        RMean = np.ascontiguousarray(MeanAlongAxisNumba(R, 0))
        FMean = np.ascontiguousarray(MeanAlongAxisNumba(F, 0))
        ZMean = np.ascontiguousarray(MeanAlongAxisNumba(Z, 0))
        #print(FMean)

        Vf = np.cov(F, rowvar=False, bias=True).reshape(K, K)



        # Hypothetical sample quantities.

        beta0 = LA.pinv(np.transpose(F) @ F) @ np.transpose(F) @ R

    #     if keyDebug:
    #         beta01 = LA.lstsq(F, R, rcond=None)[0]  # Should be the same as beta0, just checking.
    #         assert np.allclose(beta0, beta01)
    #
    #         Af0 = LA.pinv(np.transpose(iotaT) @ iotaT) @ np.transpose(iotaT) @ F  # Should equal FMean, just checking.
    #         assert np.allclose(Af0, np.transpose(FMean))

        Af0 = np.concatenate((FMean, np.zeros((K, M), dtype=np.float64)), axis=1)
        XtX = np.transpose(X) @ X
        XtXInv = LA.pinv(XtX)
        WtW = np.transpose(W) @ W

        #try:
        WtWInv = LA.pinv(WtW)
    #     except np.linalg.LinAlgError as e:
    #         print('Error %s in model number %i' % (e, model))
    #         print(factorsIndicesIncludedInModel)
    #         print(predictorsIndicesIncludedInModel)
    #         string = 'Trying normal inverse ... '
    #         try:
    #             WtWInv = LA.inv(WtW)
    #             print(string + 'LA.inv worked')
    #         except:
    #             string += 'Failed trying scipy pinvh'
    #             try:
    #                 WtWInv = SPLA.pinvh(WtW)
    #                 print(string + 'SPLA.pinvh worked')
    #             except:
    #                 print(string + 'Failed ')
    #
        Qw = np.identity(T) - W @ WtWInv @ np.transpose(W)
        # SigmaRR = (np.transpose(R) @ Qw @ R) / (T - M - K - K*M - 1) / pow(100,2) # changing from percentage to numbers.

        # Since the distribution is known the maximum likelihood estimator is without the bias correction.
        SigmaRR = (np.transpose(R) @ Qw @ R) / (T) / pow(100, 2)  # changing from percentage to numbers.

        SR2 = (np.transpose(FMean) @ LA.pinv(Vf) @ FMean).flatten()  # factors Sharpe ratio square.
        SR2 = np.sum(SR2)

        T0 = int(np.round((N * (1 + SR2 + M * (1 + SR2))) / Tau2m1SR2Mkt))
        # print(N,K,M,T0,SR2, np.transpose(FMean) @ LA.pinv(Vf) @ FMean)

        # MultiGamma(p(a)) is defined for 2*a > p - 1
        #print(str(model) +' '+ str(N)+' ' + str(K)+' ' + str(M)+' ' +str(T0))
        T0LowerBound = max(N + (K + 1) * M, K + M - N)
        if T0 <= T0LowerBound:
            nTooSmallT0 += 1
    #         if keyPrint:
    #             print('T0 ( %i ) too small so increasing it to minmum acceptable value ( %i )' % (T0, T0LowerBound + 1))
            T0 = T0LowerBound + 1

        FtF = np.transpose(F) @ F
        RtR = np.transpose(R) @ R
        XtF = np.transpose(X) @ F
        WtR = np.transpose(W) @ R

        S0 = T0 / T * (RtR - np.transpose(beta0) @ (FtF) @ beta0)
        Sf0 = T0 * Vf

        Tstar = T0 + T
        FMeanFMeanZMeant = np.concatenate((FMean, FMean @ np.transpose(ZMean)), axis=1)

        AfTilda = T / Tstar * XtXInv @ (XtF + T0 * np.transpose(FMeanFMeanZMeant))

        Sf = Tstar * (Vf + FMean @ np.transpose(FMean)) \
             - T / Tstar * (T0 * FMeanFMeanZMeant + np.transpose(XtF)) @ \
             XtXInv @ (T0 * np.transpose(FMeanFMeanZMeant) + XtF)

        # Calculating the unrestricted log marginal likelihood.
        phi0 = np.transpose(np.concatenate((
            np.concatenate((np.zeros((N, M + 1), dtype=np.float64), np.transpose(beta0)), axis=1), \
            np.zeros((N, K * M), dtype=np.float64)), axis=1))

        phiTilda = T / Tstar * WtWInv @ (WtR + T0 / T * WtW @ phi0)

        RminusWphiTilda = R - W @ phiTilda

        Sr = S0 + np.transpose(RminusWphiTilda) @ (RminusWphiTilda) + \
             T0 / T * np.transpose(phiTilda - phi0) @ WtW @ (phiTilda - phi0)
    #     if keyDebug:
    #         phiHat = WtWInv @ WtR
    #         SrHat = np.transpose(R - W @ phiHat) @ (R - W @ phiHat)
    #         SrNew = S0 + SrHat + np.transpose(phiHat) @ WtW @ phiHat + \
    #                 T0 / T * np.transpose(phi0) @ WtW @ phi0 - Tstar / T * np.transpose(phiTilda) @ WtW @ phiTilda
    #
    #         SrNew1 = Tstar / T * (RtR - np.transpose(phiTilda) @ WtW @ phiTilda)
    #
    #         assert np.allclose(SrNew, Sr) and np.allclose(Sr, SrNew1)
    #
        logMarginalLikelihood[model] = -T * (N + K) / 2 * np.log(np.pi) \
                                       + (K * (M + 1) + N * (1 + M + K + K * M)) / 2 * np.log(T0 / Tstar) \
                                       + multigammalnNumba((Tstar - (K + 1) * M - 1) / 2, N) \
                                       - multigammalnNumba((T0 - (K + 1) * M - 1) / 2, N) \
                                       + multigammalnNumba((Tstar + N - M - 1) / 2, K) \
                                       - multigammalnNumba((T0 + N - M - 1) / 2, K) \
                                       + (T0 - (K + 1) * M - 1) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                       - (Tstar - (K + 1) * M - 1) / 2 * (
                                                   np.log(LA.det(Sr / Tstar)) + len(Sr) * np.log(Tstar)) \
                                       + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                       - (Tstar + N - M - 1) / 2 * (
                                                   np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))


        # Calculating the restricted log marginal likelihood.
        phi0R = np.transpose(np.concatenate((np.transpose(beta0), np.zeros((N, K * M), dtype=np.float64)), axis=1))
        WR = np.concatenate((F, Omega), axis=1)
        WRtWR = np.transpose(WR) @ WR
    #     try:
        WRtWRInv = LA.pinv(WRtWR)
    #     except np.linalg.LinAlgError as e:
    #         print('Error %s in model number %i' % (e, model))
    #         # print(WRtWR[0,:])
    #         print(factorsIndicesIncludedInModel)
    #         print(predictorsIndicesIncludedInModel)
    #         # logMarginalLikelihoodR[model] = -np.inf
    #         string = 'Trying normal inverse ... '
    #         try:
    #             WRtWRInv = LA.inv(WRtWR)
    #             print(string + 'LA.inv worked')
    #         except:
    #             string += 'Failed trying scipy pinvh'
    #             try:
    #                 WRtWRInv = SPLA.pinvh(WRtWR)
    #                 print(string + 'SPLA.pinvh worked')
    #             except:
    #                 print(string + 'Failed assaining NINF')
    #                 continue
    #
        phiTildaR = T / Tstar * WRtWRInv @ (np.transpose(WR) @ R + T0 / T * WRtWR @ phi0R)

        SrR = S0 + np.transpose(R - WR @ phiTildaR) @ (R - WR @ phiTildaR) + \
              T0 / T * np.transpose(phiTildaR - phi0R) @ WRtWR @ (phiTildaR - phi0R)

        #if keyDebug:
    #         phiHatR = WRtWRInv @ np.transpose(WR) @ R
    #         SrRNew1 = Tstar / T * (RtR - np.transpose(phiTildaR) @ WRtWR @ phiTildaR)
    #         assert np.allclose(SrR, SrRNew1)
    #
        logMarginalLikelihoodR[model] = -T * (N + K) / 2 * np.log(np.pi) \
                                        + (K * (M + 1) + N * (K + K * M)) / 2 * np.log(T0 / Tstar) \
                                        + multigammalnNumba((Tstar - K * M) / 2, N) \
                                        - multigammalnNumba((T0 - K * M) / 2, N) \
                                        + multigammalnNumba((Tstar + N - M - 1) / 2, K) \
                                        - multigammalnNumba((T0 + N - M - 1) / 2, K) \
                                        + (T0 - K * M) / 2 * (np.log(LA.det(S0 / T0)) + len(S0) * np.log(T0)) \
                                        - (Tstar - K * M) / 2 * (np.log(LA.det(SrR / Tstar)) + len(SrR) * np.log(Tstar)) \
                                        + (T0 + N - M - 1) / 2 * (np.log(LA.det(Sf0 / T0)) + len(Sf0) * np.log(T0)) \
                                        - (Tstar + N - M - 1) / 2 * (np.log(LA.det(Sf / Tstar)) + len(Sf) * np.log(Tstar))

        #[logMarginalLikelihood[model], logMarginalLikelihoodR[model]] = calclogMarginalLikelihoods(N, K, M, T, T0, Tstar, S0, Sr, SrR, Sf0, Sf)


        T0Total = T0Total + T0
        T0Max = max(T0Max, T0)
        T0Min = min(T0Min, T0)
        T0_div_T0_plus_T += T0/(T0+T)
        T_div_T0_plus_T += T/(T0+T)
    #     del N, K, F, R, Z, M
    #
    # tictoc.toc()
    #
    T0IncreasedFraction = nTooSmallT0 / nLegitModels
    T0Avg = T0Total / nLegitModels
    T0_div_T0_plus_TAvg = T0_div_T0_plus_T/nLegitModels
    T_div_T0_plus_TAvg = T_div_T0_plus_T/nLegitModels

    return (logMarginalLikelihood, logMarginalLikelihoodR, T0IncreasedFraction, T0Max, T0Min, T0Avg, \
        T0_div_T0_plus_TAvg, T_div_T0_plus_TAvg, nLegitModels, nTooSmallT0)

# ========== End of function conditionalAssetPricingLogMarginalLikelihoodTauNumba ==========

def calculatePredictions(Z, T, AfTilda, phiTilda, phiTildaR, \
                         KMax, K, M , TEstimation, TEstimationStar, \
                         Sr, SrR, Sf, XtXInv, WtWInv, WRtWRInv, \
                         predictorsIndicesIncludedInModel, factorsIndicesIncludedInModel, otherFactors, \
                         unrestricted_model_probability, restricted_model_probability):

    iotaT = np.ones((T, 1), dtype=np.float64)

    Z = Z[:, predictorsIndicesIncludedInModel]
    if M == 0:
        Z = np.zeros((T, 0), dtype=np.float64)

    X = np.concatenate((iotaT, Z), axis=1)
    if K != 0:
        F = X @ AfTilda
    else:
        F = np.zeros((T, 0), dtype=np.float64)

    if K != 0 and M != 0:
        Omega = np.zeros((T, K * M), dtype=np.float64)
        for t in np.arange(0, T):
            Omega[t, :] = (np.kron(np.identity(K), Z[t, :].reshape(M, 1)) @ F[t, :]).flatten()
    else:
        Omega = np.zeros((T, 0), dtype=np.float64)

    W = np.concatenate((np.concatenate((X, F), axis=1), Omega), axis=1)
    WR = np.concatenate((F, Omega), axis=1)

    R = W @ phiTilda
    RR = WR @ phiTildaR

    N = R.shape[1]

    returns = np.zeros((T, KMax), dtype=np.float64)
    returns_square = np.zeros((T, KMax), dtype=np.float64)

    returns_unrestricted = np.zeros((T, KMax), dtype=np.float64)
    returns_restricted   = np.zeros((T, KMax), dtype=np.float64)
    returns_interactions = np.zeros((T, KMax, KMax), dtype=np.float64)
    # Total covariance matrix of the weighted unrestricted and restricted models.
    Covariance_matrix = np.zeros((T, KMax, KMax), dtype=np.float64)
    # Total covariance matrix of the weighted unrestricted and restricted models.
    covariance_matrix_no_ER = np.zeros((T, KMax, KMax), dtype=np.float64)
    # Total covariance matrix of the unrestricted model.
    Covariance_matrix_unrestricted = np.zeros((T, KMax, KMax), dtype=np.float64)
    # Total covariance matrix of the restricted models.
    Covariance_matrix_restricted = np.zeros((T, KMax, KMax), dtype=np.float64)
    # Total covariance of the factors. The specific model has K factors.
    Covariance_matrix_factors = np.zeros((K, K), dtype=np.float64)
    # Total covariance of the test assets for the unrestricted model. The specific model has N factors.
    Covariance_matrix_TA = np.zeros((N, N), dtype=np.float64)
    # Total covariance of the test assets for the restricted model. The specific model has N factors.
    Covariance_matrix_TAR = np.zeros((N, N), dtype=np.float64)

    # The covariance of the factors without estimation risk. The specific model has K factors.
    covariance_matrix_factors_no_ER = np.zeros((K, K), dtype=np.float64)
    # The covariance of the test assets for the unrestricted model without estimation risk. The specific model has N factors.
    covariance_matrix_TA_no_ER = np.zeros((N, N), dtype=np.float64)
    # The covariance of the test assets for the restricted model without estimation risk. The specific model has N factors.
    covariance_matrix_TAR_no_ER = np.zeros((N, N), dtype=np.float64)

    # Factors static risk premuium and time varying 
    Returns_terms_factors = np.zeros((T, K, 2), dtype=np.float64)
    for k in np.arange(0, K):
        # Static risk premium.
        Returns_terms_factors[:,k,0] = X[:,0] * AfTilda[0,k]
        # Time varying risk premium.
        #Returns_terms_factors[:,k,1] = X[:,1:].reshape(T,-1) @ AfTilda[1:,k]
        if M > 0:
            Returns_terms_factors[:,k,1] = Z @ AfTilda[1:,k]

#    if K !=0:
#       assert np.allclose(np.sum(Returns_terms_factors,axis=2),F)

    # TA = unrestrickted static mispricing, time varying mispricing, TAR = restricted (zero mispricing)
    # static beta * static premium, static beta * time varying premium 
    # time varying beta * static premium, time varying beta * time varying premium 
    Returns_terms_TA = np.zeros((T, N, 6), dtype=np.float64)
    Returns_terms_TAR = np.zeros((T, N, 6), dtype=np.float64)

    # Output
    # TA static mispricing, time varying mispricing, 
    # static beta * static premium, static beta * time varying premium 
    # time varying beta * static premium, time varying beta * time varying premium 
    # Static Risk premium, time varying risk premium. 
    Returns_terms_weighted = np.zeros((T, KMax, 8), dtype=np.float64)
    Returns_terms_square_weighted = np.zeros((T, KMax, 8), dtype=np.float64)
    Returns_terms_cumulative_probability = np.zeros((KMax, 8), dtype=np.float64)

    Sigma_FF = Sf / (TEstimationStar + N - K - M - 2)  # In the code N = N+K-k ( in the paper)
    Sigma_RR = Sr / (TEstimationStar -(K+1)*M - N - 2) 
    Sigma_RRR = SrR / (TEstimationStar - K*M - N - 1) 

    for t in np.arange(0, T):
        xt=np.concatenate((np.ones((1,1),dtype=np.float64),Z[t,:].reshape(-1,1)),axis=0)
        wt = W[t,:].reshape(-1,1)
        wtR = WR[t,:].reshape(-1,1)
        I_k_kron_zt = np.kron(np.identity(K), Z[t, :].reshape(M, 1))
        beta_t = np.transpose(phiTilda[1+M : 1+M+K, :]) + np.transpose(phiTilda[1+M+K :, :]) @ I_k_kron_zt  # Should be N X K
        # TA static mispricing
        Returns_terms_TA[t, 0:N, 0] = phiTilda[0,0:N]
        # time varying mispricing
        Returns_terms_TA[t, 0:N, 1] = X[t,1:].reshape(1,-1) @ phiTilda[1:M+1,0:N]
        # static beta and time varying premium 
        beta_static_t = np.transpose(phiTilda[1+M : 1+M+K, :])
        beta_varying_t = np.transpose(phiTilda[1+M+K :, :]) @ I_k_kron_zt
        # static beta * static premium
        Returns_terms_TA[t, 0:N, 2] = beta_static_t @ Returns_terms_factors[t,:,0]
        # static beta * time varying premium
        Returns_terms_TA[t, 0:N, 3] = beta_static_t @ Returns_terms_factors[t,:,1]
        # time varying beta * static premium
        Returns_terms_TA[t, 0:N, 4] = beta_varying_t @ Returns_terms_factors[t,:,0]
        # time varying beta * time varying premium
        Returns_terms_TA[t, 0:N, 5] = beta_varying_t @ Returns_terms_factors[t,:,1]

        # assert np.allclose(R[t,:],np.sum(Returns_terms_TA[t,:,:],axis=1))

        betaR_t = np.transpose(phiTildaR[:K,:]) + np.transpose(phiTildaR[K:,:]) @ I_k_kron_zt           # Should be N X K
        # static beta and time varying premium 
        betaR_static_t = np.transpose(phiTildaR[:K,:])
        betaR_varying_t = np.transpose(phiTildaR[K:,:]) @ I_k_kron_zt
        # static beta * static premium
        Returns_terms_TAR[t,0:N,2] = betaR_static_t @ Returns_terms_factors[t,:,0]
        # static beta * time varying premium
        Returns_terms_TAR[t,0:N,3] = betaR_static_t @ Returns_terms_factors[t,:,1]
        # time varying beta * static premium
        Returns_terms_TAR[t,0:N,4] = betaR_varying_t @ Returns_terms_factors[t,:,0]
        # time varying beta * time varying premium
        Returns_terms_TAR[t,0:N,5] = betaR_varying_t @ Returns_terms_factors[t,:,1]

        # assert np.allclose(RR[t,:],np.sum(Returns_terms_TAR[t,:,:],axis=1))

        Covariance_matrix_factors[:,:] = Sigma_FF*(1+TEstimation/TEstimationStar*np.transpose(xt)@ XtXInv @ xt)
        Covariance_matrix_TA[:,:] = Sigma_RR*(1+TEstimation/TEstimationStar*np.transpose(wt)@ WtWInv @ wt) + \
                                    beta_t @ Covariance_matrix_factors[:,:] @ np.transpose(beta_t)
        Covariance_matrix_TAR[:,:] = Sigma_RRR*(1+TEstimation/TEstimationStar*np.transpose(wtR)@ WRtWRInv @ wtR) + \
                                    betaR_t @ Covariance_matrix_factors[:,:] @ np.transpose(betaR_t)

        covariance_matrix_factors_no_ER[:,:] = Sigma_FF
        covariance_matrix_TA_no_ER[:,:] = Sigma_RR + \
                                    beta_t @ covariance_matrix_factors_no_ER[:,:] @ np.transpose(beta_t)
        covariance_matrix_TAR_no_ER[:,:] = Sigma_RRR + \
                                    betaR_t @ covariance_matrix_factors_no_ER[:,:] @ np.transpose(betaR_t)

        if N == KMax - K:
            for i in np.arange(0,len(factorsIndicesIncludedInModel)):
                for j in np.arange(0,len(factorsIndicesIncludedInModel)):
                    ii = factorsIndicesIncludedInModel[i]
                    jj = factorsIndicesIncludedInModel[j]

                    Covariance_matrix[t, ii, jj] = Covariance_matrix_factors[i, j] * \
                        (unrestricted_model_probability + restricted_model_probability)
                    
                    # Covariance matrix without estimation risk.
                    covariance_matrix_no_ER[t, ii, jj] = covariance_matrix_factors_no_ER[i, j] * \
                        (unrestricted_model_probability + restricted_model_probability)
                    
                    # Covariance matrix of the unrestricted model
                    Covariance_matrix_unrestricted[t, ii, jj] = Covariance_matrix_factors[i, j]
                    # Covariance matrix of the restricted model
                    Covariance_matrix_restricted[t, ii, jj] = Covariance_matrix_factors[i, j]
                    
            for i in np.arange(0, len(otherFactors)):
                for j in np.arange(0, len(otherFactors)):
                    ii = otherFactors[i]
                    jj = otherFactors[j]

                    Covariance_matrix[t, ii, jj] = unrestricted_model_probability * Covariance_matrix_TA[i, j] + \
                                                restricted_model_probability * Covariance_matrix_TAR[i, j]

                    # Covariance matrix without estimation risk.
                    covariance_matrix_no_ER[t, ii, jj] = unrestricted_model_probability * covariance_matrix_TA_no_ER[i, j] + \
                                                restricted_model_probability * covariance_matrix_TAR_no_ER[i, j]

                    Covariance_matrix_unrestricted[t, ii, jj] = Covariance_matrix_TA[i, j]
                    Covariance_matrix_restricted[t, ii, jj] = Covariance_matrix_TAR[i, j]


    for i in np.arange(0, len(factorsIndicesIncludedInModel)):
        ii = factorsIndicesIncludedInModel[i]
        returns[:, ii] += (unrestricted_model_probability + \
                               restricted_model_probability) * F[:, i]

        returns_square[:, ii] += (unrestricted_model_probability + \
                                      restricted_model_probability) * F[:, i] ** 2

        returns_unrestricted[:, ii] = F[:,i]
        returns_restricted[:, ii] = F[:,i]
        
        # Static Risk premium, time varying risk premium. 
        for term in np.arange(0, 2):
            Returns_terms_weighted[:, ii, 6 + term] += (unrestricted_model_probability + \
                                               restricted_model_probability) * Returns_terms_factors[:, i, term]
            
            Returns_terms_square_weighted[:, ii, 6 + term] += (unrestricted_model_probability + \
                                                restricted_model_probability) * Returns_terms_factors[:, i, term]**2
          
            Returns_terms_cumulative_probability[ii, 6 + term] += unrestricted_model_probability + restricted_model_probability

    for i in np.arange(0, len(otherFactors)):
        ii = otherFactors[i]
        returns[:, ii] += unrestricted_model_probability * R[:, i] + \
                              restricted_model_probability * RR[:, i]

        returns_square[:, ii] += unrestricted_model_probability * R[:, i] ** 2 + \
                                     restricted_model_probability * RR[:, i] ** 2

        returns_unrestricted[:, ii] = R[:,i]
        returns_restricted[:, ii] = RR[:,i]

        # TA static mispricing
        for term in np.arange(0,6):
            # TA static mispricing and time varying mispricing
            if term < 2:
                # TA static mispricing and time varying mispricing
                Returns_terms_weighted[:, ii, term] += unrestricted_model_probability * Returns_terms_TA[:, i, term]

                Returns_terms_square_weighted[:, ii, term] += unrestricted_model_probability * Returns_terms_TA[:, i, term]**2

                Returns_terms_cumulative_probability[ii, term] += unrestricted_model_probability
        
            else:
                # static beta * static premium, static beta * time varying premium, 
                # time varying beta * static premium, time varying beta * time varying premium
                Returns_terms_weighted[:, ii, term] += unrestricted_model_probability * Returns_terms_TA[:, i, term] + \
                                                restricted_model_probability * Returns_terms_TAR[:, i, term]

                Returns_terms_square_weighted[:, ii, term] += unrestricted_model_probability * Returns_terms_TA[:, i, term]**2 + \
                                                 restricted_model_probability * Returns_terms_TAR[:, i, term]**2

                Returns_terms_cumulative_probability[ii, term] += unrestricted_model_probability + restricted_model_probability

    for t in np.arange(0, T):
        returns_interactions[t,:,:] += unrestricted_model_probability * \
            (returns_unrestricted[t,:].reshape(-1,1) @ returns_unrestricted[t,:].reshape(1,-1))
        returns_interactions[t,:,:] += restricted_model_probability * \
            (returns_restricted[t,:].reshape(-1,1) @ returns_restricted[t,:].reshape(1,-1))

    return (returns, returns_square, returns_interactions, Covariance_matrix, covariance_matrix_no_ER, \
            Covariance_matrix_unrestricted, returns_unrestricted, Covariance_matrix_restricted, returns_restricted, \
            Returns_terms_weighted, Returns_terms_square_weighted, Returns_terms_cumulative_probability)

# ========== End of function calculatePredictions ==========

def conditionalAssetPricingOOSPredictionsTauNumba(ROrig, FOrig, ZOrig, OmegaOrig, Tau, SR2Mkt, ZTest, \
                                                  models_probabilities, nModelsInPrediction):

    key_Avoid_duplicate_predictors = 1

    #print("calculating both unrestricted and restricted models")
    #print("key_Avoid_duplicate_factors= %d, key_Avoid_duplicate_predictors= %d, key_use_combination_matrix= %d " \
    #      % (key_Avoid_duplicate_factors, key_Avoid_duplicate_predictors, key_use_combination_matrix))
    #print("Tau= %f " % (Tau))

    # Moving on to calculating the marginal likelihood.
    # Trying to work with nupy arrays instead of pandas dataframe.
    KMax = FOrig.shape[1]
    MMax = ZOrig.shape[1]
    KMaxPlusMMax = KMax + MMax
    T = FOrig.shape[0]

    NMinTestAssets = ROrig.shape[1]
    TOOS = len(ZTest)
    # Out of sample returns and square returns.
    returns_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    returns_square_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    returns_interactions_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    covariance_matrix_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    # covariance matrix without estimation risk
    covariance_matrix_no_ER_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    # single model 
    cov_matrix_single_unrestricted_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    returns_single_unrestricted_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    cov_matrix_single_restricted_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    returns_single_restricted_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)

    # in sample returns and square returns.
    returns_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    returns_square_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    returns_interactions_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    covariance_matrix_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    # covariance matrix without estimation risk
    covariance_matrix_no_ER_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)

    # single model 
    cov_matrix_single_unrestricted_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    returns_single_unrestricted_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    cov_matrix_single_restricted_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    returns_single_restricted_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)

    # Market squared Sharpe ratio and maximum difference between the upper bound Sharpe ratio and the market.
    # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
    Tau2m1SR2Mkt = (pow(Tau, 2) - 1) * SR2Mkt

    nModelsMax = pow(2, KMax + MMax)

    # models_probabilities - first is the unrestricted AP models' probabilities and than the restricted AP models' probabilities.
    assert len(models_probabilities) == 2 * nModelsMax

    nTooSmallT0 = 0
    nLegitModels = 0
    T0Total = 0
    T0Max = 0
    T0Min = np.inf

    iotaT = np.ones((T, 1), dtype=np.float64)
    iotaTOOS = np.ones((TOOS, 1), dtype=np.float64)

    nprintFraction = pow(10, 1 + (KMaxPlusMMax > 20))

    cumulative_probability = 0.
    mStart = 0
    mEnd = nModelsMax
    # Debug range.
    #mStart = int(nModelsMax/2 )
    #mEnd = mStart + 100
    #models_probabilities[:] = 1.

    models_range = np.arange(mStart, mEnd)

    # Use only sub sample with the nModelsInPrediction highest probabilities.
    # Sort from smallest to largest.
    if nModelsInPrediction > 0:
        I = np.argsort(-models_probabilities) % nModelsMax
        # Notice that np.unique is returning a sorted array of unique values.
        ModelsIndices = np.unique(I[0:nModelsInPrediction])
        models_range = ModelsIndices

    # Negative or zero is for single top model prediction. 
    # Where -nModelsInPrediction is the top probable model to use.
    elif nModelsInPrediction < 0:
        I = np.argsort(-models_probabilities) # Sorting the models by their probabilities from higher to lower.
        # The model should be between 0 to nModelsMax
        model_index = I[-nModelsInPrediction - 1]      # Taking the top -nModelsInPrediction model.
        # The model should be between 0 to nModelsMax
        mStart = model_index % nModelsMax
        mEnd = mStart + 1
        models_range = np.arange(mStart, mEnd)
        # Zeroing all probabilities execpt the desired model.
        models_probabilities[:] = 0.
        # Setting the top model probability to one.
        models_probabilities[model_index] = 1.

    model_number = -1
    total_number_of_models = len(models_range)
    nPrintModels =  max(np.floor(total_number_of_models / nprintFraction),100)
    # print('First model is %i. Total numbers of models is %i. nprintFraction= %d' % (mStart, nModelsMax, nprintFraction))
    for model in models_range:
        model_number += 1
        #print("model " + str(model))
        if model_number % nPrintModels == 0:
    #         totalTime += tictoc.toc(False)
            print("Model  " + str(model_number) + " out of " + str(total_number_of_models))
    #            print('Done %3d %% of total work at %12.2f sec. model= %11i, nModelsMax= %11i'
    #               % (100 * model / nModelsMax, totalTime, model, nModelsMax))
    #         # Dump
    #         with open('local_dump.pkl', 'bw') as file:
    #             pickle.dump([model, nModelsMax, KMax, MMax, factorsNames, predictorsNames, \
    #                 significantPredictors, Tau, nTooSmallT0, nLegitModels, T0Max, T0Total, \
    #                          logMarginalLikelihood, logMarginalLikelihoodR], file)

        combinationsRowFromMatrix = retreiveRowFromAllCombinationsMatrixNumba(model, KMaxPlusMMax)
        factorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[0: KMax] == 1).flatten()
        predictorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[KMax:] == 1).flatten()

        otherFactors = findOtherFactors(KMax, factorsIndicesIncludedInModel)
        #print(otherFactors)

        # MKT is not in model assign a 0 probability. Total number of combinations is 2^(KMax-1)*2^MMax.
        # However in case of all factors in the test assets continue as linear predictive regression.
        # Total number of combinations is 2^MMax.
        if not (1 - 1 in factorsIndicesIncludedInModel) and len(factorsIndicesIncludedInModel) != 0:
    #         # logMarginalLikelihood[model] = -np.inf
            continue
    #
        if key_Avoid_duplicate_predictors:
            # Add restriction when there is a linear dependency between the predictors.
            # Not including the following three predictors together: dp, ep, de.
            # predictors 1, 3, 4 out of the 8 combinations of including the
            # predictors only 5 are independent:
            # 1 - none of them, 3 - only one predictor and 1 - one pair out of the three
            # possible pairs.
            # and the tbl, lty, tms. predictors 8, 9, 11

            if (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel) or \
                    (9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel):

                continue

        # Avoid models with duplicate factors.
        if key_Avoid_duplicate_factors:
            if (2 - 1 in factorsIndicesIncludedInModel and 8 - 1 in factorsIndicesIncludedInModel) or \
                    (4 - 1 in factorsIndicesIncludedInModel and 10 - 1 in factorsIndicesIncludedInModel) or \
                    (5 - 1 in factorsIndicesIncludedInModel and 9 - 1 in factorsIndicesIncludedInModel):

                if keyPrint:
                    print('duplicate factors!!! factorsIndicesIncludedInModel= ')
                    print(factorsIndicesIncludedInModel)

                continue

        # Each model has different N K R and F. Remove the temporary values at the end of each cycle.
        # del N, K, F, R

        if len(ROrig) == T:
            R = np.concatenate((ROrig[:, :], FOrig[:, otherFactors]), axis=1)
        else:
            R = FOrig[:, otherFactors]

        N = R.shape[1]

        # No test assets. 2^MMax combinations.
        if N == 0:
            continue

        nLegitModels += 1

        F = FOrig[:, factorsIndicesIncludedInModel]
        K = F.shape[1]
        Z = ZOrig[:, predictorsIndicesIncludedInModel]
        M = Z.shape[1]

        if M == 0:
            Z = np.empty((T, 0), dtype=np.float64)

        assert R.shape[0] == T and F.shape[0] == T and Z.shape[0] == T

        OmegaIndecies = np.repeat(factorsIndicesIncludedInModel * MMax, M) + \
                        tileNumba(predictorsIndicesIncludedInModel, K)
        X = np.concatenate((iotaT, Z), axis=1)

    #     if keyDebug:
    #         Omega = np.zeros((T, K * M), dtype=float)
    #         for t in np.arange(0, T):
    #             Omega[t, :] = np.kron(np.identity(K), Z[t, :].reshape(-1, 1)) @ F[t, :]
    #
    #         assert np.allclose(Omega, OmegaOrig[:, OmegaIndecies])
    #
        Omega = OmegaOrig[:, OmegaIndecies]
    #
        W = np.concatenate((np.concatenate((X, F), axis=1), Omega), axis=1)

        RMean = np.ascontiguousarray(MeanAlongAxisNumba(R, 0))
        FMean = np.ascontiguousarray(MeanAlongAxisNumba(F, 0))
        ZMean = np.ascontiguousarray(MeanAlongAxisNumba(Z, 0))
        #print(FMean)

        Vf = np.cov(F, rowvar=False, bias=True).reshape(K, K)

        # Hypothetical sample quantities.

        beta0 = LA.pinv(np.transpose(F) @ F) @ np.transpose(F) @ R

    #     if keyDebug:
    #         beta01 = LA.lstsq(F, R, rcond=None)[0]  # Should be the same as beta0, just checking.
    #         assert np.allclose(beta0, beta01)
    #
    #         Af0 = LA.pinv(np.transpose(iotaT) @ iotaT) @ np.transpose(iotaT) @ F  # Should equal FMean, just checking.
    #         assert np.allclose(Af0, np.transpose(FMean))

        Af0 = np.concatenate((FMean, np.zeros((K, M), dtype=np.float64)), axis=1)
        XtX = np.transpose(X) @ X
        XtXInv = LA.pinv(XtX)
        WtW = np.transpose(W) @ W

        #try:
        WtWInv = LA.pinv(WtW)
    #     except np.linalg.LinAlgError as e:
    #         print('Error %s in model number %i' % (e, model))
    #         print(factorsIndicesIncludedInModel)
    #         print(predictorsIndicesIncludedInModel)
    #         string = 'Trying normal inverse ... '
    #         try:
    #             WtWInv = LA.inv(WtW)
    #             print(string + 'LA.inv worked')
    #         except:
    #             string += 'Failed trying scipy pinvh'
    #             try:
    #                 WtWInv = SPLA.pinvh(WtW)
    #                 print(string + 'SPLA.pinvh worked')
    #             except:
    #                 print(string + 'Failed ')
    #
        Qw = np.identity(T) - W @ WtWInv @ np.transpose(W)
        # SigmaRR = (np.transpose(R) @ Qw @ R) / (T - M - K - K*M - 1) / pow(100,2) # changing from percentage to numbers.

        # Since the distribution is known the maximum likelihood estimator is without the bias correction.
        SigmaRR = (np.transpose(R) @ Qw @ R) / (T) / pow(100, 2)  # changing from percentage to numbers.

        SR2 = (np.transpose(FMean) @ LA.pinv(Vf) @ FMean).flatten()  # factors Sharpe ratio square.
        SR2 = np.sum(SR2)

        T0 = int(np.round((N * (1 + SR2 + M * (1 + SR2))) / Tau2m1SR2Mkt))
        # print(N,K,M,T0,SR2, np.transpose(FMean) @ LA.pinv(Vf) @ FMean)

        # MultiGamma(p(a)) is defined for 2*a > p - 1
        T0LowerBound = max(N + (K + 1) * M, K + M - N)
        if T0 <= T0LowerBound:
            nTooSmallT0 += 1
    #         if keyPrint:
    #             print('T0 ( %i ) too small so increasing it to minmum acceptable value ( %i )' % (T0, T0LowerBound + 1))
            T0 = T0LowerBound + 1

        FtF = np.transpose(F) @ F
        RtR = np.transpose(R) @ R
        XtF = np.transpose(X) @ F
        WtR = np.transpose(W) @ R

        S0 = T0 / T * (RtR - np.transpose(beta0) @ (FtF) @ beta0)
        Sf0 = T0 * Vf

        Tstar = T0 + T
        FMeanFMeanZMeant = np.concatenate((FMean, FMean @ np.transpose(ZMean)), axis=1)

        AfTilda = T / Tstar * XtXInv @ (XtF + T0 * np.transpose(FMeanFMeanZMeant))

        Sf = Tstar * (Vf + FMean @ np.transpose(FMean)) \
             - T / Tstar * (T0 * FMeanFMeanZMeant + np.transpose(XtF)) @ \
             XtXInv @ (T0 * np.transpose(FMeanFMeanZMeant) + XtF)

        # Calculating the unrestricted log marginal likelihood.
        phi0 = np.transpose(np.concatenate((
            np.concatenate((np.zeros((N, M + 1), dtype=np.float64), np.transpose(beta0)), axis=1), \
            np.zeros((N, K * M), dtype=np.float64)), axis=1))

        phiTilda = T / Tstar * WtWInv @ (WtR + T0 / T * WtW @ phi0)

        RminusWphiTilda = R - W @ phiTilda

        Sr = S0 + np.transpose(RminusWphiTilda) @ (RminusWphiTilda) + \
             T0 / T * np.transpose(phiTilda - phi0) @ WtW @ (phiTilda - phi0)

        # Calculating the restricted.
        phi0R = np.transpose(np.concatenate((np.transpose(beta0), np.zeros((N, K * M), dtype=np.float64)), axis=1))
        WR = np.concatenate((F, Omega), axis=1)
        WRtWR = np.transpose(WR) @ WR
    #     try:
        WRtWRInv = LA.pinv(WRtWR)
    #     except np.linalg.LinAlgError as e:
    #         print('Error %s in model number %i' % (e, model))
    #         # print(WRtWR[0,:])
    #         print(factorsIndicesIncludedInModel)
    #         print(predictorsIndicesIncludedInModel)
    #         # logMarginalLikelihoodR[model] = -np.inf
    #         string = 'Trying normal inverse ... '
    #         try:
    #             WRtWRInv = LA.inv(WRtWR)
    #             print(string + 'LA.inv worked')
    #         except:
    #             string += 'Failed trying scipy pinvh'
    #             try:
    #                 WRtWRInv = SPLA.pinvh(WRtWR)
    #                 print(string + 'SPLA.pinvh worked')
    #             except:
    #                 print(string + 'Failed assaining NINF')
    #                 continue
    #
        phiTildaR = T / Tstar * WRtWRInv @ (np.transpose(WR) @ R + T0 / T * WRtWR @ phi0R)

        SrR = S0 + np.transpose(R - WR @ phiTildaR) @ (R - WR @ phiTildaR) + \
              T0 / T * np.transpose(phiTildaR - phi0R) @ WRtWR @ (phiTildaR - phi0R)

        # In sample returns and returns square predictions.
        (returns, returns_square, returns_interactions, covariance_matrix, covariance_matrix_no_ER, \
            cov_matrix_single_unrestricted_IN, return_single_unrestricted_IN, \
                cov_matrix_single_restricted_IN, return_single_restricted_IN,\
                Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, Returns_terms_cumulative_probability_IN) = \
                                calculatePredictions(ZOrig, T, AfTilda, phiTilda, phiTildaR, \
                                        KMax, K, M, T, Tstar, \
                                        Sr, SrR, Sf, XtXInv, WtWInv, WRtWRInv, \
                                        predictorsIndicesIncludedInModel, factorsIndicesIncludedInModel, otherFactors,\
                                        models_probabilities[model], models_probabilities[model + nModelsMax])

        returns_IN  += returns
        returns_square_IN += returns_square
        returns_interactions_IN += returns_interactions
        covariance_matrix_IN += covariance_matrix
        covariance_matrix_no_ER_IN += covariance_matrix_no_ER
        
        # Out of sample returns and returns square predictions.
        (returns, returns_square, returns_interactions, covariance_matrix, covariance_matrix_no_ER, \
            cov_matrix_single_unrestricted_OOS, return_single_unrestricted_OOS, \
                cov_matrix_single_restricted_OOS, return_single_restricted_OOS, \
                Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, Returns_terms_cumulative_probability_OOS) = \
                                calculatePredictions(ZTest, TOOS, AfTilda, phiTilda, phiTildaR, \
                                        KMax, K, M, T, Tstar, \
                                        Sr, SrR, Sf, XtXInv, WtWInv, WRtWRInv, \
                                        predictorsIndicesIncludedInModel, factorsIndicesIncludedInModel, otherFactors,\
                                        models_probabilities[model], models_probabilities[model + nModelsMax])

        returns_OOS += returns
        returns_square_OOS += returns_square
        returns_interactions_OOS += returns_interactions
        covariance_matrix_OOS += covariance_matrix
        covariance_matrix_no_ER_OOS += covariance_matrix_no_ER
        # del returns, returns_square, returns_interactions, covariance_matrix

        T0Total = T0Total + T0
        T0Max = max(T0Max, T0)
        T0Min = min(T0Min, T0)

        cumulative_probability += models_probabilities[model] + models_probabilities[model + nModelsMax]
    #     del N, K, F, R, Z, M

    nLegitModels = max(nLegitModels,1)

    T0IncreasedFraction = nTooSmallT0 / nLegitModels
    T0Avg = T0Total / nLegitModels

    return (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
            returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
            T0IncreasedFraction, T0Max, T0Min, T0Avg, nLegitModels, nTooSmallT0, cumulative_probability)

# ========== End of function conditionalAssetPricingOOSPredictionsTauNumba ==========

def conditionalAssetCalculateSpreadNumba(ROrig, FOrig, ZOrig, OmegaOrig, Tau, SR2Mkt, 
                                            RTest, FTest, ZTest, \
                                            models_probabilities, nModelsInPrediction):

    key_Avoid_duplicate_predictors = 1

    #print("calculating both unrestricted and restricted models")
    #print("key_Avoid_duplicate_factors= %d, key_Avoid_duplicate_predictors= %d, key_use_combination_matrix= %d " \
    #      % (key_Avoid_duplicate_factors, key_Avoid_duplicate_predictors, key_use_combination_matrix))
    #print("Tau= %f " % (Tau))

    # Moving on to calculating the marginal likelihood.
    # Trying to work with nupy arrays instead of pandas dataframe.
    KMax = FOrig.shape[1]
    MMax = ZOrig.shape[1]
    KMaxPlusMMax = KMax + MMax
    T = FOrig.shape[0]

    NMinTestAssets = ROrig.shape[1]
    TOOS = len(ZTest)
    # Out of sample returns and square returns.
    returns_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    returns_square_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    returns_interactions_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    covariance_matrix_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    # covariance matrix without estimation risk
    covariance_matrix_no_ER_OOS = np.zeros((TOOS, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)

    weights_sum_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    weights_square_sum_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    weights_sum_equal_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    weights_square_sum_equal_OOS = np.zeros((TOOS, NMinTestAssets + KMax), dtype=np.float64)
    returns_sum_OOS = np.zeros((TOOS,), dtype=np.float64)
    returns_square_sum_OOS = np.zeros((TOOS, ), dtype=np.float64)
    returns_sum_equal_OOS = np.zeros((TOOS, ), dtype=np.float64)
    returns_square_sum_equal_OOS = np.zeros((TOOS, ), dtype=np.float64)

    Returns_terms_weighted_OOS = np.zeros((TOOS, NMinTestAssets + KMax, 8), dtype=np.float64)
    Returns_terms_square_weighted_OOS = np.zeros((TOOS, NMinTestAssets + KMax, 8), dtype=np.float64)
    Returns_terms_cumulative_probability_OOS = np.zeros((NMinTestAssets + KMax, 8), dtype=np.float64)

    # in sample returns and square returns.
    returns_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    returns_square_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    returns_interactions_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    covariance_matrix_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)
    # covariance matrix without estimation risk
    covariance_matrix_no_ER_IN = np.zeros((T, NMinTestAssets + KMax, NMinTestAssets + KMax), dtype=np.float64)

    weights_sum_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    weights_square_sum_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    weights_sum_equal_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    weights_square_sum_equal_IN = np.zeros((T, NMinTestAssets + KMax), dtype=np.float64)
    returns_sum_IN = np.zeros((T, ), dtype=np.float64)
    returns_square_sum_IN = np.zeros((T, ), dtype=np.float64)
    returns_sum_equal_IN = np.zeros((T, ), dtype=np.float64)
    returns_square_sum_equal_IN = np.zeros((T, ), dtype=np.float64)

    Returns_terms_weighted_IN = np.zeros((T, NMinTestAssets + KMax, 8), dtype=np.float64)
    Returns_terms_square_weighted_IN = np.zeros((T, NMinTestAssets + KMax, 8), dtype=np.float64)
    Returns_terms_cumulative_probability_IN = np.zeros((NMinTestAssets + KMax, 8), dtype=np.float64)

    # Market squared Sharpe ratio and maximum difference between the upper bound Sharpe ratio and the market.
    # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
    Tau2m1SR2Mkt = (pow(Tau, 2) - 1) * SR2Mkt

    nModelsMax = pow(2, KMax + MMax)

    # models_probabilities - first is the unrestricted AP models' probabilities and than the restricted AP models' probabilities.
    assert len(models_probabilities) == 2 * nModelsMax

    nTooSmallT0 = 0
    nLegitModels = 0
    T0Total = 0
    T0Max = 0
    T0Min = np.inf

    iotaT = np.ones((T, 1), dtype=np.float64)

    if len(ROrig) == T:
        TAReturns_IN = np.concatenate((ROrig,FOrig), axis=1)
    else:
        TAReturns_IN = FOrig
    
    if len(ROrig) == TOOS:
        TAReturns_OOS = np.concatenate((RTest,FTest), axis=1)
    else:
        TAReturns_OOS = FTest

    nprintFraction = pow(10, 1 + (KMaxPlusMMax > 20))

    cumulative_probability = 0.
    mStart = 0
    mEnd = nModelsMax
    # Debug range.
    #mStart = int(nModelsMax/2 )
    #mEnd = mStart + 100
    #models_probabilities[:] = 1.

    models_range = np.arange(mStart, mEnd)

    # Use only sub sample with the nModelsInPrediction highest probabilities.
    # Sort from smallest to largest.
    if nModelsInPrediction > 0:
        I = np.argsort(-models_probabilities) % nModelsMax
        # Notice that np.unique is returning a sorted array of unique values.
        ModelsIndices = np.unique(I[0:nModelsInPrediction])
        models_range = ModelsIndices

    # Negative or zero is for single top model prediction. 
    # Where -nModelsInPrediction is the top probable model to use.
    elif nModelsInPrediction < 0:
        I = np.argsort(-models_probabilities) # Sorting the models by their probabilities from higher to lower.
        # The model should be between 0 to nModelsMax
        model_index = I[-nModelsInPrediction - 1]      # Taking the top -nModelsInPrediction model.
        # The model should be between 0 to nModelsMax
        mStart = model_index % nModelsMax
        mEnd = mStart + 1
        models_range = np.arange(mStart, mEnd)
        # Zeroing all probabilities execpt the desired model.
        models_probabilities[:] = 0.
        # Setting the top model probability to one.
        models_probabilities[model_index] = 1.

    model_number = -1
    total_number_of_models = len(models_range)
    nPrintModels =  max(np.floor(total_number_of_models / nprintFraction),100)
    # print('First model is %i. Total numbers of models is %i. nprintFraction= %d' % (mStart, nModelsMax, nprintFraction))
    for model in models_range:
        model_number += 1
        #print("model " + str(model))
        if model_number % nPrintModels == 0:
    #         totalTime += tictoc.toc(False)
            print("Model  " + str(model_number) + " out of " + str(total_number_of_models))
    #            print('Done %3d %% of total work at %12.2f sec. model= %11i, nModelsMax= %11i'
    #               % (100 * model / nModelsMax, totalTime, model, nModelsMax))
    #         # Dump
    #         with open('local_dump.pkl', 'bw') as file:
    #             pickle.dump([model, nModelsMax, KMax, MMax, factorsNames, predictorsNames, \
    #                 significantPredictors, Tau, nTooSmallT0, nLegitModels, T0Max, T0Total, \
    #                          logMarginalLikelihood, logMarginalLikelihoodR], file)

        combinationsRowFromMatrix = retreiveRowFromAllCombinationsMatrixNumba(model, KMaxPlusMMax)
        factorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[0: KMax] == 1).flatten()
        predictorsIndicesIncludedInModel = np.argwhere(combinationsRowFromMatrix[KMax:] == 1).flatten()

        otherFactors = findOtherFactors(KMax, factorsIndicesIncludedInModel)
        #print(otherFactors)

        # MKT is not in model assign a 0 probability. Total number of combinations is 2^(KMax-1)*2^MMax.
        # However in case of all factors in the test assets continue as linear predictive regression.
        # Total number of combinations is 2^MMax.
        if not (1 - 1 in factorsIndicesIncludedInModel) and len(factorsIndicesIncludedInModel) != 0:
    #         # logMarginalLikelihood[model] = -np.inf
            continue
    #
        if key_Avoid_duplicate_predictors:
            # Add restriction when there is a linear dependency between the predictors.
            # Not including the following three predictors together: dp, ep, de.
            # predictors 1, 3, 4 out of the 8 combinations of including the
            # predictors only 5 are independent:
            # 1 - none of them, 3 - only one predictor and 1 - one pair out of the three
            # possible pairs.
            # and the tbl, lty, tms. predictors 8, 9, 11

            if (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 3 - 1 in predictorsIndicesIncludedInModel) or \
                    (1 - 1 in predictorsIndicesIncludedInModel and 4 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel) or \
                    (8 - 1 in predictorsIndicesIncludedInModel and 9 - 1 in predictorsIndicesIncludedInModel) or \
                    (9 - 1 in predictorsIndicesIncludedInModel and 11 - 1 in predictorsIndicesIncludedInModel):

                continue

        # Avoid models with duplicate factors.
        if key_Avoid_duplicate_factors:
            if (2 - 1 in factorsIndicesIncludedInModel and 8 - 1 in factorsIndicesIncludedInModel) or \
                    (4 - 1 in factorsIndicesIncludedInModel and 10 - 1 in factorsIndicesIncludedInModel) or \
                    (5 - 1 in factorsIndicesIncludedInModel and 9 - 1 in factorsIndicesIncludedInModel):

                if keyPrint:
                    print('duplicate factors!!! factorsIndicesIncludedInModel= ')
                    print(factorsIndicesIncludedInModel)

                continue

        # Each model has different N K R and F. Remove the temporary values at the end of each cycle.
        # del N, K, F, R

        if len(ROrig) == T:
            R = np.concatenate((ROrig[:, :], FOrig[:, otherFactors]), axis=1)
        else:
            R = FOrig[:, otherFactors]

        N = R.shape[1]

        # No test assets. 2^MMax combinations.
        if N == 0:
            continue

        nLegitModels += 1

        F = FOrig[:, factorsIndicesIncludedInModel]
        K = F.shape[1]
        Z = ZOrig[:, predictorsIndicesIncludedInModel]
        M = Z.shape[1]

        if M == 0:
            Z = np.empty((T, 0), dtype=np.float64)

        assert R.shape[0] == T and F.shape[0] == T and Z.shape[0] == T

        OmegaIndecies = np.repeat(factorsIndicesIncludedInModel * MMax, M) + \
                        tileNumba(predictorsIndicesIncludedInModel, K)
        X = np.concatenate((iotaT, Z), axis=1)

    #     if keyDebug:
    #         Omega = np.zeros((T, K * M), dtype=float)
    #         for t in np.arange(0, T):
    #             Omega[t, :] = np.kron(np.identity(K), Z[t, :].reshape(-1, 1)) @ F[t, :]
    #
    #         assert np.allclose(Omega, OmegaOrig[:, OmegaIndecies])
    #
        Omega = OmegaOrig[:, OmegaIndecies]
    #
        W = np.concatenate((np.concatenate((X, F), axis=1), Omega), axis=1)

        RMean = np.ascontiguousarray(MeanAlongAxisNumba(R, 0))
        FMean = np.ascontiguousarray(MeanAlongAxisNumba(F, 0))
        ZMean = np.ascontiguousarray(MeanAlongAxisNumba(Z, 0))
        #print(FMean)

        Vf = np.cov(F, rowvar=False, bias=True).reshape(K, K)

        # Hypothetical sample quantities.

        beta0 = LA.pinv(np.transpose(F) @ F) @ np.transpose(F) @ R

        Af0 = np.concatenate((FMean, np.zeros((K, M), dtype=np.float64)), axis=1)
        XtX = np.transpose(X) @ X
        XtXInv = LA.pinv(XtX)
        WtW = np.transpose(W) @ W

        WtWInv = LA.pinv(WtW)

        Qw = np.identity(T) - W @ WtWInv @ np.transpose(W)
        # SigmaRR = (np.transpose(R) @ Qw @ R) / (T - M - K - K*M - 1) / pow(100,2) # changing from percentage to numbers.

        # Since the distribution is known the maximum likelihood estimator is without the bias correction.
        SigmaRR = (np.transpose(R) @ Qw @ R) / (T) / pow(100, 2)  # changing from percentage to numbers.

        SR2 = (np.transpose(FMean) @ LA.pinv(Vf) @ FMean).flatten()  # factors Sharpe ratio square.
        SR2 = np.sum(SR2)

        T0 = int(np.round((N * (1 + SR2 + M * (1 + SR2))) / Tau2m1SR2Mkt))
        # print(N,K,M,T0,SR2, np.transpose(FMean) @ LA.pinv(Vf) @ FMean)

        # MultiGamma(p(a)) is defined for 2*a > p - 1
        T0LowerBound = max(N + (K + 1) * M, K + M - N)
        if T0 <= T0LowerBound:
            nTooSmallT0 += 1
    #         if keyPrint:
    #             print('T0 ( %i ) too small so increasing it to minmum acceptable value ( %i )' % (T0, T0LowerBound + 1))
            T0 = T0LowerBound + 1

        FtF = np.transpose(F) @ F
        RtR = np.transpose(R) @ R
        XtF = np.transpose(X) @ F
        WtR = np.transpose(W) @ R

        S0 = T0 / T * (RtR - np.transpose(beta0) @ (FtF) @ beta0)
        Sf0 = T0 * Vf

        Tstar = T0 + T
        FMeanFMeanZMeant = np.concatenate((FMean, FMean @ np.transpose(ZMean)), axis=1)

        AfTilda = T / Tstar * XtXInv @ (XtF + T0 * np.transpose(FMeanFMeanZMeant))

        Sf = Tstar * (Vf + FMean @ np.transpose(FMean)) \
             - T / Tstar * (T0 * FMeanFMeanZMeant + np.transpose(XtF)) @ \
             XtXInv @ (T0 * np.transpose(FMeanFMeanZMeant) + XtF)

        # Calculating the unrestricted log marginal likelihood.
        phi0 = np.transpose(np.concatenate((
            np.concatenate((np.zeros((N, M + 1), dtype=np.float64), np.transpose(beta0)), axis=1), \
            np.zeros((N, K * M), dtype=np.float64)), axis=1))

        phiTilda = T / Tstar * WtWInv @ (WtR + T0 / T * WtW @ phi0)

        RminusWphiTilda = R - W @ phiTilda

        Sr = S0 + np.transpose(RminusWphiTilda) @ (RminusWphiTilda) + \
             T0 / T * np.transpose(phiTilda - phi0) @ WtW @ (phiTilda - phi0)

        # Calculating the restricted.
        phi0R = np.transpose(np.concatenate((np.transpose(beta0), np.zeros((N, K * M), dtype=np.float64)), axis=1))
        WR = np.concatenate((F, Omega), axis=1)
        WRtWR = np.transpose(WR) @ WR

        WRtWRInv = LA.pinv(WRtWR)

        phiTildaR = T / Tstar * WRtWRInv @ (np.transpose(WR) @ R + T0 / T * WRtWR @ phi0R)

        SrR = S0 + np.transpose(R - WR @ phiTildaR) @ (R - WR @ phiTildaR) + \
              T0 / T * np.transpose(phiTildaR - phi0R) @ WRtWR @ (phiTildaR - phi0R)

        # In sample returns and returns square predictions.
        (returns, returns_square, returns_interactions, covariance_matrix, covariance_matrix_no_ER, \
            cov_matrix_single_unrestricted_IN, return_single_unrestricted_IN, \
                cov_matrix_single_restricted_IN, return_single_restricted_IN, \
                Returns_terms_single_weighted_IN, Returns_terms_single_square_weighted_IN, \
                Returns_terms_single_cumulative_probability_IN) = \
                                calculatePredictions(ZOrig, T, AfTilda, phiTilda, phiTildaR, \
                                        KMax, K, M, T, Tstar, \
                                        Sr, SrR, Sf, XtXInv, WtWInv, WRtWRInv, \
                                        predictorsIndicesIncludedInModel, factorsIndicesIncludedInModel, otherFactors,\
                                        models_probabilities[model], models_probabilities[model + nModelsMax])
        
        # np.allclose are possible without Numba.
        #assert np.allclose(covariance_matrix, models_probabilities[model]*cov_matrix_single_unrestricted_IN + \
        #                        models_probabilities[model + nModelsMax] * cov_matrix_single_restricted_IN)
        #assert np.allclose(returns, models_probabilities[model]*return_single_unrestricted_IN + \
        #                            models_probabilities[model + nModelsMax] * return_single_restricted_IN)
        
        assert (TAReturns_IN.shape==(T,KMax)) & (TAReturns_OOS.shape==(TOOS,KMax))

        returns_IN  += returns
        returns_square_IN += returns_square
        returns_interactions_IN += returns_interactions
        covariance_matrix_IN += covariance_matrix
        covariance_matrix_no_ER_IN += covariance_matrix_no_ER
        Returns_terms_weighted_IN += Returns_terms_single_weighted_IN
        Returns_terms_square_weighted_IN += Returns_terms_single_square_weighted_IN
        Returns_terms_cumulative_probability_IN += Returns_terms_single_cumulative_probability_IN

        
        # Calculating the weights of the single portfolio, in-sample.
        for t in np.arange(0,T):
            # Establishing the tangency portfolio for the unrestricted model.
            w_unrestricted_t = LA.pinv(cov_matrix_single_unrestricted_IN[t,:,:]) @ return_single_unrestricted_IN[t,:]
            w_unrestricted_t = w_unrestricted_t/ (np.abs(np.sum(w_unrestricted_t)))
            r_unrestricted_t =  np.dot(w_unrestricted_t[:], TAReturns_IN[t,:])

            # Establishing the tangency portfolio for the unrestricted model.
            w_restricted_t = LA.pinv(cov_matrix_single_restricted_IN[t,:,:]) @ return_single_restricted_IN[t,:]
            w_restricted_t = w_restricted_t/ (np.abs(np.sum(w_restricted_t)))
            r_restricted_t =  np.dot(w_restricted_t[:], TAReturns_IN[t,:])

            weights_sum_IN[t,:] += models_probabilities[model] * w_unrestricted_t + \
                                    models_probabilities[model + nModelsMax] * w_restricted_t

            weights_square_sum_IN[t,:] += models_probabilities[model] * w_unrestricted_t**2 + \
                                    models_probabilities[model + nModelsMax] * w_restricted_t**2

            weights_sum_equal_IN[t,:] += ( w_unrestricted_t + w_restricted_t )/(2*total_number_of_models)

            weights_square_sum_equal_IN[t,:] += ( w_unrestricted_t**2 + w_restricted_t**2 )/(2*total_number_of_models)

            returns_sum_IN[t] += models_probabilities[model] * r_unrestricted_t + \
                                    models_probabilities[model + nModelsMax] * r_restricted_t

            returns_square_sum_IN[t] += models_probabilities[model] * r_unrestricted_t**2 + \
                                    models_probabilities[model + nModelsMax] * r_restricted_t**2

            returns_sum_equal_IN[t] += ( r_unrestricted_t + r_restricted_t )/(2*total_number_of_models)

            returns_square_sum_equal_IN[t] += ( r_unrestricted_t**2 + r_restricted_t**2 )/(2*total_number_of_models)

        # Out of sample returns and returns square predictions.
        (returns, returns_square, returns_interactions, covariance_matrix, covariance_matrix_no_ER, \
            cov_matrix_single_unrestricted_OOS, return_single_unrestricted_OOS, \
                cov_matrix_single_restricted_OOS, return_single_restricted_OOS, \
                Returns_terms_single_weighted_OOS, Returns_terms_single_square_weighted_OOS, \
                Returns_terms_single_cumulative_probability_OOS) = \
                                calculatePredictions(ZTest, TOOS, AfTilda, phiTilda, phiTildaR, \
                                        KMax, K, M, T, Tstar, \
                                        Sr, SrR, Sf, XtXInv, WtWInv, WRtWRInv, \
                                        predictorsIndicesIncludedInModel, factorsIndicesIncludedInModel, otherFactors,\
                                        models_probabilities[model], models_probabilities[model + nModelsMax])

        # np.allclose are possible without Numba.
        #assert np.allclose(covariance_matrix, models_probabilities[model]*cov_matrix_single_unrestricted_OOS + \
        #                        models_probabilities[model + nModelsMax] * cov_matrix_single_restricted_OOS)
        #assert np.allclose(returns, models_probabilities[model]*return_single_unrestricted_OOS + \
        #                            models_probabilities[model + nModelsMax] * return_single_restricted_OOS)
        
        returns_OOS += returns
        returns_square_OOS += returns_square
        returns_interactions_OOS += returns_interactions
        covariance_matrix_OOS += covariance_matrix
        covariance_matrix_no_ER_OOS += covariance_matrix_no_ER

        Returns_terms_weighted_OOS += Returns_terms_single_weighted_OOS
        Returns_terms_square_weighted_OOS += Returns_terms_single_square_weighted_OOS
        Returns_terms_cumulative_probability_OOS += Returns_terms_single_cumulative_probability_OOS

        # Calculating the weights of the single portfolio, out-of-sample
        for t in np.arange(0,TOOS):
            # Establishing the tangency portfolio for the unrestricted model.
            w_unrestricted_t = LA.pinv(cov_matrix_single_unrestricted_OOS[t,:,:]) @ return_single_unrestricted_OOS[t,:]
            w_unrestricted_t = w_unrestricted_t/ (np.abs(np.sum(w_unrestricted_t)))
            r_unrestricted_t =  np.dot(w_unrestricted_t[:], TAReturns_OOS[t,:])

            # Establishing the tangency portfolio for the unrestricted model.
            w_restricted_t = LA.pinv(cov_matrix_single_restricted_OOS[t,:,:]) @ return_single_restricted_OOS[t,:]
            w_restricted_t = w_restricted_t/ (np.abs(np.sum(w_restricted_t)))
            r_restricted_t =  np.dot(w_restricted_t[:], TAReturns_OOS[t,:])

            weights_sum_OOS[t,:] += models_probabilities[model] * w_unrestricted_t + \
                                    models_probabilities[model + nModelsMax] * w_restricted_t

            weights_square_sum_OOS[t,:] += models_probabilities[model] * w_unrestricted_t**2 + \
                                    models_probabilities[model + nModelsMax] * w_restricted_t**2

            weights_sum_equal_OOS[t,:] += ( w_unrestricted_t + w_restricted_t )/(2*total_number_of_models)

            weights_square_sum_equal_OOS[t,:] += ( w_unrestricted_t**2 + w_restricted_t**2 )/(2*total_number_of_models)

            returns_sum_OOS[t] += models_probabilities[model] * r_unrestricted_t + \
                                    models_probabilities[model + nModelsMax] * r_restricted_t

            returns_square_sum_OOS[t] += models_probabilities[model] * r_unrestricted_t**2 + \
                                    models_probabilities[model + nModelsMax] * r_restricted_t**2

            returns_sum_equal_OOS[t] += ( r_unrestricted_t + r_restricted_t )/(2*total_number_of_models)

            returns_square_sum_equal_OOS[t] += ( r_unrestricted_t**2 + r_restricted_t**2 )/(2*total_number_of_models)

        # del returns, returns_square, returns_interactions, covariance_matrix

        T0Total = T0Total + T0
        T0Max = max(T0Max, T0)
        T0Min = min(T0Min, T0)

        cumulative_probability += models_probabilities[model] + models_probabilities[model + nModelsMax]
    #     del N, K, F, R, Z, M

    nLegitModels = max(nLegitModels,1)

    T0IncreasedFraction = nTooSmallT0 / nLegitModels
    T0Avg = T0Total / nLegitModels

    # assert np.allclose(Returns_terms_cumulative_probability_OOS, Returns_terms_cumulative_probability_IN)

    return (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
            weights_sum_OOS, weights_square_sum_OOS, weights_sum_equal_OOS, weights_square_sum_equal_OOS, \
            returns_sum_OOS, returns_square_sum_OOS, returns_sum_equal_OOS, returns_square_sum_equal_OOS, \
            Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, \
            returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
            weights_sum_IN, weights_square_sum_IN, weights_sum_equal_IN, weights_square_sum_equal_IN, \
            returns_sum_IN, returns_square_sum_IN, returns_sum_equal_IN, returns_square_sum_equal_IN, \
            Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, \
            Returns_terms_cumulative_probability_IN, \
            T0IncreasedFraction, T0Max, T0Min, T0Avg, nLegitModels, nTooSmallT0, cumulative_probability, total_number_of_models)

# ========== End of function conditionalAssetCalculateSpreadNumba ==========


# Running analysis in this code.
if __name__ == '__main__':

    class Start(Enum):
        calculate_ML  = 1
        load_results_singles = 3
        predict_OOS = 5
        single_model_predict_OOS = 6
        analyse_OOS = 7
        summary_statistics = 8
        variance_matrix    = 9
        factors_variance   = 10
        factors_inclusion  = 11
        analyse_OOS_performance = 12
        calculate_spread   = 13
        calculate_spread_post_processing = 14

    class DataBase(Enum):
        testAssets80_f14_m13 = 1
        testAssets0_f20_m13 = 2

    homeDir = "/home/lior/Characteristics/python/PostProb/"
    #dataDir = r"C:\Users\Tor Osted\OneDrive\Dokumenter\BMALASTTRY\Complemetary Code Files for Submission\Data"
    dataDir = os.path.join(os.path.join(os.path.join(os.path.join(os.path.dirname(os.getcwd()),'speciale_repo'),'den_her_du_skal_kigge_i'),'Complemetary Code Files for Submission'), 'Data')

    dump_file_prefix = 'conditional_dump_models_MMax_'

    #key_start = Start.calculate_ML
    #key_start = Start.load_results_singles
    key_start = Start.predict_OOS
    #key_start = Start.single_model_predict_OOS
    #key_start = Start.analyse_OOS
    #key_start = Start.summary_statistics
    #key_start = Start.variance_matrix
    #key_start = Start.factors_variance
    #key_start = Start.analyse_OOS_performance
    #key_start = Start.calculate_spread
    #key_start = Start.calculate_spread_post_processing

    #key_DataBase = DataBase.testAssets80_f14_m13
    key_DataBase = DataBase.testAssets0_f20_m13

    # Loading the input files.
    if key_DataBase == DataBase.testAssets80_f14_m13:
        directory_name_prefix = os.path.join(homeDir,'conditionalTauDMN0F14M13s')
        significantPredictors = np.array([2,5,7,9,13]) ; significantPredictors -= 1
        significantPredictors = np.array([2])
        key_Avoid_duplicate_factors = True

        # All factors and test assets returns are in percentage.
        # Loading the input files.
        # The risk free rate.
       # RF = pd.read_csv(dataDir + "\\RF.csv")
      #  RF = RF.drop(columns='Unnamed: 0')

        # Test asstets. Loading all portfolios.

        R = pd.read_csv(os.path.join(dataDir,'returns.csv'))
        #R = R.drop(columns='Unnamed: 0')

        testAssetsPortfoliosNames = R.columns.drop('Date')
        print(' max, min of R are= %f, %f' % (np.max(R[testAssetsPortfoliosNames].values), np.min(R[testAssetsPortfoliosNames].values)))

        # Subtract the risk free rate from the test assets to get excess return.
        # The original test assets returns are numbers and not percentage. Convert all the returns to percentage.
        for name in testAssetsPortfoliosNames:
            R[name] = R[name].values * 100 #- RF['RF'].values

        print(' max, min of R are= %f, %f' % (np.max(R[testAssetsPortfoliosNames].values), np.min(R[testAssetsPortfoliosNames].values)))

        # Loading the factors.
        F = pd.read_csv(os.path.join(dataDir,'factors-20.csv'))
        #F = F.drop(columns='Unnamed: 0')
        factorsNames = F.columns.drop('Date')
        # Loading the predictors.
        Z = pd.read_csv(os.path.join(dataDir,'Z - 197706.csv'))
        Z = Z.drop(columns='Unnamed: 0')

    elif key_DataBase == DataBase.testAssets0_f20_m13:
        directory_name_prefix = homeDir + '\\conditionalTauDMN0F14M13s'

        significantPredictors = np.array([1,4,9,11])
        # significantPredictors = np.array([])
        significantPredictors = np.array([0, 1, 2, 3, 4, 5, 6])#, 7, 8, 9, 10, 11, 12])

        key_Avoid_duplicate_factors = False

        key_Avoid_duplicate_predictors = (len(significantPredictors) == 13)

        R = pd.DataFrame({'': []})

        # Loading the factors.
        F = pd.read_csv(os.path.join(dataDir,'factors-20.csv'))
        F = F.drop(columns=['SMB*','MKT','IA', 'ROE', 'ME'])
        factorsNames = F.columns.drop('Date')

        # The factors in this input file are numbers and not percentage. Convert all the returns to percentage.
        for name in factorsNames:
            F[name] = F[name].values * 100

        # Loading the predictors.
        Z = pd.read_csv(os.path.join(dataDir,'Z - 197706.csv'))
        Z = Z.drop(columns='Unnamed: 0')

        assert len(Z) == len(F)

    predictorsNames = Z.columns.drop('Date') ; predictorsNames = predictorsNames[significantPredictors]

    print(F.columns)
    print(Z.columns)

    # Tau from 2020 - Chib Zeng Zhao - On Comparing Asset Pricing Models (JF)
    # to relate the alpha covariance matrix to the residuals covariance matrix. \neu = 9Tau^2 * SR^2_Mkt - SR^2_Mtk)/N
    TauArray = np.array([1.25, 1.5, 2.0, 3.0])

    # Execute single value of TauArray and run in parallel. Use generateInputFilesForDifferentSigamaAlpha
    # to generate the single input files.
    if len(significantPredictors) == 13:
        #istart = 123456
        istart = 0
        iend   = istart + 1
    else:
        istart = 0
        iend   = len(TauArray)

    if key_start == Start.calculate_ML or key_start == Start.predict_OOS or \
        key_start == Start.single_model_predict_OOS or key_start == Start.calculate_spread:
        CLMLList = []
        for Tau in TauArray[istart:iend]:
            # This is the function that gets the pandas data frames for the log marginal likelihood calculation.
            #(CLMLU, factorsNames, factorsProbabilityU, predictorsNames, predictorsProbabilityU, T0IncreasedFraction,
            #  T0Max, T0Min, T0Avg, CLMLR, factorsProbabilityR, predictorsProbabilityR) = \
            #     conditionalAssetPricingLogMarginalLikelihoodTau(R, F, Z, significantPredictors, Tau)

            # Similat to Barillas and Shanke, they tool for the out of sample T/2 and 2T/3 for the estimation period.
            #IndexEndEstimation = 246  # Equals to T/2 and closest to years end 1997-12 (T=474).
            IndexEndEstimation = 318  # Equals to 2T/3 and closest to years end 2003-12 (T=474)
            #IndexEndEstimation = None  # Full sample
            BMA=Model(R, F, Z, significantPredictors, Tau, indexEndOfEstimation=IndexEndEstimation, key_demean_predictors=True)

            if key_start == Start.calculate_ML:
                (CLMLU, factorsNames, factorsProbabilityU, predictorsNames, predictorsProbabilityU, \
                    T0IncreasedFraction, T0Max, T0Min, T0Avg, T0_div_T0_plus_TAvg, T_div_T0_plus_TAvg, \
                    CLMLR, factorsProbabilityR, predictorsProbabilityR) = \
                        BMA.conditionalAssetPricingLogMarginalLikelihoodTauNumba()

                CLMLCombined = np.concatenate((CLMLU, CLMLR), axis=0)
                CMLCombined  = np.exp(CLMLCombined - max(CLMLCombined)); CMLCombined = CMLCombined / np.sum(CMLCombined)
                print('probability of mispricing    = %f' %(np.sum(CMLCombined[0: len(CLMLU)])))
                print('probability of no mispricing = %f' %(np.sum(CMLCombined[len(CLMLU):  ])))
                factorsProbability = np.sum(CMLCombined[0:len(CLMLU)])*factorsProbabilityU + \
                                    np.sum(CMLCombined[len(CLMLU): ]) *factorsProbabilityR

                predictorsProbability = np.sum(CMLCombined[0:len(CLMLU)])*predictorsProbabilityU + \
                                    np.sum(CMLCombined[len(CLMLU): ]) *predictorsProbabilityR

                printFactorsAndPredictorsProbabilities(factorsNames, factorsProbability, predictorsNames,
                                                   predictorsProbability)

                CLMLList.append({"Tau": Tau,     \
                            "LMLU" : CLMLU, "LMLR" :CLMLR,  \
                            "factorsProbability"    : factorsProbability,    \
                            "predictorsProbability" : predictorsProbability, \
                            "T0IncreasedFraction" : T0IncreasedFraction,     \
                            "T0Max" : T0Max, "T0Min" : T0Min, "T0Avg" : T0Avg,   \
                            "T0divT0plusTAvg" : T0_div_T0_plus_TAvg, "TdivT0plusTAvg" : T_div_T0_plus_TAvg, \
                            "MisprisingProb" : np.sum(CMLCombined[0: len(CLMLU)])})

                with open(dump_file_prefix+str(len(significantPredictors))+'.pkl', 'wb') as file:
                    pickle.dump([CLMLList, significantPredictors], file)

                del CLMLCombined, CLMLU, CLMLR, CMLCombined

            if key_start == Start.predict_OOS or key_start == Start.single_model_predict_OOS or key_start ==  Start.calculate_spread:
                file_name = os.getcwd() + "/" + dump_file_prefix + str(len(significantPredictors)) + '.pkl'
                print("In predicting OOS. Loading file - " + str(file_name))
                with open(file_name, 'rb') as file:
                    [CLMLList, significantPredictors_temp] = pickle.load(file)

                CLMLU = CLMLList[0]["LMLU"]
                CLMLR = CLMLList[0]["LMLR"]
                CLMLCombined = np.concatenate((CLMLU, CLMLR), axis=0)
                CMLCombined  = np.exp(CLMLCombined - max(CLMLCombined)); CMLCombined = CMLCombined / np.sum(CMLCombined)
                print('probability of mispricing    = %f' %(np.sum(CMLCombined[0: len(CLMLU)])))
                print('probability of no mispricing = %f' %(np.sum(CMLCombined[len(CLMLU):  ])))
                factorsProbability = CLMLList[0]["factorsProbability"]

                predictorsProbability = CLMLList[0]["predictorsProbability"]

                printFactorsAndPredictorsProbabilities(BMA.factorsNames, factorsProbability, BMA.predictorsNames,
                                                   predictorsProbability)

                if key_start == Start.predict_OOS:
                    nModelsInPrediction = 20000000
                    #nModelsInPrediction =  5000000
                    
                    (returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                        covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                        returns_IN, returns_square_IN, returns_interactions_IN, \
                        covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                        cumulative_probability) = \
                        BMA.conditionalAssetPricingOOSTauNumba(CMLCombined,nModelsInPrediction)

                    assert (returns_OOS.shape == returns_square_OOS.shape) & \
                        (returns_interactions_OOS.shape == (returns_OOS.shape[0], returns_OOS.shape[1],returns_OOS.shape[1])) & \
                        (returns_OOS.shape == (BMA.ZTest.shape[0],BMA.RTest.shape[1]+BMA.FTest.shape[1])) & \
                        (returns_IN.shape == returns_square_IN.shape) & \
                        (returns_interactions_IN.shape == (returns_IN.shape[0], returns_IN.shape[1],returns_IN.shape[1])) & \
                        (returns_IN.shape ==(BMA.ZEstimation.shape[0],BMA.REstimation.shape[1]+BMA.FEstimation.shape[1]))

                    with open(dump_file_prefix+str(len(significantPredictors))+'_OOS_new1.pkl', 'wb') as file:
                        pickle.dump([returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                                    covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                                    returns_IN, returns_square_IN, returns_interactions_IN, \
                                    covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                                    cumulative_probability, BMA], file)
                
                elif key_start == Start.single_model_predict_OOS:
                    # Analying the top three models
                    for single_top_model in np.arange(1,4):
                        (returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                            covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                            returns_IN, returns_square_IN, returns_interactions_IN, \
                            covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                            cumulative_probability) = \
                            BMA.conditionalAssetPricingSingleOOSTauNumba(CMLCombined, single_top_model)
                    
                        assert returns_OOS.shape == (BMA.ZTest.shape[0],BMA.RTest.shape[1]+BMA.FTest.shape[1])
                        assert returns_OOS.shape == returns_square_OOS.shape
                        assert returns_interactions_OOS.shape == (returns_OOS.shape[0], returns_OOS.shape[1], returns_OOS.shape[1])
                        assert covariance_matrix_OOS.shape == (returns_OOS.shape[0], returns_OOS.shape[1], returns_OOS.shape[1])
                        assert covariance_matrix_no_ER_OOS.shape == covariance_matrix_OOS.shape
                        assert returns_IN.shape ==(BMA.ZEstimation.shape[0],BMA.REstimation.shape[1]+BMA.FEstimation.shape[1])
                        assert returns_IN.shape == returns_square_IN.shape
                        assert returns_interactions_IN.shape == (returns_IN.shape[0], returns_IN.shape[1],returns_IN.shape[1])
                        assert covariance_matrix_IN.shape == (returns_IN.shape[0], returns_IN.shape[1], returns_IN.shape[1])
                        assert covariance_matrix_no_ER_IN.shape == covariance_matrix_IN.shape

                        dump_file_name = dump_file_prefix + str(len(significantPredictors)) + '_OOS_new1_single_' + str(single_top_model) + '.pkl'
                        print('dump_file_name: %s' %(dump_file_name))
                        with open(dump_file_name, 'wb') as file:
                            pickle.dump([returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                                    covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                                    returns_IN, returns_square_IN, returns_interactions_IN, \
                                    covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                                    cumulative_probability, BMA], file)

                elif key_start == Start.calculate_spread:
                    nModelsInPrediction = 20000000
                    #nModelsInPrediction =  5000000

                    (returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                        weights_sum_OOS, weights_square_sum_OOS, weights_sum_equal_OOS, weights_square_sum_equal_OOS, \
                        returns_sum_OOS, returns_square_sum_OOS, returns_sum_equal_OOS, returns_square_sum_equal_OOS, \
                        Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, \
                        returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                        weights_sum_IN, weights_square_sum_IN, weights_sum_equal_IN, weights_square_sum_equal_IN, \
                        returns_sum_IN, returns_square_sum_IN, returns_sum_equal_IN, returns_square_sum_equal_IN, \
                        Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, \
                        Returns_terms_cumulative_probability, \
                        cumulative_probability, total_number_of_models) = \
                        BMA.conditionalAssetCalculateSpread(CMLCombined,nModelsInPrediction)

                    assert returns_OOS.shape == (BMA.ZTest.shape[0],BMA.RTest.shape[1]+BMA.FTest.shape[1])
                    assert returns_OOS.shape == returns_square_OOS.shape
                    assert returns_interactions_OOS.shape == (returns_OOS.shape[0], returns_OOS.shape[1], returns_OOS.shape[1])
                    assert covariance_matrix_OOS.shape == (returns_OOS.shape[0], returns_OOS.shape[1], returns_OOS.shape[1])
                    assert covariance_matrix_no_ER_OOS.shape == covariance_matrix_OOS.shape
                    assert weights_sum_OOS.shape == (BMA.ZTest.shape[0],BMA.RTest.shape[1]+BMA.FTest.shape[1])
                    assert weights_sum_OOS.shape == weights_square_sum_OOS.shape
                    assert weights_sum_OOS.shape == weights_sum_equal_OOS.shape
                    assert weights_sum_OOS.shape == weights_square_sum_equal_OOS.shape
                    assert returns_sum_OOS.shape == (BMA.ZTest.shape[0],)
                    assert returns_sum_OOS.shape == returns_square_sum_OOS.shape
                    assert returns_sum_OOS.shape == returns_sum_equal_OOS.shape
                    assert returns_sum_OOS.shape == returns_square_sum_equal_OOS.shape
                    assert Returns_terms_weighted_OOS.shape == (BMA.ZTest.shape[0],BMA.RTest.shape[1]+BMA.FTest.shape[1], 8)
                    assert Returns_terms_square_weighted_OOS.shape == Returns_terms_weighted_OOS.shape
                    assert returns_IN.shape ==(BMA.ZEstimation.shape[0],BMA.REstimation.shape[1]+BMA.FEstimation.shape[1])
                    assert returns_IN.shape == returns_square_IN.shape
                    assert returns_interactions_IN.shape == (returns_IN.shape[0], returns_IN.shape[1],returns_IN.shape[1])
                    assert covariance_matrix_IN.shape == (returns_IN.shape[0], returns_IN.shape[1], returns_IN.shape[1])
                    assert covariance_matrix_no_ER_IN.shape == covariance_matrix_IN.shape
                    assert weights_sum_IN.shape == (BMA.ZEstimation.shape[0],BMA.REstimation.shape[1]+BMA.FEstimation.shape[1])
                    assert weights_sum_IN.shape == weights_square_sum_IN.shape
                    assert weights_sum_IN.shape == weights_sum_equal_IN.shape
                    assert weights_sum_IN.shape == weights_square_sum_equal_IN.shape
                    assert returns_sum_IN.shape == (BMA.ZEstimation.shape[0],)
                    assert returns_sum_IN.shape == returns_square_sum_IN.shape
                    assert returns_sum_IN.shape == returns_sum_equal_IN.shape
                    assert returns_sum_IN.shape == returns_square_sum_equal_IN.shape
                    assert Returns_terms_weighted_IN.shape == (BMA.ZEstimation.shape[0],BMA.REstimation.shape[1]+BMA.FEstimation.shape[1], 8)
                    assert Returns_terms_square_weighted_IN.shape == Returns_terms_weighted_IN.shape
                    assert Returns_terms_cumulative_probability.shape == (BMA.REstimation.shape[1]+BMA.FEstimation.shape[1], 8)

                    with open(dump_file_prefix+str(len(significantPredictors))+'_spread_new1.pkl', 'wb') as file:
                        pickle.dump([returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                            covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                            weights_sum_OOS, weights_square_sum_OOS, weights_sum_equal_OOS, weights_square_sum_equal_OOS, \
                            returns_sum_OOS, returns_square_sum_OOS, returns_sum_equal_OOS, returns_square_sum_equal_OOS, \
                            Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, \
                            returns_IN, returns_square_IN, returns_interactions_IN, \
                            covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                            weights_sum_IN, weights_square_sum_IN, weights_sum_equal_IN, weights_square_sum_equal_IN, \
                            returns_sum_IN, returns_square_sum_IN, returns_sum_equal_IN, returns_square_sum_equal_IN, \
                            Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, \
                            Returns_terms_cumulative_probability, \
                            cumulative_probability, total_number_of_models, BMA], file)

    elif key_start == Start.calculate_spread_post_processing:
        print(' ****** calculate_spread_post_processing ******')
        dump_directory =''
        dump_directory ='conditionalTau2T_3NumbaDMN0F14M13s1/'     

        dump_file_name = dump_directory + dump_file_prefix + str(len(significantPredictors)) + '_spread_new1.pkl'
        print('dump_file_name: %s' %(dump_file_name))
        with open(dump_file_name, 'rb') as file:
            [returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                weights_sum_OOS, weights_square_sum_OOS, weights_sum_equal_OOS, weights_square_sum_equal_OOS, \
                returns_sum_OOS, returns_square_sum_OOS, returns_sum_equal_OOS, returns_square_sum_equal_OOS, \
                Returns_terms_weighted_OOS, Returns_terms_square_weighted_OOS, \
                returns_IN, returns_square_IN, returns_interactions_IN, \
                covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                weights_sum_IN, weights_square_sum_IN, weights_sum_equal_IN, weights_square_sum_equal_IN, \
                returns_sum_IN, returns_square_sum_IN, returns_sum_equal_IN, returns_square_sum_equal_IN, \
                Returns_terms_weighted_IN, Returns_terms_square_weighted_IN, \
                Returns_terms_cumulative_probability, \
                cumulative_probability, total_number_of_models, BMA] = pickle.load(file)

            # Output
            # Fields (8) of Returns_terms_cumulative_probability.
            ReturnTermNameList = ['static mispricing', 'time varying mispricing', \
                            'static beta * static premium', 'static beta * time varying premium', \
                            'time varying beta * static premium', 'time varying beta * time varying premium',\
                            'Static Risk premium', 'time varying risk premium']
                   
            ReturnTermNameSymbolList = ['$ \\alpha_0 $', '$ \\alpha_1 z_t $', \
                            '$ \\beta_0 \\alpha_f $', '$ \\beta_0  a_F  z_t $', \
                            '$ \\beta_1 \\left(I \\otimes z_t \\right) \\alpha_f $', '$ \\beta_1 (I \\otimes z_t) a_F z_t $', \
                            '$ \\alpha_f $', '$ a_F z_t $']

            factorsNamesList = list(factorsNames)
            factorsNamesList[0]='MKT' ; factorsNamesList[5]='MOM'; factorsNamesList[13]='ICR'

            for nTA in np.arange(0,Returns_terms_cumulative_probability.shape[0]):
                print('Test asset %i' % nTA)
                assert np.allclose(Returns_terms_cumulative_probability[nTA,6], Returns_terms_cumulative_probability[nTA,7])
                assert np.allclose(Returns_terms_cumulative_probability[nTA,4], Returns_terms_cumulative_probability[nTA,5])
                assert np.allclose(Returns_terms_cumulative_probability[nTA,2] + Returns_terms_cumulative_probability[nTA,6], \
                        cumulative_probability)

        dump_file_name = dump_directory + dump_file_prefix + str(len(significantPredictors)) + '_spread_new.pkl'
        print('dump_file_name: %s' %(dump_file_name))

        with open(dump_file_name, 'rb') as file:
            [returns_OOSx , returns_square_OOSx, returns_interactions_OOSx, \
                covariance_matrix_OOSx, covariance_matrix_no_ER_OOSx, \
                weights_sum_OOSx, weights_square_sum_OOSx, weights_sum_equal_OOSx, weights_square_sum_equal_OOSx, \
                returns_sum_OOSx, returns_square_sum_OOSx, returns_sum_equal_OOSx, returns_square_sum_equal_OOSx, \
                returns_INx, returns_square_INx, returns_interactions_INx, \
                covariance_matrix_INx, covariance_matrix_no_ER_INx, \
                weights_sum_INx, weights_square_sum_INx, weights_sum_equal_INx, weights_square_sum_equal_INx, \
                returns_sum_INx, returns_square_sum_INx, returns_sum_equal_INx, returns_square_sum_equal_INx, \
                cumulative_probabilityx, total_number_of_modelsx, BMAx] = pickle.load(file)

            assert np.allclose(returns_OOSx, returns_OOS)
            assert np.allclose(returns_square_OOSx, returns_square_OOS)
            assert np.allclose(returns_interactions_OOSx, returns_interactions_OOS)
            assert np.allclose(covariance_matrix_OOSx, covariance_matrix_OOS)
            assert np.allclose(covariance_matrix_no_ER_OOSx, covariance_matrix_no_ER_OOS)
            assert np.allclose(weights_sum_OOSx, weights_sum_OOS)
            assert np.allclose(weights_square_sum_OOSx, weights_square_sum_OOS)
            assert np.allclose(weights_sum_equal_OOSx, weights_sum_equal_OOS)
            assert np.allclose(weights_square_sum_equal_OOSx, weights_square_sum_equal_OOS)
            assert np.allclose(returns_sum_OOSx, returns_sum_OOS)
            assert np.allclose(returns_square_sum_OOSx, returns_square_sum_OOS)
            assert np.allclose(returns_sum_equal_OOSx, returns_sum_equal_OOS)
            assert np.allclose(returns_square_sum_equal_OOSx, returns_square_sum_equal_OOS)
            assert np.allclose(returns_INx, returns_IN)
            assert np.allclose(returns_square_INx, returns_square_IN)
            assert np.allclose(returns_interactions_INx, returns_interactions_IN)
            assert np.allclose(covariance_matrix_INx, covariance_matrix_IN)
            assert np.allclose(covariance_matrix_no_ER_INx, covariance_matrix_no_ER_IN)
            assert np.allclose(weights_sum_INx, weights_sum_IN)
            assert np.allclose(weights_square_sum_INx, weights_square_sum_IN)
            assert np.allclose(weights_sum_equal_INx, weights_sum_equal_IN)
            assert np.allclose(weights_square_sum_equal_INx, weights_square_sum_equal_IN)
            assert np.allclose(returns_sum_INx, returns_sum_IN)
            assert np.allclose(returns_square_sum_INx, returns_square_sum_IN)
            assert np.allclose(returns_sum_equal_INx, returns_sum_equal_IN)
            assert np.allclose(returns_square_sum_equal_INx, returns_square_sum_equal_IN)
            assert cumulative_probabilityx == cumulative_probability
            assert total_number_of_modelsx == total_number_of_models

        print(' ***** Renormalizing by cumulative_probability = %f, number of models %i' % (cumulative_probability, total_number_of_models) )
        # Out-of-sample
        returns_OOS /= cumulative_probability
        returns_square_OOS /= cumulative_probability
        returns_interactions_OOS /= cumulative_probability
        covariance_matrix_OOS /= cumulative_probability
        covariance_matrix_no_ER_OOS /= cumulative_probability
        weights_sum_OOS /= cumulative_probability
        weights_square_sum_OOS /= cumulative_probability
        returns_sum_OOS /= cumulative_probability
        returns_square_sum_OOS /= cumulative_probability

        xxxx = Returns_terms_weighted_OOS / (1e-10+Returns_terms_cumulative_probability)
        xxxx1 = Returns_terms_square_weighted_OOS / (1e-10+Returns_terms_cumulative_probability)
        NTA = Returns_terms_weighted_OOS.shape[1]
        for t in np.arange(0, Returns_terms_weighted_OOS.shape[0]):
            for nTA in np.arange(0, NTA):
                Returns_terms_weighted_OOS[t, nTA, :] /= (1e-10+Returns_terms_cumulative_probability[nTA,:])
                Returns_terms_square_weighted_OOS[t, nTA, :] /= (1e-10+Returns_terms_cumulative_probability[nTA,:])
        
        assert np.allclose(xxxx, Returns_terms_weighted_OOS)
        assert np.allclose(xxxx1, Returns_terms_square_weighted_OOS)

        # In-sample
        returns_IN /= cumulative_probability
        returns_square_IN /= cumulative_probability
        returns_interactions_IN /= cumulative_probability
        covariance_matrix_IN /= cumulative_probability
        covariance_matrix_no_ER_IN /= cumulative_probability
        weights_sum_IN /= cumulative_probability
        weights_square_sum_IN /= cumulative_probability
        returns_sum_IN /= cumulative_probability
        returns_square_sum_IN /= cumulative_probability
        xxxx = Returns_terms_weighted_IN / (1e-10+Returns_terms_cumulative_probability)
        xxxx1 = Returns_terms_square_weighted_IN / (1e-10+Returns_terms_cumulative_probability)
        for t in np.arange(0, Returns_terms_weighted_IN.shape[0]):
            for nTA in np.arange(0, NTA):
                Returns_terms_weighted_IN[t, nTA, :] /= (1e-10+Returns_terms_cumulative_probability[nTA,:])
                Returns_terms_square_weighted_IN[t, nTA, :] /= (1e-10+Returns_terms_cumulative_probability[nTA,:])
        
        assert np.allclose(xxxx, Returns_terms_weighted_IN)
        assert np.allclose(xxxx1, Returns_terms_square_weighted_IN)
        
        Returns_terms_cumulative_probability /= cumulative_probability
        Returns_terms_weighted = np.concatenate((Returns_terms_weighted_IN,Returns_terms_weighted_OOS), axis=0)
        Returns_terms_square_weighted = np.concatenate((Returns_terms_square_weighted_IN,Returns_terms_square_weighted_OOS), axis=0)
        assert Returns_terms_weighted.shape == (Returns_terms_weighted_IN.shape[0]+Returns_terms_weighted_OOS.shape[0],\
            Returns_terms_weighted_OOS.shape[1],Returns_terms_weighted_OOS.shape[2])
        assert Returns_terms_weighted.shape == Returns_terms_square_weighted.shape
        #
        weights_avg = np.concatenate((weights_sum_IN, weights_sum_OOS), axis=0)
        weights_square_avg = np.concatenate((weights_square_sum_IN, weights_square_sum_OOS), axis=0)
        weights_spread_std_mean = np.sqrt(weights_square_avg-weights_avg**2)/np.abs(weights_avg)
        weights_spread_2nd_mom_first = weights_square_avg / np.abs(weights_avg)
        weights_spread_2nd_mom = copy.deepcopy(weights_square_avg)
        weights_spread_std = np.sqrt(weights_square_avg-weights_avg**2)
        weights_spread = weights_spread_std
        returns_avg = np.concatenate((returns_sum_IN, returns_sum_OOS),axis=0)
        returns_square_avg = np.concatenate((returns_square_sum_IN, returns_square_sum_OOS),axis=0)
        returns_spread = np.sqrt(returns_square_avg-returns_avg**2)

        DFDates = pd.DataFrame(F['Date'][1:].values, columns=["Date"])
        DFDates['Date Plot'] = ((DFDates['Date'].values)%100)/12+np.floor(DFDates['Date'].values/100)

        DF = pd.DataFrame(weights_avg, columns=factorsNamesList)
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.to_csv(dump_directory+"weights_avg.csv", index=False)
        #
        DF = pd.DataFrame(weights_square_avg, columns=factorsNamesList)
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.to_csv(dump_directory+"weights_square_avg.csv", index=False)
        #
        DF = pd.DataFrame(weights_spread, columns=factorsNamesList)
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.to_csv(dump_directory+"weights_spread.csv", index=False)
        #
        DF = pd.DataFrame(weights_spread_std, columns=factorsNamesList)
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.to_csv(dump_directory+"weights_spread_std.csv", index=False)
        #
        DF = pd.DataFrame()
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.insert(1, 'R_avg', returns_avg)
        DF.to_csv(dump_directory+"R_avg.csv", index=False)
        #
        DF = pd.DataFrame()
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.insert(1, 'R_square_avg', returns_square_avg)
        DF.to_csv(dump_directory+"R_square_avg.csv", index=False)
        #
        DF = pd.DataFrame()
        DF.insert(0,'Date',DFDates['Date'].values)
        DF.insert(1, 'returns_spread', returns_spread)
        DF.to_csv(dump_directory+"R_spread_std.csv", index=False)
        fig_size=(6.4, 3.6)
        line_width = 1.1
        fig, axs = plt.subplots()
        fig.set_size_inches(fig_size[0],fig_size[1])
        axs.set_xlabel('Year')
        axs.set_ylabel('Standard Deviation of Tangency Portfolio Returns')
        axs.plot(DFDates['Date Plot'].values, returns_spread, linewidth=line_width, label='Standard Deviation of Returns')
        axs.axvline(x=DFDates['Date Plot'].values[318-1], color='b', linestyle='--')
        #axs.legend(fontsize='xx-small')
        fig.savefig(dump_directory+'Return_spread_std' + '.png', dpi=700)
        plt.close(fig)
        #
        DF = pd.DataFrame(Returns_terms_cumulative_probability, columns=ReturnTermNameList, index=factorsNamesList)
        DF.insert(0,'Factor',factorsNamesList)
        DF.to_csv(dump_directory+"Returns_terms_cumulative_probability.csv", index=False)

        figAllW, axsAllW = plt.subplots(7,2,sharex=True, sharey=False, figsize=(15,15))
        figAllRT, axsAllRT = plt.subplots(7,2,sharex=True, sharey=False, figsize=(15,15))
        plot_fields = [1, 3, 4, 5, 7]
        
        font_size='small'
        #figAllW.set_size_inches(fig_size[0],1.2*fig_size[1])
        key_only_combined_figure = True
        for k in np.arange(0, NTA):
            #
            if not key_only_combined_figure:
                #
                DF = pd.DataFrame(Returns_terms_weighted[:,k,:], columns=ReturnTermNameList)
                DF.insert(0,'Date',DFDates['Date'].values)
                DF.to_csv(dump_directory+"Returns_terms_avg_"+str(factorsNamesList[k])+".csv", index=False)
                #
                DF = pd.DataFrame(Returns_terms_square_weighted[:,k,:], columns=ReturnTermNameList)
                DF.insert(0,'Date',DFDates['Date'].values)
                DF.to_csv(dump_directory+"Returns_terms_square_avg_"+str(factorsNamesList[k])+".csv", index=False)

                fig_size=(6.4, 3.6)
                fig, axs = plt.subplots()
                fig.set_size_inches(fig_size[0],fig_size[1])
                axs.set_xlabel('Year')
                axs.set_ylabel('Weight Spread')
                axs.plot(DFDates['Date Plot'].values, weights_spread[:,k], linewidth=line_width, label='Weight Spread')
                axs.axvline(x=DFDates['Date Plot'].values[318-1], color='b', linestyle='--')
                axs.legend(fontsize='xx-small')
                fig.savefig(dump_directory+'Weight_spread_' + str(factorsNamesList[k]) + '.png', dpi=700)
                plt.close(fig)

                fig_size=(6.4, 3.6)
                fig, axs = plt.subplots()
                fig.set_size_inches(fig_size[0],fig_size[1])
                axs.set_xlabel('Year')
                axs.set_ylabel('Return Terms Spread')
            
                fig1, axs1 = plt.subplots()
                fig1.set_size_inches(fig_size[0],fig_size[1])
                axs1.set_xlabel('Year')
                axs1.set_ylabel('Standard Deviation')

            # Regressions
            #x = np.ones((len(portfolio_weights), 1), dtype=np.float64)
            #x = np.concatenate((x, 1/DF.loc[:,"VMR " + sample].values.reshape(-1,1)), axis=1)
            #x = np.concatenate((x, relative_contribution[:,k].reshape(-1,1)), axis=1)
            #y = portfolio_weights.loc[:,factorsNames[k]].values
            #mod = sm.OLS(y,x)

            #res = mod.fit()
            #print('Factor %s - Relative contribution %f Regression parameters - ' \
            #    %(factorsNames[k],np.mean(relative_contribution[:,k],axis=0)))
            #print(np.concatenate((res.params,res.tvalues),axis=0))

            returns_terms_spread = np.sqrt(Returns_terms_square_weighted[:,k,:]-Returns_terms_weighted[:,k,:]**2) #/np.abs(1e-10+np.abs(Returns_terms_weighted[:,k,:]))
            returns_terms_spread_prob_scaled = copy.deepcopy(returns_terms_spread)
            print(str(factorsNamesList[k])+ ' ' + str(k))
            print(np.max(returns_terms_spread,axis=0))
            print(Returns_terms_cumulative_probability[k,:])
            for field in np.arange(0,Returns_terms_square_weighted.shape[2]):
                colorSTR='C'+str(field+1)
                if not key_only_combined_figure:
                    axs.plot(DFDates['Date Plot'].values, returns_terms_spread[:,field], color=colorSTR, linewidth=line_width, label=ReturnTermNameList[field])
                    axs.axvline(x=DFDates['Date Plot'].values[318-1], color='b', linestyle='--')
                    axs.legend(fontsize='xx-small')
                    fig.savefig(dump_directory+'Spread_of_returns_terms_' + str(factorsNamesList[k]) + '.png', dpi=700)
                #
                returns_terms_spread_prob_scaled[:,field] *= Returns_terms_cumulative_probability[k,field]
                if not key_only_combined_figure:
                    axs1.plot(DFDates['Date Plot'].values, returns_terms_spread_prob_scaled[:,field], color=colorSTR, linewidth=line_width, label=ReturnTermNameList[field])
                    axs1.axvline(x=DFDates['Date Plot'].values[318-1], color='b', linestyle='--')
                    axs1.legend(fontsize='xx-small')
                    #fig1.savefig(dump_directory+'Spread_of_return_terms_prob_scaled_' + str(factorsNamesList[k]) + '.png', dpi=700)
                    
                # Plot all factors in one plot.
                if field in plot_fields:
                    if k == 1:
                        axsAllRT[int(np.floor(k/2)), np.mod(k,2)].plot(DFDates['Date Plot'].values, returns_terms_spread[:,field], color=colorSTR, linewidth=line_width, label=ReturnTermNameSymbolList[field])
                        axsAllRT[int(np.floor(k/2)), np.mod(k,2)].legend(loc='upper left', fontsize=font_size, bbox_to_anchor=(1., 1.15))
                    else:
                        axsAllRT[int(np.floor(k/2)), np.mod(k,2)].plot(DFDates['Date Plot'].values, returns_terms_spread[:,field], color=colorSTR, linewidth=line_width)
    
                    # axsAllW[int(np.floor(k/2)), np.mod(k,2)].axvline(x=DFDates['Date Plot'].values[246-1], color='b', linestyle='--')
                    axsAllRT[int(np.floor(k/2)), np.mod(k,2)].axvline(x=DFDates['Date Plot'].values[318-1], color='b', linestyle='--')
                    axsAllRT[int(np.floor(k/2)), np.mod(k,2)].set_title(str(factorsNamesList[k]), fontsize = 'medium') # add pad=1.
                
                    if k>=12: 
                        axsAllRT[int(np.floor(k/2)), np.mod(k,2)].set_xlabel('Year')
                    if k>=6 and k<=7:
                        axsAllRT[int(np.floor(k/2)), np.mod(k,2)].set_ylabel('Standard Deviation')
                    # figAllW.tight_layout()  # Not working well.
                    # # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
                    
                    figAllRT.subplots_adjust(left  = 0.05, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.15, hspace = 0.5)
#                    figAllRT.savefig(dump_directory + 'ReturnTermsSTDCombined' + '.png', dpi=700)
                
            # Plot all factors in one plot.
            if k == 1:
                axsAllW[int(np.floor(k/2)), np.mod(k,2)].plot(DFDates['Date Plot'].values, weights_spread[:,k], linewidth=line_width, label='Weight Dispersion')
                #axsAllW[int(np.floor(k/2)), np.mod(k,2)].legend(loc='upper left', fontsize=font_size, bbox_to_anchor=(1., 1.15))
            else:
                axsAllW[int(np.floor(k/2)), np.mod(k,2)].plot(DFDates['Date Plot'].values, weights_spread[:,k], linewidth=line_width)
    
                # axsAllW[int(np.floor(k/2)), np.mod(k,2)].axvline(x=DFDates['Date Plot'].values[246-1], color='b', linestyle='--')
            axsAllW[int(np.floor(k/2)), np.mod(k,2)].axvline(x=DFDates['Date Plot'].values[318-1], color='b', linestyle='--')
            axsAllW[int(np.floor(k/2)), np.mod(k,2)].set_title(str(factorsNamesList[k]), fontsize = 'medium') # add pad=1.
                
                
            if k>=12: 
                axsAllW[int(np.floor(k/2)), np.mod(k,2)].set_xlabel('Year')
            if k>=6 and k<=7:
                axsAllW[int(np.floor(k/2)), np.mod(k,2)].set_ylabel('Standard Deviation of Tangency Portfolio Weights')
                # figAllW.tight_layout()  # Not working well.
                # # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
            
            figAllW.subplots_adjust(left  = 0.05, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.15, hspace = 0.5)
            #figAllW.savefig(dump_directory + 'WeightsSpreadCombined' + '.png', dpi=700)
            
            if not key_only_combined_figure:
                #
                plt.close(fig)
                plt.close(fig1)
            #
            print(np.max(returns_terms_spread_prob_scaled,axis=0))
            if not key_only_combined_figure:
                #
                DF = pd.DataFrame(returns_terms_spread, columns=ReturnTermNameList)
                DF.insert(0,'Date',DFDates['Date'].values)
                DF.to_csv(dump_directory + "Spread_of_returns_terms_" + str(factorsNamesList[k]) + ".csv", index=False)
                #
                DF = pd.DataFrame(returns_terms_spread_prob_scaled, columns=ReturnTermNameList)
                DF.insert(0,'Date',DFDates['Date'].values)
                #DF.to_csv(dump_directory + "Spread_of_return_terms_prob_scaled_" + str(factorsNamesList[k]) + ".csv", index=False)
        
        figAllW.savefig(dump_directory + 'WeightsSpreadCombined' + '.png', dpi=700)
        figAllRT.savefig(dump_directory + 'ReturnTermsSTDCombined' + '.png', dpi=700)
        
    elif key_start == Start.analyse_OOS:
        print("****** In analyse OOS ******")
        print("Loading all four Tau values and writing the results into Latex tables.")
        sample_list = ["T", "T_2", "2T_3"]

        for tau_index in np.arange(0,len(TauArray)):
            columns_names=[]
            multicolumn = ""
            table_data = None
            table_data_single = None
            
            for sample in sample_list:
                multicolumn = multicolumn + " & "
                if sample == sample_list[0]:
                    columns_names.append("EST")
                    multicolumn = multicolumn + " \multicolumn{1}{c} {$" + str(sample) + "$}"
                    dump_directory = homeDir + "conditionalTauDMN0F14M13s" + str(tau_index) + "/"
                else:
                    x = sample.split('_')            
                    multicolumn = multicolumn + " \multicolumn{2}{c} {$\\frac{" + str(x[0]) + "}{" + str(x[1]) + "}$} "
                    columns_names.append("EST")            
                    columns_names.append("OOS")
                    dump_directory = homeDir + "conditionalTau" + str(sample) + "NumbaDMN0F14M13s" + str(tau_index) + "/"

                file_name = dump_directory + dump_file_prefix + str(len(significantPredictors))+'_OOS.pkl'
                file_name = dump_directory + dump_file_prefix + str(len(significantPredictors))+'_OOS_new1.pkl'
                print("Loading file: %s" %(file_name))
                with open(file_name, 'rb') as file:
                    #[returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix, \
                    #    returns_IN, returns_square_IN, returns_interactions_IN, cumulative_probability, BMA] \
                    #    = pickle.load(file)

                    # New format _OOS.pkl from 10.9.2021
                    #[returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, \
                    #     returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, \
                    #    cumulative_probability, BMA] \
                    #    = pickle.load(file)
                    
                    # new1 format
                    [returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                        covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                        returns_IN, returns_square_IN, returns_interactions_IN, \
                        covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                        cumulative_probability, BMA] \
                        = pickle.load(file)

                    print(' ***** Renormalizing by cumulative_probability = %f' % cumulative_probability)
                    returns_OOS /= cumulative_probability
                    returns_square_OOS /= cumulative_probability
                    returns_interactions_OOS /= cumulative_probability
                    covariance_matrix_OOS /= cumulative_probability
                    covariance_matrix_no_ER_OOS /= cumulative_probability
                    returns_IN /= cumulative_probability
                    returns_square_IN /= cumulative_probability
                    returns_interactions_IN /= cumulative_probability
                    covariance_matrix_IN /= cumulative_probability
                    covariance_matrix_no_ER_IN /= cumulative_probability
                
                gamma = np.mean(np.concatenate((BMA.FEstimation[:,0],BMA.FTest[:,0]),axis=0))/ \
                            np.cov(np.concatenate((BMA.FEstimation[:,0],BMA.FTest[:,0]),axis=0), rowvar=False, bias=False)
                
                gamma = np.mean(BMA.FEstimation[:,0], axis=0)/ \
                            np.cov(BMA.FEstimation[:,0], rowvar=False, bias=False)
                #gamma *= 2
                print("gamma= %f" %(gamma))
                results = \
                    BMA.AnalyseInSampleAndOOSPortfolioReturns(returns_IN, returns_square_IN, returns_interactions_IN, \
                                                            covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                                                            returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                                                            covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                                                            gamma=gamma, only_top_model=False, dump_directory=dump_directory)

                if table_data is None:
                    table_data = results
                else:
                    table_data = np.concatenate((table_data, results), axis=1)
                
                del results
                # Adding the single top models performance.
                table_data_single_temp = None
                for top in np.arange(1,4):
                    file_name = dump_directory + dump_file_prefix + str(len(significantPredictors))+'_OOS_single_'+str(top)+'.pkl'
                    file_name = dump_directory + dump_file_prefix + str(len(significantPredictors))+'_OOS_new1_single_'+str(top)+'.pkl'
                    print("Loading file: %s" %(file_name))
                    with open(file_name, 'rb') as file:
                        # New format _OOS.pkl from 10.9.2021
                        #[returns_OOS , returns_square_OOS, returns_interactions_OOS, covariance_matrix_OOS, \
                        #    returns_IN, returns_square_IN, returns_interactions_IN, covariance_matrix_IN, \
                        #    cumulative_probability, BMA] \
                        #    = pickle.load(file)

                        # new1 format
                        [returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                            covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                            returns_IN, returns_square_IN, returns_interactions_IN, \
                            covariance_matrix_IN, covariance_matrix_no_ER_IN, \
                            cumulative_probability, BMA] \
                            = pickle.load(file)

                    results = \
                        BMA.AnalyseInSampleAndOOSPortfolioReturns(returns_IN, returns_square_IN, returns_interactions_IN, \
                                                            covariance_matrix_IN, covariance_matrix_no_ER_IN ,\
                                                            returns_OOS , returns_square_OOS, returns_interactions_OOS, \
                                                            covariance_matrix_OOS, covariance_matrix_no_ER_OOS, \
                                                            gamma=gamma, only_top_model=True, dump_directory=dump_directory, num_top=top)

                    single_range = np.arange(16,19)
                    if table_data_single_temp is None:
                        table_data_single_temp = results[single_range,:]
                    else:
                        table_data_single_temp = np.concatenate((table_data_single_temp, results[single_range,:]), axis=0)

                    del results

                if table_data_single is None:
                    table_data_single = table_data_single_temp
                else:
                    table_data_single = np.concatenate((table_data_single, table_data_single_temp), axis=1)

            total_table_data = np.zeros((28,5), dtype=np.float64) ; total_table_data.fill(np.NaN)
            # ===========    No constrains    ===========
            # CAPM
            total_table_data[0,:] = table_data[0,:]
            # FF3, FF6, AQR6
            total_table_data[1:4,:] = table_data[1:10:3,:]
            # Conditional Top 1-3
            total_table_data[4:7,:] = table_data_single[0:9:3,:]
            # BMA full, BMA diag, BMA static
            total_table_data[7,:] = table_data[10,:]
            total_table_data[8,:] = table_data[13,:]
            total_table_data[9,:] = table_data[16,:]
            # ===========     Regulation T  constrains    ===========
            # FF3, FF6, AQR6
            total_table_data[10:13,:] = table_data[3:10:3,:]
            # Top 1-3
            total_table_data[13:16,:] = table_data_single[2:9:3,:]
            # BMA full, BMA  diag, BMA static
            total_table_data[16,:] = table_data[12,:]
            total_table_data[17,:] = table_data[15,:]
            total_table_data[18,:] = table_data[18,:]
            # ===========         GMVP             ===========
            # FF3, FF6, AQR6
            total_table_data[19:22,:] = table_data[2:10:3,:]
            # Top 1-3
            total_table_data[22:25,:] = table_data_single[1:9:3,:]
            # BMA full, BMA  diag, BMA static
            total_table_data[25,:] = table_data[11,:]
            total_table_data[26,:] = table_data[14,:]
            total_table_data[27,:] = table_data[17,:]

            multicolumn = multicolumn + "\\\\"
            multicolumn_lines = "\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}"
            
            with open('InSampleAndOOSStatistics'+str(tau_index)+'.tex','w') as file:
                caption="The annulized Sharpe ratios of three benchmark models (CAPM, FF3, FF6 and AQR6), all factors unconditional model, \
                    all factors top 3 most probable single model, and all factors Bayesian model averaging (BMA). Followed by the last \
                    five models subjected to regulation T constrains. Namely, the portfolio weights sum of absolut values are  \
                    smaller equal to 2, $\\sum_{i=1}^{14} |w_i| \\leq 2 $. \
                    The prior multiple is $\\tau=" + str(TauArray[tau_index]) + "$. \
                    Columns labeled EST report the in sample Sharpe ratio corresponding to the full sample $(T)$, \
                    half $(\\frac{T}{2})$, and two-third $(\\frac{2T}{3})$ of the sample. Columns labeled with OOS report the out-of-sample Sharpe ratios. \
                    Where for the first four unconditional models the tangency portfolio weights are determined by the first and second moments from the estimation period. \
                    In the Bayesian model averaging setup, the models' probabilities and regression coefficients are determined from the estimation period. \
                    Information at time $t-1$ is used to estimate the returns at time $t$ and the models disagreement component in the covariance matrix, \
                    $\\Omega_t$ in Eq. \\eqref{eq:time_varying_variance}."

                latex_table = tex.LatexTable(file=file, sidewaystable=False)
                latex_table.writeLatexTableHeader(caption=caption, label="tab:InOutStatTau"+str(tau_index))

                rowsNames=["CAPM", "FF3", "FF6", "AQR6", \
                            "Top 1", "Top 2", "Top 3", 
                            "BMA full $\\Omega_t$", "BMA diagonal $\\Omega_t$", "BMA $V_t$ only ",\
                            "FF3 s.t. regulation T", "FF6 s.t. regulation T", "AQR6 s.t. regulation T",
                            "Top 1 s.t. regulation T", "Top 2 s.t. regulation T", "Top 3 s.t. regulation T", \
                            "BMA full $\\Omega_t$ s.t. regulation T", "BMA diagonal $\\Omega_t$ s.t. regulation T", "BMA $V_t$ only s.t. regulation T", \
                            "FF3 GMVP", "FF6 GMVP", "AQR6 GMVP", \
                            "Top 1 GMVP", "Top 2 GMVP", "Top 3 GMVP", \
                            "BMA full $\\Omega_t$ GMVP","BMA diagonal $\\Omega_t$ GMVP","BMA $V_t$ only GMVP"]

                latex_table.writeProfLatexTabular(data=total_table_data, rowsNames=rowsNames,
                                   columnsNames=columns_names,float_format="%.3f",rows_title='Model', \
                                    multicolumn=multicolumn, multicolumn_lines= multicolumn_lines)
                                                        
                latex_table.writeLatexTableEnd()

    elif key_start == Start.analyse_OOS_performance:
        print("****** In analyse OOS performance ******")
        # Analyse the OOS performance for the 2T_3 estimation period
        sample = "2T_3"
        tau_index = 1 
       
        models_list = ["CAPM", "FF3", "FF6", "AQR6", \
                        "conditional_top_1", "conditional_top_2", "conditional_top_3", \
                        "BMA_full", "BMA_diagonal", "BMA_static_cov", \
                        "CAPM_regulation_T","FF3_regulation_T", "FF6_regulation_T", "AQR6_regulation_T", \
                        "regulation_T_top_1", "regulation_T_top_2", "regulation_T_top_3", \
                        "BMA_full_regulation_T", "BMA_diagonal_regulation_T","BMA_static_cov_regulation_T",\
                        "FF3_GMVP", "FF6_GMVP", "AQR6_GMVP", \
                        "conditional_GMVP_top_1", "conditional_GMVP_top_2", "conditional_GMVP_top_3", \
                        "BMA_GMVP_full", "BMA_GMVP_diagonal", "BMA_GMVP_static_cov"]

        dump_directory = homeDir + "conditionalTau" + str(sample) + "NumbaDMN0F14M13s" + str(tau_index) + "/"
        returns = pd.DataFrame(F['Date'][1:].values, columns=["Date"])
        for model in models_list:
            if model == "CAPM_regulation_T":
                temp_model = pd.read_csv(dump_directory+"R_"+"CAPM"+".csv", header = None)
                temp_model *= 2
            else:
                temp_model = pd.read_csv(dump_directory+"R_"+model+".csv", header = None)

            returns[model]=temp_model.values

        # October 1987, the Russian default in August 1998, the bursting of the tech bubble in April 2000, 
        # the Quant crisis in August 2007, 
        # and the Bear Stearns bailout and Lehman bankruptcy during the recent financial crisis, i.e., March, September, and October 2008.
        crisis_period = [198710, 199808, 200004, 200708, 200803, 200809, 200810]
        
        start_index = 318
        crisis_indicies = []

        for i in np.arange(0,len(returns['Date'].values)):
            if i < start_index:
                continue
            else:
                if int(returns.loc[i]['Date']) in crisis_period:
                    crisis_indicies.append(i)
        
        crisis_indicies = np.array(crisis_indicies)

        performance = pd.DataFrame(models_list, columns=["Model Name"])
        performance['Mean'] = np.mean(returns.loc[start_index:][models_list].values,axis=0) 
        performance['Std.Dev.'] = np.std(returns.loc[start_index:][models_list].values,axis=0,ddof=1)
        performance['Sharpe Ratio'] = np.mean(returns.loc[start_index:][models_list].values,axis=0) \
            /np.std(returns.loc[start_index:][models_list].values,axis=0,ddof=1) * np.sqrt(12)
        performance['Skewness'] = scipy.stats.skew(returns.loc[start_index:][models_list].values,axis=0, bias=False)
        performance['Excess Kurtosis'] = scipy.stats.kurtosis(returns.loc[start_index:][models_list].values,axis=0, fisher=True, bias=False)

        cum_return = np.log(np.cumprod(1 + returns.loc[start_index:][models_list].values/100, axis=0))
        # Equal to
        #cum_return = np.cumsum(np.log(1 + returns.loc[start_index:][models_list].values/100), axis=0)
        maximum_drawdown = np.zeros((len(models_list),), dtype=np.float64)
        maximum_drawdown_start = np.zeros((len(models_list),), dtype=np.int64)
        maximum_drawdown_end = np.zeros((len(models_list),), dtype=np.int64)
        for m in np.arange(0, len(models_list)):
            for i in np.arange(0, len(cum_return)-1):
                for ii in np.arange(i + 1, len(cum_return)):
                    if cum_return[i,m] - cum_return[ii, m] > maximum_drawdown[m]:
                        maximum_drawdown[m] = cum_return[i, m]-cum_return[ii, m]
                        maximum_drawdown_start[m] = i
                        maximum_drawdown_end[m] = ii

            print("Model %s Maximum drawdown %f Maximum drawdown2 %f  starting %i until %i " %(models_list[m], np.exp(-maximum_drawdown[m]), \
                (1-np.exp(-maximum_drawdown[m]))*100, \
                returns.loc[start_index+maximum_drawdown_start[m]]['Date'], returns.loc[start_index+maximum_drawdown_end[m]]['Date']))


        performance['Maximum Drawdown'] = (1-np.exp(-maximum_drawdown))*100
        performance['Return in Crisis'] = np.mean(returns.loc[crisis_indicies][models_list].values,axis=0)
        
        # Write the performance into a tex table file.
        with open('OOSPerformance'+'_Tau_'+str(tau_index)+'.tex','w') as file:

            latex_table = tex.LatexTable(file = file, sidewaystable = False)
            label = "tab:OOSPerformance"
            caption = "Out-of-Sample model performance"
            midrule_at_lines = [4,11,14,21,24]

            latex_table.writeLatexTableHeader(caption=caption, label=label)

            # Join two lists.
            columnsNames = list(performance.columns)[1:]
            rowsNames = list(performance.loc[:,performance.columns[0]].values)

            latex_table.writeProfLatexTabular(data=performance.loc[:,columnsNames].values, rowsNames=rowsNames  ,
                        columnsNames=columnsNames, float_format="%.3f", midrule_at_lines=midrule_at_lines)

            latex_table.writeLatexTableEnd()

    elif key_start == Start.summary_statistics:
        print("****** In summary_statistics ******")
        
        # Test asset are the 14 factors, 14 factors equal weight portfolio, and the BMA in-sample, e.g. estimation period (T) with tau=1.5.
        KMax = len(factorsNames)
        TA = np.zeros((len(F[factorsNames[0]].values), KMax + 2),dtype=np.float64)
        
        dump_directory = homeDir + "conditionalTauDMN0F14M13s" + str(1) + "/"
        BMA_T_Tau_1_5 = pd.read_csv(dump_directory+"R_BMA_full"+".csv", header = None)
        BMA_T_Tau_1_5 = BMA_T_Tau_1_5.values
        
        equal_weght_portfolio = np.zeros((len(F[factorsNames[0]].values),), dtype=np.float64)
        
        for i in np.arange(0, KMax):
            equal_weght_portfolio += F[factorsNames[i]].values
        
        equal_weght_portfolio /= KMax
                
        TA = np.concatenate((F[factorsNames].values, equal_weght_portfolio.reshape(-1,1)), axis=1)
        assert TA.shape == (len(F[factorsNames[0]].values), KMax + 1)
        NTA = TA.shape[1]

        # The length of BMA_T_Tau_1_5 is smaller by one than the factors length.
        TAMean = np.mean(TA, axis=0); TAMean = np.concatenate((TAMean,np.mean(BMA_T_Tau_1_5).reshape(-1,)),axis=0)
        TAStd = np.std(TA, axis=0); TAStd = np.concatenate((TAStd,np.std(BMA_T_Tau_1_5).reshape(-1,)),axis=0)
        TAMedian = np.median(TA, axis=0); TAMedian = np.concatenate((TAMedian,np.median(BMA_T_Tau_1_5).reshape(-1,)),axis=0)
        TASR = TAMean/TAStd*np.sqrt(12)
        
        AlphaCAPMFF3FF6AQR6  = np.zeros((NTA+1,4), dtype=np.float64); AlphaCAPMFF3FF6AQR6.fill(np.NaN)
        AlphatCAPMFF3FF6AQR6 = np.zeros((NTA+1,4), dtype=np.float64); AlphatCAPMFF3FF6AQR6.fill(np.NaN)
        for i in np.arange(0,NTA+1):
            if i < NTA:
                y = TA[:,i]
            else:
                y = BMA_T_Tau_1_5
            # CAPM alpha
            model_indecies = np.array([0])
            if not i in model_indecies:
                x = F[factorsNames[model_indecies]].values.reshape(-1,1)
                if i==NTA:
                    x = x[1:,:]
                x = np.concatenate((np.ones((len(x),1), dtype=np.float64), x), axis=1)
                mod = sm.OLS(y, x)
                res = mod.fit()
                AlphaCAPMFF3FF6AQR6[i,0] = res.params[0]
                AlphatCAPMFF3FF6AQR6[i,0] = res.tvalues[0]
            # FF3 alpha
            model_indecies = np.array([0,1,2])
            if not i in model_indecies:
                x = F[factorsNames[model_indecies]].values
                if i==NTA:
                    x = x[1:,:]
                x = np.concatenate((np.ones((len(x),1), dtype=np.float64), x), axis=1)
                mod = sm.OLS(y, x)
                res = mod.fit()
                AlphaCAPMFF3FF6AQR6[i,1] = res.params[0]
                AlphatCAPMFF3FF6AQR6[i,1] = res.tvalues[0]
            # FF6 alpha
            model_indecies = np.array([0,1,2,3,4,5])
            if not i in model_indecies:
                x = F[factorsNames[model_indecies]].values
                if i==NTA:
                    x = x[1:,:]
                x = np.concatenate((np.ones((len(x),1), dtype=np.float64), x), axis=1)
                mod = sm.OLS(y, x)
                res = mod.fit()
                AlphaCAPMFF3FF6AQR6[i,2] = res.params[0]
                AlphatCAPMFF3FF6AQR6[i,2] = res.tvalues[0]
            # AQR6 alpha
            model_indecies = np.array([0,1,2,5,8,9])
            if not i in model_indecies:
                x = F[factorsNames[model_indecies]].values
                if i==NTA:
                    x = x[1:,:]
                x = np.concatenate((np.ones((len(x),1), dtype=np.float64), x), axis=1)
                mod = sm.OLS(y, x)
                res = mod.fit()
                AlphaCAPMFF3FF6AQR6[i,3] = res.params[0]
                AlphatCAPMFF3FF6AQR6[i,3] = res.tvalues[0]

        ZMean = np.mean(Z[predictorsNames].values, axis=0)
        ZStd = np.std(Z[predictorsNames].values, axis=0)
        ZMedian = np.median(Z[predictorsNames].values, axis=0)
        # Calculating the AR(1) autoregressive parameter
        MMax = len(predictorsNames)
        phi = np.zeros((2, MMax), dtype=np.float64)
        for i in np.arange(0, MMax):
            ZZ = Z[predictorsNames[i]].values.reshape(-1,1) 
            Y = ZZ[1:]
            X = ZZ[0:-1]
            X = np.concatenate((np.ones((len(X),1),dtype=np.float64), X), axis=1)
            beta = LA.pinv(np.transpose(X) @ X) @ np.transpose(X) @ Y
            phi[:,i] = beta.reshape(-1)

        with open('SummaryStatistics'+'.tex','w') as file:
                midrule_at_line = 4 + 2 + 2 + 2 + 2
                table_data = np.zeros((midrule_at_line + NTA + 1, NTA + 1), dtype=np.float64)
                table_data[0:NTA, 0:NTA] = np.corrcoef(TA, rowvar=False, bias=False)
                temp = np.concatenate((TA[1:,:], BMA_T_Tau_1_5), axis=1)
                temp_corrcoef = np.corrcoef(temp, rowvar=False, bias=False)
                table_data[0:NTA+1, NTA] = temp_corrcoef[:, NTA]
                for i in np.arange(1, NTA+1):
                    table_data[i, 0:i]=np.NaN
                
                table_data[midrule_at_line:,:] = table_data[0: NTA + 1,0: NTA + 1]
                table_data[0,:] = TAMean ; table_data[1,:] = TAMedian ; table_data[2,:] = TAStd; table_data[3,:] = TASR
                for i in np.arange(0,4):
                    table_data[4+2*i,:] = AlphaCAPMFF3FF6AQR6[:,i] ; table_data[4+2*i+1,:] = AlphatCAPMFF3FF6AQR6[:,i]

                latex_table = tex.LatexTable(file=file, sidewaystable=True)
                label = "tab:SummaryStatistics"
                caption="Summary characteristics of the factors, and predictors."

                latex_table.writeLatexTableHeader(caption=caption, label=label)

                label = "tab:SummaryStatisticsFactors"
                caption="Summary statistics of the monthly factors returns and the sample correlations of the \
                        factor returns in our sample during the period from June 1977 to December 2016.  \
                        The factors are described in detail in panel A of table \\ref{tab:factors_variables}."
                rowsNames=['Mean', 'Median', 'Std', 'SR', \
                    'CAPM $\\alpha$', 'CAPM $\\alpha$ t-value', 'FF3 $\\alpha$', 'FF3 $\\alpha$ t-value', \
                    'FF6 $\\alpha$', 'FF6 $\\alpha$ t-value', 'AQR6 $\\alpha$', 'AQR6 $\\alpha$ t-value']; 
                # Join two lists.
                columnsNames = list(factorsNames) + ['14-equal', 'BMA']
                rowsNames = rowsNames + columnsNames

                latex_table.writeLatexSubTableHeader(caption=caption, label=label)
                latex_table.writeProfLatexTabular(data=table_data, rowsNames=rowsNames  ,
                                   columnsNames=columnsNames, float_format="%.3f", midrule_at_lines=[midrule_at_line])
                latex_table.writeLatexSubTableEnd()
                latex_table.writeLatexTableAddSpace()

                del table_data
                label = "tab:SummaryStatisticsPredictors"
                caption="Summary statistics of the predictive variables in our sample during the period from \
                        June 1977 to December 2016. The construction of these variables is \
                        explained in more detail in panel B of table \\ref{tab:factors_variables}. The variable Ar(1) corresponds to the maximum \
                        likelihood estimator of the autoregressive parameter $\\hat{\\varphi}$ of the regression \
                            $z_{i,t} = c + \\varphi z_{i,t-1} + \\epsilon_t$. "
                
                rowsNames=['Mean', 'Median', 'Std', 'AR(1)']

                table_data = np.zeros((4, MMax), dtype=np.float64) ; table_data.fill(np.NaN)
                table_data[0,:] = ZMean ; table_data[1,:] = ZMedian ; table_data[2,:] = ZStd
                table_data[3,:] = phi[1,:]

                latex_table.writeLatexSubTableHeader(caption=caption, label=label)
                latex_table.writeProfLatexTabular(data=table_data, rowsNames=rowsNames, \
                                                   columnsNames=list(predictorsNames),float_format="%.3f")
                latex_table.writeLatexSubTableEnd()
                latex_table.writeLatexTableEnd()

    elif key_start == Start.variance_matrix:
        print("****** In variance_matrix ******")
        # Modify factorNames to comply with the paper
        factorsNamesList = list(factorsNames)
        factorsNamesList[0]='MKT' ; factorsNamesList[5]='MOM'; factorsNamesList[13]='ICR'

        samples = ["T","T_2","2T_3"]
        tau_index = 1
        fig_size=(6.4, 3.6)
        font_size='xx-small'
       
        DF = pd.DataFrame(F['Date'][1:].values, columns=["Date"])
        DF['Date Plot'] = ((DF['Date'].values)%100)/12+np.floor(DF['Date'].values/100)
        DF["MKT"] = F['Mkt-RF'][1:].values

        T_2_index_start = 246
        T2_3_index_start = 318
        variance_matrix_dict = {}
        
        figAll, axsAll = plt.subplots(7,2,sharex=True, sharey=True)
        for sample in samples:
            if sample == "T":
                dump_directory = homeDir + "conditionalTau" + "DMN0F14M13s" + str(tau_index) + "/"
            else:                
                dump_directory = homeDir + "conditionalTau" + str(sample) + "NumbaDMN0F14M13s" + str(tau_index) + "/"
                if sample == "T_2":
                    start_OOS_index = T_2_index_start
                elif sample == "2T_3":
                    start_OOS_index = T2_3_index_start

            temp_model = pd.read_csv(dump_directory+"variance_matrix_ratio_full.csv", header = None)
            # VMR variance matrix ratio |V_t|/|V_t+\Omega_t|
            DF["VMR " + sample] = temp_model.values
            #
            temp_model = pd.read_csv(dump_directory+"R_BMA_full.csv", header = None)
            DF["R BMA " + sample] = temp_model.values
            #
            temp_model = pd.read_csv(dump_directory+"R_BMA_full_leveraged.csv", header = None)
            DF["R BMA leveraged " + sample] = temp_model.values
            #
            temp_model = pd.read_csv(dump_directory+"R_BMA_full_regulation_T.csv", header = None)
            DF["R BMA reg T " + sample] = temp_model.values
            #
            temp_model = pd.read_csv(dump_directory+"R_BMA_GMVP_full.csv", header = None)
            DF["R BMA GMVP " + sample] = temp_model.values

            temp_model = pd.read_csv(dump_directory+"variance_matrix_ratio_contribution_full.csv", header = 0)
            portfolio_weights = pd.read_csv(dump_directory+"w_BMA_diagonal.csv", header = 0)
            relative_contribution = np.zeros((len(DF), len(factorsNames)), dtype=np.float64)
            relative_contribution_prod = np.zeros((len(DF), ), dtype=np.float64)
            relative_contribution_sum = np.zeros((len(DF), ), dtype=np.float64)
            for t in np.arange(0, len(DF)):
                relative_contribution_prod[t] = np.prod(temp_model.loc[t,:].values)
                #relative_contribution[t,:]=(temp_model.loc[t,:].values-1)/(1/DF.loc[t,"VMR " + sample]-1)*100
                relative_contribution[t,:] = np.log(temp_model.loc[t,:].values)/np.log(1/DF.loc[t,"VMR " + sample])
                relative_contribution_sum[t] = np.sum(relative_contribution[t,:])
                relative_contribution[t,:] = relative_contribution[t,:]/relative_contribution_sum[t]*100
            
            relaviteMean = np.mean(np.abs(relative_contribution_prod[:]-1/DF.loc[:,"VMR " + sample].values)/1/DF.loc[:,"VMR " + sample].values)
            relaviteMax = np.max(np.abs(relative_contribution_prod[:]-1/DF.loc[:,"VMR " + sample].values)/1/DF.loc[:,"VMR " + sample].values)
            print("Approximation mean %f max %f" %(relaviteMean, relaviteMax))

            fig, axs = plt.subplots()
            fig.set_size_inches(fig_size[0],fig_size[1])
            axs.set_xlabel('Year')
            axs.set_ylabel('Relative Contribution')
            KMax = len(factorsNames)
            #factorsNames[0]='MKT'; 
            colorKeys=list(mcd.CSS4_COLORS.keys())
            for k in np.arange(0, KMax):
                cn = 'C'+str(np.mod(k,10))
                if k<=9:
                    axs.plot(DF['Date Plot'].values, relative_contribution[:,k], label=factorsNamesList[k], linewidth=0.5, linestyle='solid', color=cn)
                else:
                    axs.plot(DF['Date Plot'].values, relative_contribution[:,k], label=factorsNamesList[k], linewidth=0.5, linestyle='dotted', color=cn)

            axs.legend(loc='upper left',fontsize=font_size)
            fig.savefig('Relative_Contribution_Tau_1_5'+str(sample)+'.png', dpi=500)

            # Regressions
            for k in np.arange(0, KMax):
                x = np.ones((len(portfolio_weights), 1), dtype=np.float64)
                x = np.concatenate((x, 1/DF.loc[:,"VMR " + sample].values.reshape(-1,1)), axis=1)
                x = np.concatenate((x, relative_contribution[:,k].reshape(-1,1)), axis=1)
                y = portfolio_weights.loc[:,factorsNames[k]].values
                mod = sm.OLS(y,x)

                res = mod.fit()
                print('Factor %s - Relative contribution %f Regression parameters - ' \
                    %(factorsNames[k],np.mean(relative_contribution[:,k],axis=0)))
                print(np.concatenate((res.params,res.tvalues),axis=0))

                fig, ax1 = plt.subplots()

                fig.set_size_inches(fig_size[0],fig_size[1])
                color = 'tab:blue'
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Relative Contribution', color=color)
                ax1.plot(DF['Date Plot'].values, relative_contribution[:,k], color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:red'
                ax2.set_ylabel('Weight', color=color)  # we already handled the x-label with ax1
                ax2.plot(DF['Date Plot'].values, portfolio_weights.loc[:,factorsNames[k]].values, color=color)
                ax2.tick_params(axis = 'y', labelcolor = color)

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fig.savefig('WeightVsRelCont_' + sample + '_' + str(factorsNames[k]) + '.png', dpi = 500)
                
                line_width = 1.1
                if sample == samples[0]:
                    if k==1:
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].plot(DF['Date Plot'].values, relative_contribution[:,k], color='C1', linewidth=line_width, label="$(T)$")
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1., 1.15))
                    else:
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].plot(DF['Date Plot'].values, relative_contribution[:,k], color='C1', linewidth=line_width)
                elif sample == samples[1]:
                    if k==1:
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].plot(DF['Date Plot'].values, relative_contribution[:,k], color='C2', linewidth=line_width, label="$(\\frac{T}{2})$")
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1., 1.15))
                    else:
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].plot(DF['Date Plot'].values, relative_contribution[:,k], color='C2', linewidth=line_width)
                elif sample == samples[2]:
                    if k==1:
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].plot(DF['Date Plot'].values, relative_contribution[:,k], color='C3', linewidth=line_width, label="$(\\frac{2T}{3})$")
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1., 1.15))
                    else:
                        axsAll[int(np.floor(k/2)), np.mod(k,2)].plot(DF['Date Plot'].values, relative_contribution[:,k], color='C3', linewidth=line_width)

                    axsAll[int(np.floor(k/2)), np.mod(k,2)].axvline(x=DF['Date Plot'].values[246-1], color='b', linestyle='--')
                    axsAll[int(np.floor(k/2)), np.mod(k,2)].axvline(x=DF['Date Plot'].values[318-1], color='b', linestyle='--')
                    axsAll[int(np.floor(k/2)), np.mod(k,2)].set_title(str(factorsNamesList[k]), fontsize = font_size) # add pad=1.
                    #axsAll[int(np.floor(k/2)), np.mod(k,2)].legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1., 1.15))
                    #axsAll[int(np.floor(k/2)), np.mod(k,2)].
                if k>=12: 
                    axsAll[int(np.floor(k/2)), np.mod(k,2)].set_xlabel('Year')
                if k>=6 and k<=7:
                    axsAll[int(np.floor(k/2)), np.mod(k,2)].set_ylabel('Relative Contribution')
                # figAll.tight_layout()  # Not working well.
                # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
                figAll.subplots_adjust(left  = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.6)
                figAll.savefig('RelativeContributionCombined' + '.png', dpi=700)


            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('time')
            ax1.set_ylabel('sum rel cont', color=color)
            ax1.plot(np.arange(0, len(DF)), relative_contribution_sum, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('VMR', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.arange(0, len(DF)), 1/DF.loc[:,"VMR " + sample].values, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            fig.savefig('VMRandSumRelCont' +sample+ '.png', dpi=500)
            plt.close(fig)

            mod = sm.OLS(1/DF.loc[:,"VMR " + sample].values, \
                np.concatenate((np.ones((len(relative_contribution_prod), 1), dtype=np.float64), relative_contribution_prod.reshape(-1,1)), axis=1))
            mod = sm.OLS(1/DF.loc[:,"VMR " + sample].values, relative_contribution_prod)

            res = mod.fit()
            print('regression parameters - ') ; print(res.params)

            if sample == "T":                
                variance_matrix_dict["VMR Avg "+sample+" EST"] = np.mean(1/DF["VMR " + sample].values)
                variance_matrix_dict["VMR Max "+sample+" EST"] = np.max(1/DF["VMR " + sample].values)
                variance_matrix_dict["VMR Per 99 "+sample+" EST"] = np.percentile(1/DF["VMR " + sample].values,99)
                variance_matrix_dict["VMR Per 95 "+sample+" EST"] = np.percentile(1/DF["VMR " + sample].values,95)
                variance_matrix_dict["Rel Cont "+sample+" EST"] = np.mean(relative_contribution,axis=0)
                variance_matrix_dict["Rel Cont Max "+sample+" EST"] = np.max(relative_contribution,axis=0)
            else:
                variance_matrix_dict["VMR Avg "+sample+" EST"] = np.mean(1/DF["VMR " + sample].values[:start_OOS_index])
                variance_matrix_dict["VMR Max "+sample+" EST"] = np.max(1/DF["VMR " + sample].values[:start_OOS_index])
                variance_matrix_dict["VMR Per 99 "+sample+" EST"] = np.percentile(1/DF["VMR " + sample].values[:start_OOS_index],99)
                variance_matrix_dict["VMR Per 95 "+sample+" EST"] = np.percentile(1/DF["VMR " + sample].values[:start_OOS_index],95)
                variance_matrix_dict["Rel Cont "+sample+" EST"] = np.mean(relative_contribution[:start_OOS_index,:],axis=0)
                variance_matrix_dict["Rel Cont Max "+sample+" EST"] = np.max(relative_contribution[:start_OOS_index,:],axis=0)
                variance_matrix_dict["VMR Avg "+sample+" OOS"] = np.mean(1/DF["VMR " + sample].values[start_OOS_index:])
                variance_matrix_dict["VMR Max "+sample+" OOS"] = np.max(1/DF["VMR " + sample].values[start_OOS_index:])
                variance_matrix_dict["VMR Per 99 "+sample+" OOS"] = np.percentile(1/DF["VMR " + sample].values[start_OOS_index:],99)
                variance_matrix_dict["VMR Per 95 "+sample+" OOS"] = np.percentile(1/DF["VMR " + sample].values[start_OOS_index:],95)
                variance_matrix_dict["Rel Cont "+sample+" OOS"] = np.mean(relative_contribution[start_OOS_index:,:],axis=0)
                variance_matrix_dict["Rel Cont Max "+sample+" OOS"] = np.max(relative_contribution[start_OOS_index:,:],axis=0)       

        with open('VarianceMatrix'+'.tex','w') as file:
            latex_table = tex.LatexTable(file=file, sidewaystable=False)
            label = "tab:VarianceMatrix"
            caption="BMA Covariance Matrix"

            latex_table.writeLatexTableHeader(caption=caption, label=label)
            caption = 'Full Sample $\\frac{|V_t+\\Omega_t|}{|V_t|}$'
            label = "tab:EntropyFS"
            latex_table.writeLatexSubTableHeader(caption = caption,label = label)
            rows_names=["Average $\\frac{|V_t+\\Omega_t|}{|V_t|}$","Max $\\frac{|V_t+\\Omega_t|}{|V_t|}$"]
            columns_names = ["$(T)$","$(\\frac{T}{2})$","$(\\frac{2T}{3})$"]
            table_data = np.array([[np.mean(1/DF["VMR " + "T"].values), np.mean(1/DF["VMR " + "T_2"].values),\
                np.mean(1/DF["VMR " + "2T_3"].values)],\
                    [np.max(1/DF["VMR " + "T"].values), np.max(1/DF["VMR " + "T_2"].values),\
                np.max(1/DF["VMR " + "2T_3"].values)]])

            latex_table.writeProfLatexTabular(data=table_data, rowsNames=rows_names,
                    columnsNames=columns_names,float_format="%.3f")
            latex_table.writeLatexSubTableEnd()
            latex_table.writeLatexTableAddSpace()
            caption = 'Estimation and Out-of-Sample $\\frac{|V_t+\\Omega_t|}{|V_t|}$'
            label = "tab:EntropyESTOOS"
            latex_table.writeLatexSubTableHeader(caption = caption,label = label)
            columns_names = ["EST","EST","OOS","EST","OOS"]
            table_data = np.array([[variance_matrix_dict["VMR Avg T EST"], \
                variance_matrix_dict["VMR Avg T_2 EST"],variance_matrix_dict["VMR Avg T_2 OOS"],\
                variance_matrix_dict["VMR Avg 2T_3 EST"],variance_matrix_dict["VMR Avg 2T_3 OOS"]],
                [variance_matrix_dict["VMR Max T EST"], \
                variance_matrix_dict["VMR Max T_2 EST"],variance_matrix_dict["VMR Max T_2 OOS"],\
                variance_matrix_dict["VMR Max 2T_3 EST"],variance_matrix_dict["VMR Max 2T_3 OOS"]],\
                [variance_matrix_dict["VMR Per 99 T EST"], \
                variance_matrix_dict["VMR Per 99 T_2 EST"],variance_matrix_dict["VMR Per 99 T_2 OOS"],\
                variance_matrix_dict["VMR Per 99 2T_3 EST"],variance_matrix_dict["VMR Per 99 2T_3 OOS"]],\
                [variance_matrix_dict["VMR Per 95 T EST"], \
                variance_matrix_dict["VMR Per 95 T_2 EST"],variance_matrix_dict["VMR Per 95 T_2 OOS"],\
                variance_matrix_dict["VMR Per 95 2T_3 EST"],variance_matrix_dict["VMR Per 95 2T_3 OOS"]]])
            
            rows_names = ["Average","Max","99th Percentile","95th Percentile"]
            latex_table.writeProfLatexTabular(data=table_data, rowsNames=rows_names,
                    columnsNames=columns_names,float_format="%.3f")
            latex_table.writeLatexSubTableEnd()
            latex_table.writeLatexTableAddSpace()
            caption = 'Average Factors\'s Relativ Contribution to Entropy'
            label = "tab:EntropyRelCont"
            latex_table.writeLatexSubTableHeader(caption = caption,label = label)
            columns_names = ["$(T)$ EST","$(\\frac{T}{2})$ EST","$(\\frac{T}{2})$ OOS", \
                "$(\\frac{2T}{3})$ EST","$(\\frac{2T}{3})$ OOS"]
            rows_names = factorsNamesList
            table_data = variance_matrix_dict["Rel Cont T EST"].reshape(1,-1)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont T_2 EST"].reshape(1,-1)), axis=0)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont T_2 OOS"].reshape(1,-1)), axis=0)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont 2T_3 EST"].reshape(1,-1)), axis=0)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont 2T_3 OOS"].reshape(1,-1)), axis=0)
            latex_table.writeProfLatexTabular(data=np.transpose(table_data), rowsNames=rows_names,
                    columnsNames=columns_names,float_format="%.3f")
            
            latex_table.writeLatexSubTableEnd()

            latex_table.writeLatexTableAddSpace()
            caption = 'Maximum Factors\'s Relativ Contribution to Entropy'
            label = "tab:EntropyRelContMax"
            latex_table.writeLatexSubTableHeader(caption = caption,label = label)
            columns_names = ["$(T)$ EST","$(\\frac{T}{2})$ EST","$(\\frac{T}{2})$ OOS", \
                "$(\\frac{2T}{3})$ EST","$(\\frac{2T}{3})$ OOS"]
            rows_names = factorsNamesList
            table_data = variance_matrix_dict["Rel Cont Max T EST"].reshape(1,-1)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont Max T_2 EST"].reshape(1,-1)), axis=0)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont Max T_2 OOS"].reshape(1,-1)), axis=0)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont Max 2T_3 EST"].reshape(1,-1)), axis=0)
            table_data = np.concatenate((table_data, variance_matrix_dict["Rel Cont Max 2T_3 OOS"].reshape(1,-1)), axis=0)
            latex_table.writeProfLatexTabular(data=np.transpose(table_data), rowsNames=rows_names,
                    columnsNames=columns_names,float_format="%.3f")
            
            latex_table.writeLatexSubTableEnd()

            latex_table.writeLatexTableEnd()

        fig, axs = plt.subplots()
        fig.set_size_inches(fig_size[0],fig_size[1])
        axs.set_xlabel('Year')
        axs.set_ylabel('$\\frac{|V_t+\Omega_t|}{|V_t|}$')
        
        axs.plot(DF['Date Plot'].values, 1/DF['VMR T'].values, 'C1', label='BMA (T)')
        axs.plot(DF['Date Plot'].values, 1/DF['VMR T_2'].values, 'C2', label='BMA $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values, 1/DF['VMR 2T_3'].values, 'C3', label='BMA $(\\frac{2T}{3})$')
        axs.axvline(x=DF['Date Plot'].values[246-1], color='b', linestyle='--')
        axs.axvline(x=DF['Date Plot'].values[318-1], color='b', linestyle='--')
        axs.legend(loc='upper left',fontsize=font_size)

        fig.savefig('variance_matrix_ratio_Tau_1_5.png', dpi=500)
        plt.close(fig)

        print("avg ratio: T %f, T/2 %f, 2T/3 %f" \
                %(np.mean(1/DF['VMR T'].values),np.mean(1/DF['VMR T_2'].values), \
                   np.mean(1/DF['VMR 2T_3'].values) ))

        fig, axs = plt.subplots()
        fig.set_size_inches(fig_size[0],fig_size[1])
        axs.set_yscale('log')

        axs.set_xlabel('Year')       
        axs.set_ylabel('Cumulative Excess Return')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['R BMA T'].values/100), 'C1', label='BMA $(T)$')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['R BMA T_2'].values/100),'C2', label='BMA $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['R BMA 2T_3'].values/100),'C3', label='BMA $(\\frac{2T}{3})$')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['MKT'].values/100), 'k-', label='MKT')
        axs.legend(loc='upper left',fontsize=font_size)
        axs.axvline(x=DF['Date Plot'].values[246-1], color='b', linestyle='--')
        axs.axvline(x=DF['Date Plot'].values[318-1], color='b', linestyle='--')
        axs.yaxis.set_major_formatter(ScalarFormatter())

        fig.savefig('compound_return_Tau_1_5.png', dpi=500)
        plt.close(fig)
    
        # Laveraged cumulative Excess Return where the leverage is set so that the in-sample volatility 
        # of the BMA is equal to the MKT volatilitydatetime A combination of a date and a time. Attributes: ()
        fig, axs = plt.subplots()
        fig.set_size_inches(fig_size[0],fig_size[1])
        axs.set_yscale('log')

        axs.set_xlabel('Year')       
        axs.set_ylabel('Cumulative Excess Return')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['R BMA leveraged T'].values/100), 'C1', label='BMA $(T)$')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['R BMA leveraged T_2'].values/100),'C2', label='BMA $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['R BMA leveraged 2T_3'].values/100),'C3', label='BMA $(\\frac{2T}{3})$')
        axs.plot(DF['Date Plot'].values, np.cumprod(1+DF['MKT'].values/100), 'k-', label='MKT')
        axs.legend(loc='upper left',fontsize=font_size)
        axs.axvline(x=DF['Date Plot'].values[246-1], color='b', linestyle='--')
        axs.axvline(x=DF['Date Plot'].values[318-1], color='b', linestyle='--')
        #axs.yaxis.set_major_formatter(ScalarFormatter())

        fig.savefig('leveraged_compound_return_Tau_1_5.png', dpi=500)
        plt.close(fig)
        #
        fig, axs = plt.subplots()
        fig.set_size_inches(fig_size[0],fig_size[1])
        axs.set_yscale('log')
        axs.set_xlabel('Year')
        axs.set_ylabel('Cumulative Excess Return')

        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['R BMA T_2'].values[246:]/100),'g-', label='BMA $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['R BMA reg T T_2'].values[246:]/100),'g--', label='BMA $(\\frac{T}{2})$ reg. T')
        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['R BMA GMVP T_2'].values[246:]/100),'g:', label='BMA $(\\frac{T}{2})$ GMVP')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['R BMA 2T_3'].values[318:]/100),'r-', label='BMA $(\\frac{2T}{3})$')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['R BMA reg T 2T_3'].values[318:]/100),'r--', label='BMA $(\\frac{2T}{3})$ reg. T')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['R BMA GMVP 2T_3'].values[318:]/100),'r:', label='BMA $(\\frac{2T}{3})$ GMVP')
        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['MKT'].values[246:]/100),'k--', label='MKT $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['MKT'].values[318:]/100),'k-', label='MKT $(\\frac{2T}{3})$')

        axs.legend(fontsize=font_size)

        axs.yaxis.set_major_formatter(ScalarFormatter())
        axs.set_xticks(np.arange(1997,2021,4))

        fig.savefig('compound_return_Tau_1_5_OOS.png', dpi=500)
        plt.close(fig)
        
        # Laveraged cumulative Excess Return where the leverage is set so that the in-sample volatility 
        # of the BMA is equal to the MKT volatilitydatetime A combination of a date and a time. Attributes: ()
        print('Leverage ratio for (T) %f (T/2) %f (2T/3) %f ' %(DF['R BMA leveraged T'].values[0]/DF['R BMA T'].values[0],
            DF['R BMA leveraged T_2'].values[0]/DF['R BMA T_2'].values[0], DF['R BMA leveraged 2T_3'].values[0]/DF['R BMA 2T_3'].values[0] ))

        fig, axs = plt.subplots()
        fig.set_size_inches(fig_size[0],fig_size[1])
        axs.set_yscale('log')
        axs.set_xlabel('Year')
        axs.set_ylabel('Cumulative Excess Return')

        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['R BMA leveraged T_2'].values[246:]/100),'g-', label='BMA $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['R BMA reg T T_2'].values[246:]/100),'g--', label='BMA $(\\frac{T}{2})$ reg. T')
        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['R BMA GMVP T_2'].values[246:]/100),'g:', label='BMA $(\\frac{T}{2})$ GMVP')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['R BMA leveraged 2T_3'].values[318:]/100),'r-', label='BMA $(\\frac{2T}{3})$')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['R BMA reg T 2T_3'].values[318:]/100),'r--', label='BMA $(\\frac{2T}{3})$ reg. T')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['R BMA GMVP 2T_3'].values[318:]/100),'r:', label='BMA $(\\frac{2T}{3})$ GMVP')
        axs.plot(DF['Date Plot'].values[246:], np.cumprod(1+DF['MKT'].values[246:]/100),'k--', label='MKT $(\\frac{T}{2})$')
        axs.plot(DF['Date Plot'].values[318:], np.cumprod(1+DF['MKT'].values[318:]/100),'k-', label='MKT $(\\frac{2T}{3})$')

        axs.legend(fontsize=font_size)
        axs.yaxis.set_major_formatter(ScalarFormatter())
        axs.set_xticks(np.arange(1997,2021,4))

        fig.savefig('leveraged_compound_return_Tau_1_5_OOS.png', dpi=500)
        plt.close(fig)

    elif key_start == Start.factors_variance:
        print("****** In factors_variance ******")
        # Modify factorNames to comply with the paper
        factorsNamesList = list(factorsNames)
        factorsNamesList[0]='MKT' ; factorsNamesList[5]='MOM'; factorsNamesList[13]='ICR'

        with open('FactorsVariance'+'.tex','w') as file:

            samples = ["T","T_2","2T_3"]
            # sample = "2T_3"

            tau_index = 1
            columns_names = []
            for sample in samples:
                if sample == "T":
                    dump_directory = homeDir + "conditionalTau" + "DMN0F14M13s" + str(tau_index) + "/"
                else:                
                    dump_directory = homeDir + "conditionalTau" + str(sample) + "NumbaDMN0F14M13s" + str(tau_index) + "/"

#               dump_directory = homeDir + "conditionalTau" + str(sample) + "NumbaDMN0F14M13s" + str(tau_index) + "/"
                print(dump_directory)
            
                cov_matrix_full_TS_avg_in_sample = pd.read_csv(dump_directory+"cov_matrix_full_TS_avg_in_sample.csv", header=0).values                
                omega_TS_avg_in_sample = pd.read_csv(dump_directory+"omega_TS_avg_in_sample.csv", header=0).values
                cov_matrix_full_obs_in_sample = pd.read_csv(dump_directory+"cov_matrix_full_obs_in_sample.csv", header=0).values                
                if len(columns_names) == 0:
                    table_data = np.diag(cov_matrix_full_TS_avg_in_sample).reshape(-1,1)
                else:
                    table_data = np.concatenate((table_data,np.diag(cov_matrix_full_TS_avg_in_sample).reshape(-1,1)),axis=1)

                table_data = np.concatenate((table_data, np.diag(omega_TS_avg_in_sample).reshape(-1,1)), axis=1)
                table_data = np.concatenate((table_data, np.diag(cov_matrix_full_obs_in_sample).reshape(-1,1)), axis=1)

                columns_names += ["$\\overline{V_t+\\Omega_t} $ IN "+str(sample),"$\\overline{\\Omega_t}$ IN "+str(sample), " OBS IN "+str(sample)]

                if sample != "T":
                    cov_matrix_full_TS_avg_OOS = pd.read_csv(dump_directory+"cov_matrix_full_TS_avg_OOS.csv", header=0).values
                    omega_TS_avg_OOS = pd.read_csv(dump_directory+"omega_TS_avg_OOS.csv", header=0).values
                    cov_matrix_full_obs_OOS = pd.read_csv(dump_directory+"cov_matrix_full_obs_OOS.csv", header=0).values
                
                    table_data = np.concatenate((table_data, np.diag(cov_matrix_full_TS_avg_OOS).reshape(-1,1)), axis=1)
                    table_data = np.concatenate((table_data, np.diag(omega_TS_avg_OOS).reshape(-1,1)), axis=1)
                    table_data = np.concatenate((table_data, np.diag(cov_matrix_full_obs_OOS).reshape(-1,1)), axis=1)
                    columns_names += ["$\\overline{V_t+\\Omega_t} $ OOS "+str(sample),"$\\overline{\\Omega_t}$ OOS "+str(sample), " OBS OOS "+str(sample)]         
            
            latex_table = tex.LatexTable(file=file, sidewaystable=True)
            label = "tab:FactorsVariance"
            caption = "Estimated Vs. Observed Foctors Variance"
            latex_table.writeLatexTableHeader(caption=caption, label=label)
            rows_names = factorsNamesList

            latex_table.writeProfLatexTabular(data=table_data, rowsNames=rows_names,
                    columnsNames=columns_names, float_format="%.3f")
            
            latex_table.writeLatexTableEnd()

    # Plot T0Max as a function of Sigma Alpha.
    elif key_start == Start.load_results_singles:

        directories_indecies = [0, 1, 2, 3]
        #directories_indecies = [0]

        KMax = len(factorsNames)
        MMax = len(significantPredictors)
        KMaxPlusMMax = KMax + MMax
        nModelsMax = pow(2, KMaxPlusMMax)

        T0MaxList = []
        T0MinList = []
        T0AvgList =[]
        T0divT0plusTAvgList = []
        TdivT0plusTAvgList = []
        T0IncreasedFractionList = []
        MispricingProbList = []
        TimeVaryingAlphaProbList = []
        TimeVaryingBetaProbList = []


        factorsProbabilityMat    = np.full([len(directories_indecies) , len(factorsNames)]   , fill_value = np.nan, dtype = float)
        predictorsProbabilityMat = np.full([len(directories_indecies) , len(predictorsNames)], fill_value = np.nan, dtype = float)
        CLMLMax = -np.inf

        TauList = []
        ntopModels = 10; row = 0
        filter = np.array([], dtype=int)

        for dir_index in directories_indecies:
            directory_name = directory_name_prefix + str(dir_index)
            print(directory_name)

            with open(directory_name + '/' +dump_file_prefix+str(len(significantPredictors))+'.pkl', 'rb') as file:
                [CLMLList, significantPredictors_temp] = pickle.load(file)

            assert np.allclose(significantPredictors_temp, significantPredictors) and len(CLMLList) == 1

            model = CLMLList[0]

            TauList.append(model["Tau"])
            T0MaxList.append(int(model["T0Max"]))
            T0MinList.append(int(model["T0Min"]))
            T0AvgList.append(int(model["T0Avg"]))
            T0divT0plusTAvgList.append(model["T0divT0plusTAvg"])
            TdivT0plusTAvgList.append(model["TdivT0plusTAvg"])
            MispricingProbList.append(model["MisprisingProb"])

            T0IncreasedFractionList.append(model["T0IncreasedFraction"])
            factorsProbabilityMat[row,:]     = model["factorsProbability"]
            predictorsProbabilityMat[row, :] = model["predictorsProbability"]
            CLMLU = model["LMLU"]
            CLMLR = model["LMLR"]
            CLMLCombined = np.concatenate((CLMLU, CLMLR), axis=0)
            CMLCombined = np.exp(CLMLCombined - max(CLMLCombined))
            CMLCombined /= np.sum(CMLCombined)
            print('Tau= %.3f T0 increace fraction= %.3f' %(model["Tau"], model["T0IncreasedFraction"]))
            print('T0Max = %.2f, T0Avg= %.2f' %(model["T0Max"], model["T0Avg"]))
            print('probability of mispricing= %f  new calculation %f' %(model["MisprisingProb"], np.sum(CMLCombined[0: len(CLMLU)])))
            print('probability of no mispricing= %f' %(np.sum(CMLCombined[len(CLMLU) : ])))

            # I = np.argsort(-CMLCombined) # Adding minus in order to sort in such a way that the first index is the maximum.
            #
            # print('top %d probable models' %(ntopModels))
            # print(CMLCombined[I[0: ntopModels]])                                   # first ntopModels most probable models.
            # print(I[0: ntopModels])

            printFactorsAndPredictorsProbabilities(factorsNames, model["factorsProbability"], predictorsNames,
                                                   model["predictorsProbability"])


            # Maximum log marginal likelihood of all models and restricted and unrestricted.
            CLMLMax = max(np.max(CLMLCombined), CLMLMax)
            print("CLMLMax= %8.4e" %(CLMLMax))
            del CLMLCombined

            # np.arsgsort sorts from in an ascending order, adding a minus will result in descending order.
            nShowModels = 10000
            I = np.argsort(-CMLCombined)
            if dir_index == directories_indecies[0]:
                fig, ax = plt.subplots()
                #plt.title('Cumulative Probability of the Models')
                ax.set_xscale('log')
                ax.set_xlabel('Number of Models')
                ax.set_ylabel('Cumulative Probability')

            # Print the top 10, 100, 500 models cumulative probabilities. (For Si...)
            print("Cumulative Probability of the Models for Tau= %s. Top 10= %f, top 100=%f , top 500=%f " \
                %(str(model["Tau"]), np.sum(CMLCombined[I[0:10]]), np.sum(CMLCombined[I[0:100]]), np.sum(CMLCombined[I[0:500]]) ))

            label = '$\\tau= ' + str(model["Tau"]) + " $"
            ax.plot(np.arange(0, nShowModels) + 1, np.cumsum(CMLCombined[I[0:nShowModels]]), '.-', label=label)
            ax.legend()

            fig.savefig('ModelsCumulativeProbability.png', dpi=600)

            # =====================================
            key_filter_unconditional = True
            if key_filter_unconditional:
                if len(filter) != nModelsMax:
                    UnconditionalMispricingProbList = []
                    filter = setUncontionalFilter(KMax, MMax)
                    # filter = np.zeros((nModelsMax,), dtype=np.int)
                    # for model in np.arange(0, nModelsMax):
                    #     factorsIncludedInModel, predictorsIncludedInModel = \
                    #         retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(model, KMax, MMax)
                    #
                    #     if np.sum(predictorsIncludedInModel)==0:
                    #         filter[model] = 1

                # filter is 1 for the unconditional models and inf for the conditional.
                assert np.sum(1/filter) == pow(2,KMax)
                # Probability of time varying mispricing =
                # Probability of all unrestricted models minus the probability of all un restricted conditional models.
                TimeVaryingAlphaProbList.append(np.sum(CMLCombined[0: len(CLMLU)]) - np.sum(CMLCombined[0: len(CLMLU)]/filter))
                # Probability of time varying beta =
                # 1 - Probability of all unrestricted unconditional - the probability of all the restricted unconditional models.
                TimeVaryingBetaProbList.append(1 - np.sum(CMLCombined[0: len(CLMLU)]/filter) - np.sum(CMLCombined[len(CLMLU):]/filter))
                ratio = 5*5*2**7-1
                print("The ratio between the number of conditional models and unconditiona models is %i " % (ratio))
                print("A priori the probability ratio between conditional and unconditional model should be the ratio of models")
                print("A priori the probability of unconditional models is %.2e and the calculated is %.2e" \
                    %(1/(ratio+1), np.sum(CMLCombined[0: len(CLMLU)]/filter) + np.sum(CMLCombined[len(CLMLU):]/filter)))

                nLast = 0
                assert int(np.sum(1/filter[np.mod(I[0:],len(CLMLU))])) == 2*pow(2,KMax)
                for nTopModels in [int(1e2), int(5e2), int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]:
                    print("In top %i models: Comulative probability %f, a priori number of uncondtional models %i actual number of models %i " \
                        % (nTopModels, np.sum(CMLCombined[I[0:nTopModels]]), \
                        int(nTopModels/(1+ratio)), int(np.sum(1/filter[np.mod(I[0:nTopModels],len(CLMLU))]))))
                    
                    # for model in I[nLast:nTopModels]:
                    #     modelmod = np.mod(model,len(CLMLU))
                    #     factorsIncludedInModel, predictorsIncludedInModel = \
                    #         retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(modelmod, KMax, MMax)
                        
                    #     assert np.sum(predictorsIncludedInModel)==0 and filter[modelmod] == 1
                        
                    # nLast = nTopModels

                assert not np.any(CLMLU >= 0) and not np.any(CLMLR >= 0)
                del CMLCombined
                CLMLCombined = np.concatenate((CLMLU*filter, CLMLR*filter), axis=0)

                CMLCombined = np.exp(CLMLCombined - max(CLMLCombined)); CMLCombined = CMLCombined / np.sum(CMLCombined)
                print('unconditional probability of mispricing= %f ' % (np.sum(CMLCombined[0: len(CLMLU)])))
                print('unconditional probability of no mispricing= %f' % (np.sum(CMLCombined[len(CLMLU):])))
                UnconditionalMispricingProbList.append(np.sum(CMLCombined[0: len(CLMLU)]))
                I = np.argsort(-CMLCombined)
                print(I[0:ntopModels])
                print(I[0:ntopModels]%len(CLMLU))
                print(CMLCombined[I[0:ntopModels]])
                print(np.sum(CMLCombined[I[0:ntopModels]]))
                for m in np.arange(0, ntopModels):
                    factorsIncludedInModel, predictorsIncludedInModel = \
                        retreiveFactorsAndPredictorsFromRowFromAllCombinationsMatrix(I[m], KMax, MMax)
                    print(factorsIncludedInModel)

#                del factorsIncludedInModel, predictorsIncludedInModel

            row += 1

#       End loop on the different values of Tau.

        print('Done loading')

        fig, ax = plt.subplots()
        ax.plot(np.array(TauList), np.array(T0MaxList), '.-', label='T0Max')
        ax.plot(np.array(TauList), np.array(T0MinList), '.-', label='T0Min')
        ax.plot(np.array(TauList), np.array(T0AvgList), '.-', label='T0Avg')

        ax.set_xlabel("Tau")
        ax.legend()
        fig.savefig('T0MaxMinAvgVsTau.png')

        fig, axs = plt.subplots(2)
        axs[0].plot(np.array(TauList), np.array(T0MaxList), '.-', label='T0Max')
        axs[0].plot(np.array(TauList), np.array(T0MinList), '.-', label='T0Min')
        axs[0].plot(np.array(TauList), np.array(T0AvgList), '.-', label='T0Avg')
        ax.set_xlabel("Tau")
        axs[0].legend()

        axs[1].plot(np.array(TauList), np.array(T0IncreasedFractionList), '.-', label='T0IncreasedFraction')
        axs[1].plot(np.array(TauList), np.array(MispricingProbList),      '.-', label='MispricingProbList')
        ax.set_xlabel("Tau")
        axs[1].legend()
        fig.savefig('T0MaxMinAvgT0IncMispricingVsTau.png')

        fig, ax = plt.subplots()
        fig.set_size_inches(14, 6)
        for i in np.arange(0 , len(factorsNames)):
            ax.plot(np.array(TauList), np.array(factorsProbabilityMat[:, i]), '.-', label= factorsNames[i])

        ax.set_xlabel("Tau")
        ax.set_ylabel("Factors Probability")
        ax.legend(loc='upper right')
        fig.savefig('FactorsProbabilityVsTau.png')


        fig, ax = plt.subplots()
        fig.set_size_inches(14, 6)
        for i in np.arange(0 , len(predictorsNames)):
            ax.plot(np.array(TauList), np.array(predictorsProbabilityMat[:, i]), '.-', label= predictorsNames[i])

        ax.set_xlabel("Tau")
        ax.set_ylabel("Predictors Probability")
        ax.legend(loc='upper right')
        fig.savefig('PredictorsProbabilityVsTau.png')

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        plt.title('Factors Histogram')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Number of Predictors')
        for row in np.arange(0 , factorsProbabilityMat.shape[0]):
            I = np.argsort(-factorsProbabilityMat[row,:])
            x = factorsProbabilityMat[row, I]
            label = '$\\tau= ' + str(TauList[row]) + " $"
            ax.plot(x, np.arange(1,KMax+1), '.-', label=label)
            del x , I
        ax.legend()
        ax.grid()

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        plt.title('Predictors Histogram')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Number of Predictors')
        for row in np.arange(0 , predictorsProbabilityMat.shape[0]):
            I = np.argsort(-predictorsProbabilityMat[row,:])
            x = predictorsProbabilityMat[row, I]
            label = '$\\tau= ' + str(TauList[row]) + " $"
            ax.plot(x, np.arange(1,MMax+1), '.-', label=label)
            #ax.hist(predictorsProbabilityMat[row, I],10)
            del x , I
        ax.legend()
        ax.grid()


        for i in np.arange(0 , len(predictorsNames)):
            ax.plot(np.array(TauList), np.array(predictorsProbabilityMat[:, i]), '.-', label= predictorsNames[i])

        ax.set_xlabel("Tau")
        ax.set_ylabel("Predictors Probability")
        ax.legend(loc='upper right')
        fig.savefig('PredictorsProbabilityVsTau.png')


        # Writing the factors, predictors and mispricing probabilities into a Latex table.
        with open('FactorsPredictorsMispricingProbabilitiesTable.tex','w') as file:
                columns_names=[]
                for tau in TauList:
                    columns_names.append('$\\tau= ' + str(tau) + '$')

                caption="The probabilities of the factors and predictors in the asset pricing setup for different values of $\\tau$."
                latex_table = tex.LatexTable(file=file, sidewaystable=False)
                latex_table.writeLatexTableHeader(caption=caption, label=None)
                latex_table.writeLatexSubTableHeader(caption='Probabilities of factors.')
                latex_table.writeProfLatexTabular(data=np.transpose(factorsProbabilityMat), rowsNames=list(factorsNames),
                                          n=columns_names,float_format="%.2f")
                latex_table.writeLatexSubTableEnd()
                # ==================================================================================
                latex_table.writeLatexTableAddSpace()
                # ==================================================================================
                latex_table.writeLatexSubTableHeader(caption='Probabilities of predictors.')
                latex_table.writeProfLatexTabular(data=np.transpose(predictorsProbabilityMat), rowsNames=list(predictorsNames),
                                          n=columns_names,float_format="%.2f")
                latex_table.writeLatexSubTableEnd()
                # ==================================================================================
                latex_table.writeLatexTableAddSpace()
                # ==================================================================================
                latex_table.writeLatexSubTableHeader(caption='Probabilities of mispricing and $T_0$ Statistics.')
                data= np.concatenate((np.array(MispricingProbList).reshape(1,-1),np.array(T0AvgList).reshape(1,-1)),axis=0)
                data= np.concatenate((data,np.array(T0divT0plusTAvgList).reshape(1,-1)),axis=0)
                data= np.concatenate((data,np.array(TdivT0plusTAvgList).reshape(1,-1)),axis=0)
                data= np.concatenate((data, np.array(TimeVaryingAlphaProbList).reshape(1,-1)),axis=0)
                data= np.concatenate((data, np.array(TimeVaryingBetaProbList).reshape(1,-1)),axis=0)
                rowsNames=['Mispricing Probability','Average $T_0$','Average $\\frac{T_0}{T_0+T}$', \
                    'Average $\\frac{T}{T_0+T}$','Probability($\\alpha_1 \\neq 0 $)','Probability($\\beta_1 \\neq 0 $)']
                latex_table.writeProfLatexTabular(data=data, rowsNames=rowsNames,
                                          n=columns_names,float_format="%.3f")
                latex_table.writeLatexSubTableEnd()

                latex_table.writeLatexTableEnd()

        # Integrate over all T0
        CMLRSum = np.zeros((len(directories_indecies),), dtype=np.float64)
        CMLUSum = np.zeros((len(directories_indecies),), dtype=np.float64)

        iter = 0
        for dir_index in directories_indecies:
            directory_name = directory_name_prefix + str(dir_index)
            print(directory_name)

            with open(directory_name + '/' +dump_file_prefix+str(len(significantPredictors))+'.pkl', 'rb') as file:
                [CLMLList, significantPredictors_temp] = pickle.load(file)

            model = CLMLList[0]

            CLMLU = model["LMLU"]
            CLMLR = model["LMLR"]

            CMLUSum[iter] = np.sum(np.exp(CLMLU - CLMLMax))
            CMLRSum[iter] = np.sum(np.exp(CLMLR - CLMLMax))

            print("Iter %i : Unrestricted sum= %8.4e, Restricted sum= %8.4e, CLMLMax= %8.4e"
                  %(iter, CMLUSum[iter] , CMLRSum[iter] , CLMLMax))

            iter += 1

        CMLSum = np.sum(CMLUSum) + np.sum(CMLRSum)

        # End integration over all T0

    else:
        print('Error in start key')

    print('sof tov hakol tov')

    KMax=14
    MMax=13


    T00 = 4700
    T03 = 330
    T=475
    Tstar0 = T00 +T
    Tstar3 = T03 + T
    gf0 = np.zeros((KMax,))
    gf3 = np.zeros((KMax,))
    gN0 = np.zeros((KMax,))
    gN3 = np.zeros((KMax,))
    gN0R = np.zeros((KMax,))
    gN3R = np.zeros((KMax,))

    M=3
    for kp1 in np.arange(1,KMax+1):
        k=kp1-1
        gf0[k]  = multigammaln((Tstar0 + KMax - k - M - 1) / 2, k) - multigammaln((T00 + KMax - M - 1) / 2, k)
        gN0[k]  = multigammaln((Tstar0 - (k+1)* M - 1) / 2, KMax - k) - multigammaln((T00 - (k+1)* M - 1) / 2, KMax - k)
        gN0R[k] = multigammaln((Tstar0 - (k)* M ) / 2, KMax - k) - multigammaln((T00 - (k)* M ) / 2, KMax - k)
        gf3[k]  = multigammaln((Tstar3 + KMax - k - M - 1) / 2, k) - multigammaln((T03 + KMax - M - 1) / 2, k)
        gN3[k]  = multigammaln((Tstar3 - (k+1)* M - 1) / 2, KMax - k)- multigammaln((T03 - (k+1)* M - 1) / 2, KMax - k)
        gN3R[k] = multigammaln((Tstar3 - (k) * M) / 2, KMax - k) - multigammaln((T03 - (k) * M) / 2, KMax - k)

    plt.plot(np.arange(1, KMax + 1), gf0, label='gf0')
    plt.plot(np.arange(1, KMax + 1), gf3, label='gf3')
    plt.plot(np.arange(1, KMax + 1), gN0, label='gN0')
    plt.plot(np.arange(1, KMax + 1), gN3, label='gN3')
    plt.plot(np.arange(1, KMax + 1), gN0R, label='gN0R')
    plt.plot(np.arange(1, KMax + 1), gN3R, label='gN3R')
    plt.plot(np.arange(1, KMax + 1), gf0+gN0, label='gf0+gN0')
    plt.plot(np.arange(1, KMax + 1), gf3 + gN3, label='gf3+gN3')

    plt.plot(np.arange(1, KMax + 1), gf0+gN0R, label='gf0+gN0R')
    plt.plot(np.arange(1, KMax + 1), gf3 + gN3R, label='gf3+gN3R')
    gN0-gN0R
    gN3 - gN3R

    plt.legend()

    fig = plt.figure()
    plt.plot(np.arange(1, KMax + 1), gN0-gN0R, label='gN0-gN0R')
    plt.plot(np.arange(1, KMax + 1), gN3 - gN3R, label='gN0-gN0R')
    plt.legend()
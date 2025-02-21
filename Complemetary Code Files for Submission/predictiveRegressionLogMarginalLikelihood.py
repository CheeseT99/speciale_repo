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
import matplotlib
import matplotlib.pyplot as plt

from CommonFunctions import constructAllCombinationsMatrix, printPredictorsProbabilities
import tictoc
import writeProfLatexTable as tex

keyPrint        = False
keyPrintResults = True
keyDebug        = False


def predictiveRegressionLogMarginalLikelihood(rr, ff, zz, keyInteractions, T0perPredictor):

    # Constants.
    testAssetStartIndex = 1

    if not ((keyInteractions == 0) or  (keyInteractions == 1)):
        print('Error: keyInteractions must be 0 or 1')
        sys.exit()

    # moving on to calculating the marginal likelihood of the predictive regression.
    FOrig = copy.deepcopy(ff)
    factorsNames = FOrig.columns.drop('Date')
    K = len(factorsNames)

    ZOrig = copy.deepcopy(zz)
    predictorsNames = zz.columns.drop('Date')
    MMax  = len(predictorsNames)

    if not R.empty:
        T = rr.shape[0]; T -= 1                       # We are looking for predictability the sample size is smaller by one.
        ROrig = copy.deepcopy(rr)
        testAssetsPortfoliosNames = ROrig.columns.drop('Date')
        N = len(testAssetsPortfoliosNames)

        Y = np.concatenate((ROrig.loc[testAssetStartIndex : , testAssetsPortfoliosNames].values , \
                            FOrig.loc[testAssetStartIndex : , factorsNames].values), axis=1)

    else:
        T = ff.shape[0]; T -= 1
        N = 0
        Y = FOrig.loc[testAssetStartIndex : , factorsNames].values

    assert (Y.shape[0] == T) and (Y.shape[1] == N + K)
    YMean =  np.mean(Y, 0).reshape(-1,1)
    Vy = np.cov(Y, rowvar=False, bias=True).reshape(N + K, N + K)

    Z = ZOrig.loc[0 : T-testAssetStartIndex, predictorsNames].values
    assert (Z.shape[0] == Y.shape[0]) and (Z.shape[1] == MMax)
    ZMean = np.mean(Z, 0).reshape(-1,1)

    predictorsInModel = constructAllCombinationsMatrix(MMax)
    nModelsMax = pow(2, MMax)
    # T0perPredictor = 10;                                                       # number of data entries per parameter.
    logMarginalLikelihood = np.zeros((nModelsMax,), dtype=float); logMarginalLikelihood.fill(np.NINF)

    nLegitModels = 0
    T0Total = 0; T0Max = 0; T0Min = np.inf

    iotaT = np.ones((T,1), dtype=float)

    totalTime = 0.0
    tictoc.tic()
    for model in np.arange(0 , nModelsMax):
        if model % np.floor(nModelsMax/10) == 0:
            totalTime += tictoc.toc(False)
            print('Done %d %% of total work at %.2f sec' %(100*model/nModelsMax , totalTime))

        predictorsIndicesIncludedInModel = np.argwhere(predictorsInModel[model,:] == 1).flatten()

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

            logMarginalLikelihood[model] = np.NINF

            continue

        nLegitModels += 1

        Z = ZOrig.loc[testAssetStartIndex - 1: T-1, predictorsNames[predictorsIndicesIncludedInModel]].values
        assert Z.shape[0] == Y.shape[0]

        X = np.concatenate((iotaT, Z), axis=1)

        modelPredictorsNames = predictorsNames[predictorsIndicesIncludedInModel]
        npredictorsInModel = len(modelPredictorsNames)
        # Adding interactions between the predictors if necessary.
        if (keyInteractions == 1) and  (npredictorsInModel > 0):
            modelPredictorsNames = modelPredictorsNames.to_list()
            ZZInteractions = np.zeros((T , int(npredictorsInModel * (npredictorsInModel + 1) / 2)), dtype=float)
            pp = 0
            for p1 in np.arange(0 , npredictorsInModel ):
                for p2 in np.arange(p1 , npredictorsInModel ):
                    ZZInteractions[: , pp]=Z[: , p1]*Z[: , p2]
                    modelPredictorsNames.append(predictorsNames[predictorsIndicesIncludedInModel[p1]] + '*' + \
                                                     predictorsNames[predictorsIndicesIncludedInModel[p2]])
                    pp += 1

            # Adding the interactions to X.
            X = np.concatenate((X, ZZInteractions), axis=1)

        totalNumberOfPredictors = int(npredictorsInModel + keyInteractions * npredictorsInModel * (npredictorsInModel + 1) / 2)
        # T0 = T0perPredictor * (npredictorsInModel + 1);                                       # + 1 for the intercept.
        T0 = T0perPredictor * (K + N + 2 * totalNumberOfPredictors + 3) / 2                  # + 1 for the intercept.
        T0 = T0perPredictor * (1 + totalNumberOfPredictors + (N + K + 1) / 2)
        T0 = T0perPredictor * (K + N) * (1 + totalNumberOfPredictors + (N + K + 1) / 2) / (
            N + K + npredictorsInModel) # numerator = total number of parameters.denominator = (number of data entries) / T_0.
        T0 = T0perPredictor * (K + N) * (1 + totalNumberOfPredictors + (N + K + 1) / 2) / (
            N + K + totalNumberOfPredictors) # not sure that this is the correct one...i think the previous one is the correct.
        # T0 = T0perPredictor * (K + N) * (N + K + 1) / 2 / (1 + totalNumberOfPredictors + N + K); # Stephan
        # T0 = 1 + N + K + totalNumberOfPredictors; # T0 min = 1 + N + K + totalNumberOfPredictors
        # T0 = 2e6
        if T0 < 1 + N + K + totalNumberOfPredictors:
            print('Error: T0 too small')
            sys.exit()

        T0 = max(np.floor(T0), 1 + N + K + totalNumberOfPredictors)
        if keyPrint:
            print('Model # %i T0= %i T0min= %i npredictorsInModel= %i totalNumberOfPredictors= %i' %                  \
                  (model, T0, (1 + N + K + totalNumberOfPredictors), npredictorsInModel, totalNumberOfPredictors))

        Tstar = T0 + T

        XMean =  np.mean(X, 0).reshape(-1,1)

        X0tX0 = T0 / T * (np.transpose(X) @ X)
        #if LA.det(np.transpose(X) @ X) < 1e-10:
        #    print("det(X'X)= %f parameters= %s" %(LA.det(np.transpose(X) @ X), predictorsInModel[model,:]))

        XtXInv = LA.pinv(np.transpose(X) @ X)
        Btilda = np.transpose(T / Tstar * XtXInv @ (T0 * XMean @ np.transpose(YMean) + np.transpose(X) @ Y))
        Stilda = Tstar * (Vy + YMean @ np.transpose(YMean)) \
                - T / Tstar * (T0 * (YMean @ np.transpose(XMean)) + np.transpose(Y) @ X) @ XtXInv @ (T0 * np.transpose( YMean @ np.transpose(XMean)) + np.transpose(X) @ Y)
        StildaTstar = (Vy + YMean @ np.transpose(YMean)) \
                     - T * ( T0 / Tstar *( YMean @ np.transpose(XMean)) + (np.transpose(Y) @ X) / Tstar ) \
                      @ XtXInv @ ( T0 / Tstar * np.transpose(YMean @ np.transpose(XMean)) + (np.transpose(X) @ Y) / Tstar)
        # logMarginalLikelihood(model) = -T * (N + K) / 2 * log(pi)...
                                 # +(T0 - totalNumberOfPredictors - 1) / 2 * log(det(T0 * Vy))...
                                 # -(Tstar - totalNumberOfPredictors - 1) / 2 * log(det(Stilda))...
                                 # -logmvgamma((T0 - totalNumberOfPredictors - 1) / 2, N + K)...
                                 # +logmvgamma((Tstar - totalNumberOfPredictors - 1) / 2, N + K)...
                                 # -(N + K) * (totalNumberOfPredictors + 1) / 2 * log(Tstar / T0);
        #logMarginalLikelihood[model] = -T * (N + K) / 2 * np.log(np.pi)                                                \
        #                       + (T0    - totalNumberOfPredictors - 1) / 2 * np.log(LA.det(T0 * Vy))                   \
        #                       - (Tstar - totalNumberOfPredictors - 1) / 2 * np.log(LA,det(Stilda))                    \
        #                       - multigammaln((T0    - totalNumberOfPredictors - 1) / 2, N + K)                        \
        #                       + multigammaln((Tstar - totalNumberOfPredictors - 1) / 2, N + K)                        \
        #                       - (N + K) * (totalNumberOfPredictors + 1) / 2 * np.log(Tstar / T0)
        # Rewriting the equation.
        #logMarginalLikelihood[model] = -T * (N + K) / 2 * np.log(np.pi)                                                \
        #                       + (T0    - totalNumberOfPredictors - 1) / 2 * (np.log(LA.det(Vy))             + np.log(T0)    * len(Vy))     \
        #                       - (Tstar - totalNumberOfPredictors - 1) / 2 * (np.log(LA.det(Stilda / Tstar)) + np.log(Tstar) * len(Stilda)) \
        #                       - multigammaln((T0    - totalNumberOfPredictors - 1) / 2, N + K)                        \
        #                       + multigammaln((Tstar - totalNumberOfPredictors - 1) / 2, N + K)                        \
        #                       - (N + K) * (totalNumberOfPredictors + 1) / 2 * np.log(Tstar / T0)
        logMarginalLikelihood[model] = -T * (N + K) / 2 * np.log(np.pi)                                                \
                               + (T0    - totalNumberOfPredictors - 1) / 2 * (np.log(LA.det(Vy))          + np.log(T0)    * len(Vy))     \
                               - (Tstar - totalNumberOfPredictors - 1) / 2 * (np.log(LA.det(StildaTstar)) + np.log(Tstar) * len(Stilda)) \
                               - multigammaln((T0    - totalNumberOfPredictors - 1) / 2, N + K)                        \
                               + multigammaln((Tstar - totalNumberOfPredictors - 1) / 2, N + K)                        \
                               - (N + K) * (totalNumberOfPredictors + 1) / 2 * np.log(Tstar / T0)

        T0Total = T0Total + T0
        T0Max = max(T0Max, T0)
        T0Min = min(T0Min, T0)

    tictoc.toc()

    print('All combinations= %i Total number of legit models= %i %i T0 Average= %f T0 Max= %f T0 Min= %f' % \
        (nModelsMax, np.count_nonzero(logMarginalLikelihood != np.NINF), nLegitModels , T0Total / nLegitModels, T0Max, T0Min))


    return [logMarginalLikelihood, predictorsInModel, predictorsNames]

# Running analysis in this code.
if __name__ == '__main__':

    # Constants.
    T0perPredictor = 50                                                          # number of data entries per parameter.
    print('T0perPredictor= %i' %(T0perPredictor))
    ntopModels     = 10
    class Start(Enum):
        calculate_ML = 1
        load_results = 2

    class DataBase(Enum):
        testAssets80_f14_m13 = 1
        testAssets0_f20_m13  = 2


    dataDir = "/home/lior/Characteristics/python/PostProb/Data/"

    dump_file = 'predictive_regression_dump_models'+'.pkl'

    key_start = Start.calculate_ML

#    key_start = Start.load_results

    key_DataBase = DataBase.testAssets80_f14_m13
    key_DataBase = DataBase.testAssets0_f20_m13

#    significantPredictors = np.array([2,5,7,9,13]) ; significantPredictors -= 1
#    significantPredictors = np.array([])

#    LatexTablesPath = C:\Users\user\Dropbox\financeResearch\PostProb\writeups\tempLatex\tables\';

#    figureFormat = "png"
#    dumpsDirectory = "dumps"


    # Loading the input files.

    if key_DataBase == DataBase.testAssets80_f14_m13:
        # All factors and test assets returns are in percentage.
        # The risk free rate.

        RF = pd.read_csv(dataDir + "RF.csv")
        RF = RF.drop(columns='Unnamed: 0')

        # Test asstets. Loading all portfolios.

        R = pd.read_csv(dataDir + 'AnomaliesPortfoliosReturnsVW.csv')
        R = R.drop(columns='Unnamed: 0')

        testAssetsPortfoliosnames = R.columns.drop('Date')
        print(' max, min of R are= %f, %f' %(np.max(R[testAssetsPortfoliosnames].values),np.min(R[testAssetsPortfoliosnames].values)))

        # Subtract the risk free rate from the test asstes to get excess return.
        for name in testAssetsPortfoliosnames:
            R[name] = R[name].values * 100 - RF['RF'].values

        print(' max, min of R are= %f, %f' %(np.max(R[testAssetsPortfoliosnames].values),np.min(R[testAssetsPortfoliosnames].values)))

        # Loading the factors.
        F = pd.read_csv(dataDir + 'ff.csv')
        F = F.drop(columns='Unnamed: 0')
        # Loading the predictors.
        Z = pd.read_csv(dataDir + 'Z.csv')
        Z = Z.drop(columns='Unnamed: 0')

        assert len(Z) == len(R) and len(R) == len(F)

    elif key_DataBase == DataBase.testAssets0_f20_m13:
        R = pd.DataFrame({'': []})

        # Loading the factors.
        F = pd.read_csv(dataDir + 'factors-20.csv')
        F = F.drop(columns=['MKTRF', 'SMB*', 'MKT', 'CON', 'IA', 'ROE', 'ME'])

        # Loading the predictors.
        Z = pd.read_csv(dataDir + 'Z - 197706.csv')
        Z = Z.drop(columns='Unnamed: 0')

        assert len(Z) == len(F)

    print(F.columns)
    print(Z.columns)

    if key_start == Start.calculate_ML:
        CLMLUList = []
        # Predictive regression log marginal likelihood calculation when the data generating process is linear in z.
        print(' ****** No interactions ****** ')
        [LMLNI, predictorsInModel, predictorsNames] = predictiveRegressionLogMarginalLikelihood(R, F, Z, 0, T0perPredictor)
        MLNI = np.exp(LMLNI - max(LMLNI)); MLNI = MLNI / np.sum(MLNI)
        INI  = np.argsort(-MLNI)       # Adding minus in order to sort in such a way the the first index is the maximum.

        print('Top %d probable models' % (ntopModels))
        print(MLNI[INI[0 : ntopModels]])  # First ntopModels most probable models.
        print('Top %d probable models predictors' % (ntopModels))
        print(predictorsInModel[INI[0 : ntopModels] , :])
        noInteractionsPredictorsProbabilities = MLNI @ predictorsInModel
        printPredictorsProbabilities(predictorsNames, noInteractionsPredictorsProbabilities)

        # Predictive regression log marginal likelihood calculation when the data generating process is squered in z.
        print(' ****** With interactions ****** ')
        [LMLWI, predictorsInModel, predictorsNames] = predictiveRegressionLogMarginalLikelihood(R, F, Z, 1, T0perPredictor)
        MLWI = np.exp(LMLWI - max(LMLWI)); MLWI = MLWI / np.sum(MLWI)
        IWI  = np.argsort(-MLWI)

        print('Top %d probable models' % (ntopModels))
        print(MLWI[IWI[0 : ntopModels]])  # First ntopModels most probable models.
        print('Top %d probable models predictors' % (ntopModels))
        print(predictorsInModel[IWI[0 : ntopModels] , :])
        withInteractionsPredictorsProbabilities = MLWI @ predictorsInModel
        printPredictorsProbabilities(predictorsNames, withInteractionsPredictorsProbabilities)

        # The first model in without and with interactions is the same, namely a model with no predictors.
        # Therfore the model with interaction is starting with index 1 and not 0.
        LMLCombined = np.concatenate((LMLNI, LMLWI[1 :]), axis=0)
        MLCombined  = np.exp(LMLCombined - max(LMLCombined)); MLCombined = MLCombined / np.sum(MLCombined)
        ICombined   = np.argsort(-MLCombined)
        print('Top %d probable models' % (ntopModels))
        print(MLCombined[ICombined[0 : ntopModels]])
        # mod(i - 1, N) + 1 transforms i in 1: N -> 1: N and i in N + 1:2 N -> 1: N.
        # The addition of another 1 is for discarding the first model from the
        # interactions sample since it is the same as the first model without the interactions.
        print(predictorsInModel[np.mod(ICombined[0 : ntopModels]-1, len(LMLNI)) + 1 + 1 , :])
        print('probability of no interactions= %10.3E' %(np.sum(MLCombined[0 : len(LMLNI)])))
        print('probability of interactions=    %10.3E' %(np.sum(MLCombined[len(LMLNI) + 1 : ])))
        predictorsProbability = (MLCombined[0: predictorsInModel.shape[0]] @ predictorsInModel) + (
                    MLCombined[len(LMLNI):] @ predictorsInModel[1:, :])
        printPredictorsProbabilities(predictorsNames, predictorsProbability)

        with open('PredictiveRegressionProbabilitiesTable.tex','w') as file:
                caption="The probabilities of the predictors in the predictive regression setup."
                probs=np.concatenate((noInteractionsPredictorsProbabilities.reshape(1,-1),withInteractionsPredictorsProbabilities.reshape(1,-1),predictorsProbability.reshape(1,-1)),axis=0)
                latex_table = tex.LatexTable(file=file, sidewaystable=True)
                latex_table.writeLatexTableHeader(caption=caption, label=None)
                latex_table.writeProfLatexTabular(np.transpose(probs), rowsNames=list(predictorsNames),
                                          columsNames=['No Interaction', 'With Interactions','Combined'],float_format="%.2f")

                latex_table.writeLatexTableEnd()

        with open(dump_file, 'wb') as file:
                pickle.dump([CLMLUList], file)

        del LMLCombined, LMLNI, LMLWI

    # Plot T0Max as a function of Sigma Alpha.
    if key_start == Start.load_results:

        with open(dump_file, 'rb') as file:
            CLMLUList = pickle.load(file)

    print('sof tov hakol tov')


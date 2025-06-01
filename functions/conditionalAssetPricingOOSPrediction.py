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
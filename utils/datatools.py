import os
import numpy as np
import scipy.io as sio
import pickle as pkl
import copy
from sklearn.decomposition import LatentDirichletAllocation


def readData(dataset):
    if dataset == "BC":
        print ("Load BlogCatalog from mat files. ")
        data = sio.loadmat("./data/realGraph/BC/BC0.mat")
        with open('./data/realGraph/BC/BC_parts.pkl','rb') as f:
            parts = pkl.load(f)
    
    if dataset == "Flickr":
        print ("Load Flickr from mat files. ")
        data = sio.loadmat("./data/realGraph/Flickr/Flickr01.mat")
        with open('./data/realGraph/Flickr/Flickr_parts.pkl','rb') as f:
            parts = pkl.load(f)
            
    return data, parts

def dataSplit(parts):
    trainIndex = []
    valIndex = []
    testIndex = []
    for i in range(len(parts["parts"])):
        if parts["parts"][i]==0:
            trainIndex.append(i)
        elif parts["parts"][i]==1:
            valIndex.append(i)
        else:
            testIndex.append(i)
    print ("Size of train graph:{}, val graph:{}, test graph:{}".format(len(trainIndex),len(valIndex),len(testIndex)))
    return trainIndex,valIndex,testIndex

def covariateTransform(data,dimension,trainIndex,valIndex,testIndex):
    X = data["Attributes"]
    lda = LatentDirichletAllocation(n_components=dimension)
    lda.fit(X)
    X = lda.transform(X)
    trainX = X[trainIndex]
    valX = X[valIndex]
    testX = X[testIndex]
    print ("Shape of graph covariate train:{}, val:{}, test:{}".format(trainX.shape,valX.shape,testX.shape))

    return trainX,valX,testX

def adjMatrixSplit(data,trainIndex,valIndex,testIndex,dataset):
    if dataset == "Flickr":
        A = data["Network"]
    else:
        A = data["Network"].toarray()

    trainA = np.array([a[trainIndex] for a in A[trainIndex]])
    valA = np.array([a[valIndex] for a in A[valIndex]])
    testA = np.array([a[testIndex] for a in A[testIndex]])
    print ("Shape of adj matrix train:{}, val:{}, test:{}".format(trainA.shape,valA.shape,testA.shape))

    return trainA,valA,testA


def sigmod(x):
    return 1/(1 + np.exp(-x))

def distance_matrix(X):
    pairwise_distances = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    return pairwise_distances

def noiseSimulation(data,trainIndex,valIndex,testIndex):

    X = data["Attributes"]
    epsilon = np.random.normal(0,1,X.shape[0])
    epsilonTrain = epsilon[trainIndex]
    epsilonVal = epsilon[valIndex]
    epsilonTest = epsilon[testIndex]

    return epsilonTrain,epsilonVal,epsilonTest


def treatmentSimulationB(w_c,X,A,betaConfounding,betaNeighborConfounding,mode):

    covariate2TreatmentMechanism = sigmod(np.matmul(w_c,X.T))
    neighbors = np.sum(A,1)
    neighbors[neighbors < 0.5] = 1
    neighborAverage = np.divide(np.matmul(A, covariate2TreatmentMechanism.reshape(-1)), neighbors)
    print(mode, np.sum(neighbors==0), np.mean(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage).round(3),np.std(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage).round(3))
    propensityT= sigmod(betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage)
    meanT = np.mean(propensityT)
    T = np.array([1 if x>meanT else 0 for x in propensityT])

    return T,meanT

def potentialOutcomeSimulationB(w,X,A,T,epsilon,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise):

    covariate2OutcomeMechanism = sigmod(np.matmul(w,X.T))
    neighbors = np.sum(A,1)
    neighbors[neighbors < 0.5] = 1
    neighborAverage = np.divide(np.matmul(A, covariate2OutcomeMechanism.reshape(-1)), neighbors)

    Tzero = T-T
    neighborAggregateT = np.divide(np.matmul(A, T.reshape(-1)), neighbors)
    neighborAggregateTzero = np.divide(np.matmul(A, Tzero.reshape(-1)), neighbors)

    potentialOutcomeTN = betaTreat2Outcome*T + betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage+betaNeighborTreatment2Outcome*neighborAggregateT+betaNoise*epsilon
    potentialOutcome0N = betaTreat2Outcome*Tzero + betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage+betaNeighborTreatment2Outcome*neighborAggregateT+betaNoise*epsilon
    potentialOutcomeT0 = betaTreat2Outcome*T + betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage+betaNeighborTreatment2Outcome*neighborAggregateTzero+betaNoise*epsilon
    potentialOutcome00 = betaTreat2Outcome*Tzero + betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage+betaNeighborTreatment2Outcome*neighborAggregateTzero+betaNoise*epsilon
    
    return potentialOutcome00, potentialOutcome0N, potentialOutcomeT0, potentialOutcomeTN


def treatmentSimulation4Neighbor(w_c,X,A,betaConfounding,betaNeighborConfounding,mode):
    covariate2TreatmentMechanism = sigmod(np.matmul(w_c,X.T))
    neighbors = np.sum(A,1)
    neighbors[neighbors < 0.5] = 1
    neighborAverage = np.divide(np.matmul(A, covariate2TreatmentMechanism.reshape(-1)), neighbors)
    mu_T = betaConfounding*covariate2TreatmentMechanism+betaNeighborConfounding*neighborAverage
    propensityT= sigmod(mu_T)
    meanT = np.mean(propensityT)
    QQT = np.array([1 if x>meanT else 0 for x in propensityT])

    T = np.zeros_like(propensityT)
    time = X[:,0:1] + X[:,1:2]

    sorted_index = np.argsort(time, axis=0).flatten()
    for max_ind in sorted_index:
        NNsum = A[:,max_ind].sum()
        if NNsum < 1: NNsum = 1
        temp = sigmod(mu_T[max_ind] + np.dot(A[:,max_ind], T)/NNsum/10)
        T[max_ind] = 1 if temp > meanT else 0

    print(mode, np.sum(neighbors==0), T.mean().round(3), QQT.mean().round(3))
    
    return (covariate2TreatmentMechanism-neighborAverage), T, meanT


def f_Z2Y(A, X, T):
    T = T.reshape(-1, 1)
    distance = distance_matrix(X)
    diag_please = np.matmul(A, distance * T)
    return np.diag(diag_please)

def f_T2Y(covariate2OutcomeMechanism, neighborAverage, function):
    if function == 'M**2+1':
        return (covariate2OutcomeMechanism - neighborAverage) ** 2 + 1
    elif function == 'M+1**2':
        return (covariate2OutcomeMechanism - neighborAverage + 1) ** 2
    else:
        return np.sqrt(covariate2OutcomeMechanism - neighborAverage + 1)
    

def potentialOutcomeSimulation4U(x_dim,sh,fun,w,X,A,T,epsilon,betaTreat2Outcome,betaCovariate2Outcome,betaNeighborCovariate2Outcome,betaNeighborTreatment2Outcome, betaNoise):

    neighbors = np.sum(A,1)
    neighbors[neighbors < 0.5] = 1
    X2Y = sigmod(np.matmul(w[:x_dim],X.T[:x_dim]))
    U2Y = (sigmod(np.matmul(w[x_dim:],X.T[x_dim:])) - 0.5) * sh
    N_X2Y = np.divide(np.matmul(A, X2Y.reshape(-1)), neighbors)
    N_U2Y = np.divide(np.matmul(A, U2Y.reshape(-1)), neighbors)
    
    covariate2OutcomeMechanism = X2Y + U2Y
    neighborAverage = N_X2Y + N_U2Y

    Tzero = T-T
    N2YT = np.divide(f_Z2Y(A, X[:,:x_dim], T), neighbors)
    N2YTzero = np.divide(f_Z2Y(A, X[:,:x_dim], Tzero), neighbors)

    T2Y = f_T2Y(X2Y, N_X2Y, fun)
    XU2Y = betaCovariate2Outcome*covariate2OutcomeMechanism + betaNeighborCovariate2Outcome*neighborAverage + betaNoise*epsilon
    potentialOutcomeTN = betaTreat2Outcome*T     * T2Y + betaNeighborTreatment2Outcome * N2YT     + XU2Y
    potentialOutcome0N = betaTreat2Outcome*Tzero * T2Y + betaNeighborTreatment2Outcome * N2YT     + XU2Y
    potentialOutcomeT0 = betaTreat2Outcome*T     * T2Y + betaNeighborTreatment2Outcome * N2YTzero + XU2Y
    potentialOutcome00 = betaTreat2Outcome*Tzero * T2Y + betaNeighborTreatment2Outcome * N2YTzero + XU2Y
    
    return (covariate2OutcomeMechanism-neighborAverage), T2Y, XU2Y, N2YT, potentialOutcome00, potentialOutcome0N, potentialOutcomeT0, potentialOutcomeTN






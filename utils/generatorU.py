import torch
import numpy as np
import networkx as nx
import scipy.io as sio
import pickle as pkl
import scipy.sparse as sp
from .datatools import *
from collections import Counter
import copy

def calculate_n_hop_neighbors(adjacency_matrix, n):
    if n == 1:
        return adjacency_matrix
    else:
        return np.dot(calculate_n_hop_neighbors(adjacency_matrix, n-1), adjacency_matrix)

def distance_matrix(X):
    pairwise_distances = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    return pairwise_distances

def shortest_path(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    all_shortest_paths = dict(nx.shortest_path_length(G))
    num_nodes = len(G.nodes())
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0
            elif j in all_shortest_paths[i]:
                distance_matrix[i, j] = all_shortest_paths[i][j]
            else:
                distance_matrix[i, j] = 0 

    return distance_matrix

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class Loader:
    def __init__(self, dataset, expID, cuda=False, norm=True, backPNum=False, x_dim=10, sh=0, NL=1, changeA=0, fun='sqrt', bNT2Y=1.0, datatype=None):
        print("=======================================================")
        print("Dataset:{}, expID:{}. Start Loader!!!".format(dataset,expID))
        print("=======================================================")

        self.x_dim, self.sh, self.NL, self.changeA, self.fun, self.bNT2Y = x_dim, sh, NL, changeA, fun, bNT2Y
        if dataset == "BC":
            print ("BC Load")
            file = f"./data/BC/{datatype}/"+str(dataset)+"_expID_"+str(expID)+".pkl"
        if dataset == "Flickr":
            print ("Flickr Load")
            file = f"./data/Flickr/{datatype}/"+str(dataset)+"_expID_"+str(expID)+".pkl"
        with open(file,"rb") as f:
            data = pkl.load(f)
        
        dataTrain,dataValid,dataTest,Ws = data["train"],data["valid"],data["test"],data["weight"]
        self.train = self.dataTransform(dataTrain, cuda, norm, backPNum)
        self.valid = self.dataTransform(dataValid, cuda, norm, backPNum)
        self.test  = self.dataTransform(dataTest,  cuda, norm, backPNum)
        self.backPNum = backPNum

        print ("***************************************************************************************")
        print ("Dataset:{},expID:{} is Done!!".format(dataset,expID))
        print ("***************************************************************************************")
    
    def details(self):
        if self.backPNum:
            trainA, trainX, trainT, trainY, trainNCs, trainTYs, trainPNum = self.train
            validA, validX, validT, validY, validNCs, validTYs, validPNum = self.valid
            testA,  testX,  testT,  testY,  testNCs,  testTYs,  testPNum  = self.test
            return trainA,trainX,trainT,trainY,trainNCs,trainTYs,trainPNum.long(), validA,validX,validT,validY,validNCs,validTYs,validPNum.long(), testA,testX,testT,testY,testNCs,testTYs,testPNum.long()
        else:
            trainA, trainX, trainT, trainY, trainNCs, trainTYs = self.train
            validA, validX, validT, validY, validNCs, validTYs = self.valid
            testA,  testX,  testT,  testY,  testNCs,  testTYs  = self.test
            return trainA,trainX,trainT,trainY,trainNCs,trainTYs, validA,validX,validT,validY,validNCs,validTYs, testA,testX,testT,testY,testNCs,testTYs

    def dataTransform(self, data, cuda=False, norm=True, backPNum=False):
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        A, X, T, Y, NCs, TYs, meanT = data['network'],data['features'],data["T"],data["Y"],data["NCfeatures"],data["TYs"],data["meanT"]
        PNum = Tensor(data["PNum"])
        X = Tensor(normalize(X)) if norm else Tensor(X)
        A, T, Y, NCs, TYs= Tensor(A),Tensor(T),Tensor(Y),Tensor(NCs),Tensor(TYs)
        if backPNum:
            return A, X, T, Y, NCs, TYs, PNum
        else:
            return A, X, T, Y, NCs, TYs

class Generator:
    def __init__(self, dataset, dimension, expID, x_dim=10, sh=0, NL=1, changeA=0, fun='sqrt', bNT2Y=1.0, seed=2024):
        print("=======================================================")
        print("Dataset:{}, expID:{}. Start Generator!!!".format(dataset,expID))
        print("=======================================================")

        np.random.seed(seed+expID)
        self.expID = expID
        data, parts = readData(dataset)
        print(f"Attributes: {data['Attributes'].shape}; Network: {data['Network'].shape}")
        print(f"cut: {parts['cut']};" + " "*15 + f"parts: {Counter(parts['parts'])}")

        print("Split Graph Data into Training/Validation/Testing Graph. ")
        trainIndex,valIndex,testIndex = dataSplit(parts)
        self.trainX, self.valX, self.testX = covariateTransform(data,dimension,trainIndex,valIndex,testIndex)
        self.trainA, self.valA, self.testA = adjMatrixSplit(data,trainIndex,valIndex,testIndex,dataset)
        self.x_dim, self.sh, self.NL, self.changeA, self.fun, self.bNT2Y = x_dim, sh, NL, changeA, fun, bNT2Y
        ''' Set Data Generation Parameters '''
        self.betaConfounding = 1 # effect of features X to T (confounding1)
        self.betaNeighborConfounding = 1 # effect of Neighbor features to T (confounding2)
        self.betaTreat2Outcome = 1 # effect of treatment to potential outcome
        self.betaCovariate2Outcome = 1 # effect of features to potential outcome (confounding1)
        self.betaNeighborCovariate2Outcome = 0.5 # effect of Neighbor features to potential outcome
        self.betaNeighborTreatment2Outcome = bNT2Y # effect of interence
        self.betaNoise = 0.1 # noise

        print(f"Epsilon: Sample from Normal(0.0, 1.0).")
        self.epsilonTrain, self.epsilonVal, self.epsilonTest = noiseSimulation(data,trainIndex,valIndex,testIndex)

        print(f"Effect of X to T (w_z1): Sample {dimension} Parameters from Uniform(-1.0, 1.0).")
        self.w_z1 = 2 * np.random.random_sample((dimension)) - 1 # effect of X to T 

        print(f"Effect of X to Y (w_z2): Sample {dimension} Parameters from Uniform(-1.0, 1.0).")
        self.w_z2 = 2 * np.random.random_sample((dimension)) - 1 #effect of T to Y

        print(f"Effect of X to NCT (w_nc1): Sample {dimension} Parameters from Uniform(-1.0, 1.0).")
        self.w_nc1 = 2 * np.random.random_sample((dimension)) - 1 # effect of X to T 

        print(f"Effect of X to NCY (w_nc2): Sample {dimension} Parameters from Uniform(-1.0, 1.0).")
        self.w_nc2 = 2 * np.random.random_sample((dimension)) - 1 #effect of T to Y

        Ws = np.array([self.w_z1,self.w_z2,self.w_nc1,self.w_nc2])
        print(f"Weights: {Ws.shape}")

        train, valid, test = self.gen()
        data = {"train":train,"valid":valid,"test":test,"weight":Ws}

        self.saveData(dataset,data)

        print ("***************************************************************************************")
        print ("Dataset:{},expID:{} is Done!!".format(dataset,self.expID))
        print ("***************************************************************************************")

    def saveData(self,dataset,data):
        if dataset == "BC":
            print ("BC Save")
            os.makedirs(f"./data/BC/simulationU{self.x_dim}{self.sh}{self.NL}{self.changeA}_{self.fun}{self.bNT2Y}/", exist_ok=True)
            file = f"./data/BC/simulationU{self.x_dim}{self.sh}{self.NL}{self.changeA}_{self.fun}{self.bNT2Y}/"+str(dataset)+"_expID_"+str(self.expID)+".pkl"
        if dataset == "Flickr":
            print ("Flickr Save")
            os.makedirs(f"./data/Flickr/simulationU{self.x_dim}{self.sh}{self.NL}{self.changeA}_{self.fun}{self.bNT2Y}/", exist_ok=True)
            file = f"./data/Flickr/simulationU{self.x_dim}{self.sh}{self.NL}{self.changeA}_{self.fun}{self.bNT2Y}/"+str(dataset)+"_expID_"+str(self.expID)+".pkl"
        
        with open(file,'wb') as f:
            pkl.dump(data,f)

    def gen(self):
        train = self.XA2NCTY(self.trainX,self.trainA,self.epsilonTrain,'train')
        valid = self.XA2NCTY(self.valX,self.valA,self.epsilonVal,'valid')
        test  = self.XA2NCTY(self.testX,self.testA,self.epsilonTest,'test')

        return train, valid, test
    
    def XA2NCTY(self, dataX, dataA, epsilon=0, dataM='train'):

        if self.NL > 3:
            disM = distance_matrix(dataX[:,:5])
            dataAA = (disM < 0.25).astype(float) * (disM > 0.20).astype(float)
            print("Random Edges: ", int(dataAA.sum()), )
        else:
            dataAA = copy.deepcopy(dataA)
            if self.NL > 1:
                newA = dataAA
                for i in range(2, self.NL+1):
                    newA = newA + calculate_n_hop_neighbors(dataAA, i)
                np.fill_diagonal(newA, 0)
                dataAA = copy.deepcopy(newA)
                dataAA = (dataAA > 0.5).astype(float)
            
            if self.changeA == 1:
                disM = distance_matrix(dataX[:,:5])
                dataA1 = (disM < 1.0).astype(float) * dataA
                print("Full Edges: ", int(dataAA.sum()), )
                dataA2 = (disM < 0.2).astype(float) * (disM > 0.1).astype(float) * dataAA
                if self.NL < 1.5:
                    dataAA = dataA1
                else:
                    dataAA = ((dataA1 + dataA2) > 0.5).astype(float)
                print('Edges Number changes:', int(dataA.sum()), int(dataA1.sum()), int(dataA2.sum()), int(dataAA.sum()))
                
            print(f"Edges Number: {int(dataAA.sum())}. ")

        _, T_data, meanT_data = treatmentSimulation4Neighbor(self.w_z1,dataX,dataAA,self.betaConfounding,self.betaNeighborConfounding,dataM)
        _,_,_,_, Y00_data, Y0N_data, YT0_data, YTN_data = potentialOutcomeSimulation4U(self.x_dim,self.sh,self.fun,self.w_z2,dataX,dataAA,T_data,epsilon,self.betaTreat2Outcome,self.betaCovariate2Outcome,self.betaNeighborCovariate2Outcome,self.betaNeighborTreatment2Outcome,self.betaNoise)

        NCAddT, NCT_data, meanNCT_data = treatmentSimulation4Neighbor(self.w_nc1,dataX,dataAA,self.betaConfounding,self.betaNeighborConfounding,dataM)
        NCAddY, NCT2Y, NCXU2Y, NCN2YT, NCY00, NCY0N, NCYT0, NCYTN = potentialOutcomeSimulation4U(self.x_dim,self.sh,self.fun,self.w_nc2,dataX,dataAA,np.ones_like(NCT_data),epsilon,self.betaTreat2Outcome,self.betaCovariate2Outcome,self.betaNeighborCovariate2Outcome,self.betaNeighborTreatment2Outcome,self.betaNoise)

        T_data2 = self.newTwithP(T_data, 0.2)
        _,_,_,_, Y00_data2, Y0N_data2, YT0_data2, YTN_data2 = potentialOutcomeSimulation4U(self.x_dim,self.sh,self.fun,self.w_z2,dataX,dataAA,T_data2,epsilon,self.betaTreat2Outcome,self.betaCovariate2Outcome,self.betaNeighborCovariate2Outcome,self.betaNeighborTreatment2Outcome,self.betaNoise)

        T_data5 = self.newTwithP(T_data, 0.5)
        _,_,_,_, Y00_data5, Y0N_data5, YT0_data5, YTN_data5 = potentialOutcomeSimulation4U(self.x_dim,self.sh,self.fun,self.w_z2,dataX,dataAA,T_data5,epsilon,self.betaTreat2Outcome,self.betaCovariate2Outcome,self.betaNeighborCovariate2Outcome,self.betaNeighborTreatment2Outcome,self.betaNoise)

        T_data8 = self.newTwithP(T_data, 0.8)
        _,_,_,_, Y00_data8, Y0N_data8, YT0_data8, YTN_data8 = potentialOutcomeSimulation4U(self.x_dim,self.sh,self.fun,self.w_z2,dataX,dataAA,T_data8,epsilon,self.betaTreat2Outcome,self.betaCovariate2Outcome,self.betaNeighborCovariate2Outcome,self.betaNeighborTreatment2Outcome,self.betaNoise)

        T_data10 = self.newTwithP(T_data, 1.0)
        _,_,_,_, Y00_data10, Y0N_data10, YT0_data10, YTN_data10 = potentialOutcomeSimulation4U(self.x_dim,self.sh,self.fun,self.w_z2,dataX,dataAA,T_data10,epsilon,self.betaTreat2Outcome,self.betaCovariate2Outcome,self.betaNeighborCovariate2Outcome,self.betaNeighborTreatment2Outcome,self.betaNoise)

        dataTYs = np.array([[T_data, Y00_data, Y0N_data, YT0_data, YTN_data],
                   [T_data2, Y00_data2, Y0N_data2, YT0_data2, YTN_data2],
                   [T_data5, Y00_data5, Y0N_data5, YT0_data5, YTN_data5],
                   [T_data8, Y00_data8, Y0N_data8, YT0_data8, YTN_data8],
                   [T_data10, Y00_data10, Y0N_data10, YT0_data10, YTN_data10]])
        dataNCs = np.array([NCAddT, NCAddY, NCT2Y, NCXU2Y, NCN2YT, NCT_data, NCY00, NCY0N, NCYT0, NCYTN])

        data = {'T':np.array(T_data),
                'Y':np.array(YTN_data), 
                'features': dataX, 
                'NCfeatures': dataNCs, 
                'TYs': dataTYs, 
                'network':dataA,
                'PNum':shortest_path(dataA),
                "meanT":meanT_data}
        
        return data

    def newTwithP(self, T, p=0.2):
        num = len(T)
        num_ones = int(num * p)
        new_T = np.zeros_like(T)
        new_T[np.random.choice(num, num_ones, replace=False)] = 1
        return new_T

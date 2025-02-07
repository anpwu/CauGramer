import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import XGraphMultiHeadAttention, TGraphMultiHeadAttention, XGraphConvolution, TGraphConvolution

SQRT_CONSTL = 1e-10
SQRT_CONSTR = 1e10

def safe_sqrt(x, lbound=SQRT_CONSTL, rbound=SQRT_CONSTR):
    return torch.sqrt(torch.clamp(x, lbound, rbound))

class CauGramer(nn.Module):
    def __init__(self, modelName, nodeDim, hiddenDim, headsNum, attsNum, inputNum=1, outputNum=1, dropout=0.0, max_degree=5, max_path_distance=4, cuda=False, Xatt_use=True, Tatt_use=True, FCatt_use=False, sa=1.0, sb=1.0, add=0.0):
        super(CauGramer, self).__init__()

        self.sa = sa
        self.sb = sb
        self.add = add
        self.modelName  = modelName
        self.nodeDim    = nodeDim
        self.hiddenDim  = hiddenDim
        self.headsNum   = headsNum
        self.attsNum    = attsNum
        self.inputNum   = inputNum 
        self.outputNum  = outputNum
        self.dropout    = dropout
        self.max_degree = max_degree
        self.max_path_distance = max_path_distance
        self.cuda     = cuda
        self.Xatt_use = Xatt_use
        self.Tatt_use = Tatt_use
        self.FCatt_use = FCatt_use


        if self.Xatt_use:
            self.XREP = nn.ModuleList([XGraphMultiHeadAttention(self.headsNum, self.nodeDim, self.hiddenDim, self.hiddenDim, 0.0)]+[XGraphMultiHeadAttention(self.headsNum, self.hiddenDim, self.hiddenDim, self.hiddenDim, self.dropout) for _ in range(self.inputNum - 1)])
        else:
            self.XREP = nn.ModuleList([XGraphConvolution(self.nodeDim, self.hiddenDim)] + [XGraphConvolution(self.hiddenDim, self.hiddenDim) for _ in range(self.inputNum - 1)])
        if self.Tatt_use:
            self.TREP = nn.ModuleList([TGraphMultiHeadAttention(self.headsNum, self.nodeDim, self.hiddenDim, self.hiddenDim, 0.0)]+[TGraphMultiHeadAttention(self.headsNum, self.hiddenDim, self.hiddenDim, self.hiddenDim, self.dropout) for _ in range(self.attsNum - 1)])
        else:
            self.TREP = nn.ModuleList([TGraphConvolution(self.hiddenDim, self.hiddenDim)] + [TGraphConvolution(self.hiddenDim, self.hiddenDim) for _ in range(self.attsNum - 1)])

        if self.FCatt_use:
            self.OUT_t0 = nn.ModuleList([nn.Linear(self.hiddenDim*2+1,self.hiddenDim)] + [nn.Linear(self.hiddenDim,self.hiddenDim) for i in range(self.outputNum-1)])
            self.OUT_t1 = nn.ModuleList([nn.Linear(self.hiddenDim*2+1,self.hiddenDim)] + [nn.Linear(self.hiddenDim,self.hiddenDim) for i in range(self.outputNum-1)])
            self.qk_self = nn.ModuleList([nn.Linear(self.hiddenDim*2+1,self.hiddenDim)] + [nn.Linear(self.hiddenDim,self.hiddenDim) for i in range(self.outputNum-1)])
            self.q = nn.Linear(self.hiddenDim, 1)
            self.k = nn.Linear(self.hiddenDim, 1)
            self.PRED_t0 = nn.Linear(self.hiddenDim,1)
            self.PRED_t1 = nn.Linear(self.hiddenDim,1)
            self.pp_self = nn.ModuleList([nn.Linear(self.hiddenDim*2+1,self.hiddenDim)] + [nn.Linear(self.hiddenDim,self.hiddenDim) for i in range(self.outputNum-1)])
            self.pp = nn.Linear(self.hiddenDim, 1)
            self.pp_act = nn.Sigmoid()
        else:
            self.OUT_t0 = nn.ModuleList([nn.Linear(self.hiddenDim*2+1,self.hiddenDim)] + [nn.Linear(self.hiddenDim,self.hiddenDim) for i in range(self.outputNum-1)])
            self.OUT_t1 = nn.ModuleList([nn.Linear(self.hiddenDim*2+1,self.hiddenDim)] + [nn.Linear(self.hiddenDim,self.hiddenDim) for i in range(self.outputNum-1)])
            self.PRED_t0 = nn.Linear(self.hiddenDim,1)
            self.PRED_t1 = nn.Linear(self.hiddenDim,1)

        
    
    def forward(self, adj, x, t, PNum):
        neighbors = torch.sum(adj, 1)
        neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        
        num = adj.shape[0]
        diag = torch.diag(torch.FloatTensor([1 for _ in range(num)]))

        if self.Xatt_use:
            Xrep = F.relu(self.XREP[0](adj, x, t, PNum))
            for i in range(1, self.inputNum):
                if self.add == 0:
                    Xrep = F.relu(self.XREP[i](adj, Xrep, t, PNum))
                else:
                    Xrep = self.add * Xrep + F.relu(self.XREP[i](adj, Xrep, t, PNum))
        else:
            Xrep = F.relu(self.XREP[0](x, adj+diag))
            for i in range(1, self.inputNum):
                Xrep = F.relu(self.XREP[i](Xrep, adj+diag))
                Xrep = F.dropout(Xrep, self.dropout, training=self.training)
        # Xrep_norm = Xrep / safe_sqrt(torch.sum(torch.square(Xrep), dim=1, keepdim=True))

        if self.Tatt_use:
            Trep = F.relu(self.TREP[0](adj, x, t, PNum))
            for i in range(1, self.attsNum):
                if self.add == 0:
                    Trep = F.relu(self.TREP[i](adj, Xrep, t, PNum))
                else:
                    Trep = self.add * Trep + F.relu(self.TREP[i](adj, Xrep, t, PNum))
        else:
            Trep = F.relu(self.TREP[0](Xrep, adj, t))
            for i in range(1, self.attsNum):
                Trep = F.relu(self.TREP[i](Trep, adj, t))
        # Trep_norm = Trep / safe_sqrt(torch.sum(torch.square(Trep), dim=1, keepdim=True))
        
        rep = torch.cat((Xrep, Trep), 1)
        rep1 = torch.cat((rep, neighborAverageT.reshape(-1, 1)), 1)
        out_y0 = rep1
        out_y1 = rep1
        for i in range(self.outputNum):
            out_y0 = F.dropout(F.relu(self.OUT_t0[i](out_y0)), self.dropout, training=self.training)
            out_y1 = F.dropout(F.relu(self.OUT_t1[i](out_y1)), self.dropout, training=self.training)

        pred_y0 = self.PRED_t0(out_y0).view(-1)
        pred_y1 = self.PRED_t1(out_y1).view(-1)

        pred_y = torch.where(t > 0, pred_y1, pred_y0)

        if self.FCatt_use:
            qk_x = rep1
            pp_x = rep1
            for i in range(self.outputNum):
                qk_x=self.qk_self[i](qk_x)
                pp_x=self.pp_self[i](pp_x)
            query = self.q(qk_x)
            key = self.k(qk_x)
            attention = (query * key).view(-1)
            pt = self.pp_act(self.pp(pp_x)).view(-1)
            return pred_y0, pred_y1, pred_y, rep, attention*(self.sb-self.sa)+self.sa, pt
        else:
            return pred_y0, pred_y1, pred_y, rep, -1, -1

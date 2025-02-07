import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MLPMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, inputDim, hiddenDim, dim_qkv=32, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([MLPAttentionHead(inputDim, dim_qkv, dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_qkv, hiddenDim)

    def forward(self, x):
        return self.linear(torch.cat([attention_head(x) for attention_head in self.heads], dim=-1))
    
class MLPAttentionHead(nn.Module):
    def __init__(self, inputDim, dim_qkv=32, dropout=0.0):
        super().__init__()

        self.dropout = dropout
        self.q = nn.Linear(inputDim, dim_qkv)
        self.k = nn.Linear(inputDim, dim_qkv)
        self.v = nn.Linear(inputDim, dim_qkv)
    
    def forward(self, x, fcFlag=False):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        if fcFlag:
            a = query * key
            attention = torch.softmax(a, dim=-1)
            return attention, value
        else:
            a = self.compute_a(key, query)
            attention = torch.softmax(a, dim=-1)
            return attention.mm(value)

    def compute_a(self, key, query):
        return query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
    
class TGraphMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, inputDim, hiddenDim, dim_qkv=32, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [TGraphAttentionHead(inputDim, hiddenDim, dim_qkv, dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_qkv, hiddenDim)

    def forward(self,adj, x, t, PNum):
        return self.linear(
            torch.cat([
                attention_head(adj, x, t, PNum) for attention_head in self.heads
            ], dim=-1)
        )
    
class TGraphAttentionHead(nn.Module):
    def __init__(self, inputDim, hiddenDim, dim_qkv=32, dropout=0.0):
        super().__init__()

        self.dropout = dropout
        self.x_self = nn.Linear(inputDim, hiddenDim)
        self.x_neigh = TGraphConvolution(inputDim, hiddenDim)
        self.x_comb = TGraphConvolution(inputDim, hiddenDim)

        self.q = nn.Linear(hiddenDim, dim_qkv)
        self.k = nn.Linear(hiddenDim, dim_qkv)
        self.v = nn.Linear(hiddenDim, dim_qkv)
    
    def forward(self, adj, x, t, PNum):
        
        num = adj.shape[0]
        diag = torch.diag(torch.FloatTensor([1 for _ in range(num)]))

        h_x = F.relu(self.x_self(x))
        h_x = F.dropout(h_x, self.dropout, training=self.training)

        h_n = F.relu(self.x_neigh(x, adj, t))
        h_n = F.dropout(h_n, self.dropout, training=self.training)

        h_c = F.relu(self.x_comb(x, adj+diag, t))
        h_c = F.dropout(h_c, self.dropout, training=self.training)

        query = self.q(h_x)
        key = self.k(h_n)
        value = self.v(h_c)

        a = self.compute_a(key, query)
        softmax = torch.softmax(a, dim=-1)
        x = softmax.mm(value)

        return x

    def compute_a(self, key, query):
        return query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
    
class XGraphMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, inputDim, hiddenDim, dim_qkv=32, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [XGraphAttentionHead(inputDim, hiddenDim, dim_qkv, dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_qkv, hiddenDim)

    def forward(self,adj, x, t, PNum):
        return self.linear(
            torch.cat([
                attention_head(adj, x, t, PNum) for attention_head in self.heads
            ], dim=-1)
        )
    
class XGraphAttentionHead(nn.Module):
    def __init__(self, inputDim, hiddenDim, dim_qkv=32, dropout=0.0):
        super().__init__()

        self.dropout = dropout
        self.x_self = nn.Linear(inputDim, hiddenDim)
        self.x_neigh = XGraphConvolution(inputDim, hiddenDim)
        self.x_comb = XGraphConvolution(inputDim, hiddenDim)

        self.q = nn.Linear(hiddenDim, dim_qkv)
        self.k = nn.Linear(hiddenDim, dim_qkv)
        self.v = nn.Linear(hiddenDim, dim_qkv)
    
    def forward(self, adj, x, t, PNum):
        num = adj.shape[0]
        diag = torch.diag(torch.FloatTensor([1 for _ in range(num)]))

        h_x = F.relu(self.x_self(x))
        h_x = F.dropout(h_x, self.dropout, training=self.training)

        h_n = F.relu(self.x_neigh(x, adj))
        h_n = F.dropout(h_n, self.dropout, training=self.training)

        h_c = F.relu(self.x_comb(x, adj+diag))
        h_c = F.dropout(h_c, self.dropout, training=self.training)

        query = self.q(h_x)
        key = self.k(h_n)
        value = self.v(h_c)

        a = self.compute_a(key, query)
        softmax = torch.softmax(a, dim=-1)
        x = softmax.mm(value)

        return x

    def compute_a(self, key, query):
        return query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
    

class XGraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(XGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.register_parameter("weight", self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class TGraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.register_parameter("weight", self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, t):
        support = torch.mm(input, self.weight) * t.view(-1,1)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
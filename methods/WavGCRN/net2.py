import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time
from layer import *
import sys
from collections import OrderedDict
import pywt
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as expit
import matplotlib.pyplot as plt

class feature_agg1(nn.Module):
    def __init__(self, input_size, output_size):
        super(feature_agg1, self).__init__()
        self.CAM = nn.Sequential(nn.Linear(input_size, output_size), nn.GELU())

    def forward(self, x1, x2):
        score = self.CAM(torch.cat((x1,x2), dim=-1)) #(B,N,1)
        return score

class feature_agg2(nn.Module):
    def __init__(self, input_size, output_size):
        super(feature_agg2, self).__init__()
        self.CAM = nn.Sequential(nn.Linear(input_size, output_size), nn.GELU())

    def forward(self, x1):
        score = self.CAM(x1) #(B,N,1)
        return score

class IDWTL(nn.Module):
    def __init__(self, input_dim):
        super(IDWTL, self).__init__()
        self.input_dim = input_dim

        # 定义可学习参数，即一个形状为(207, 64)的矩阵
        #self.CAM = nn.Sequential(nn.Linear(self.input_dim[1], self.input_dim[1]), nn.GELU())
        self.weight = nn.Parameter(torch.Tensor(self.input_dim[0], self.input_dim[1]))
        self.gelu = nn.GELU()

        # 初始化权重
        #nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.weight, a=15, b=30)
        #self.weight.data *= 5

    def forward(self, x):
        # x的shape为(64, 207, 64)
        # 对输入张量的每个切片与可学习矩阵进行Hadamard乘积
        x = x * self.weight.unsqueeze(0)
        x = self.gelu(x)

        return x


class WavGCRN(nn.Module):
    def __init__(self,
                 gcn_depth,
                 num_nodes,
                 device,
                 predefined_A=None,
                 dropout=0.3,
                 subgraph_size=20,
                 node_dim=40,
                 middle_dim=2,
                 seq_length=12,
                 in_dim=2,
                 out_dim=12,
                 layers=3,
                 list_weight=[0.05, 0.95, 0.95],
                 tanhalpha=3,
                 cl_decay_steps=4000,
                 rnn_size=64,
                 hyperGNN_dim=16,):
        super(WavGCRN, self).__init__()
        self.output_dim = 1
        self.config = config
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A

        self.seq_length = seq_length

        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

        self.idx = torch.arange(self.num_nodes).to(device)

        self.rnn_size = rnn_size
        self.in_dim = in_dim

        #hidden_size = self.rnn_size
        self.hidden_size_de = self.rnn_size
        self.hidden_size_en = self.rnn_size #int(self.rnn_size/2)

        '''
        dims_hyper = [ self.hidden_size + in_dim, hyperGNN_dim, middle_dim, node_dim ]

        self.GCN1_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        '''

        self.fc_final = nn.Linear(self.hidden_size_de, self.output_dim)

        self.alpha = tanhalpha
        self.device = device
        self.k = subgraph_size
        dims_en = [in_dim + self.hidden_size_en, self.hidden_size_en]
        dims_de = [in_dim + self.hidden_size_de, self.hidden_size_de]

        self.aggC = feature_agg1(self.hidden_size_en * 2, self.hidden_size_de)
        self.aggH = feature_agg1(self.hidden_size_en * 2, self.hidden_size_de)
        self.camH = feature_agg2(self.hidden_size_en * 2, self.hidden_size_de)
        self.camC = feature_agg2(self.hidden_size_en * 2, self.hidden_size_de)
        self.IDWT_L = IDWTL([207, self.hidden_size_en])
        self.IDWT_H = IDWTL([207, self.hidden_size_en])

        self.gz1D = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2D = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1D = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2D = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1D = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2D = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')

        self.gz1AD = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2AD = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1AD = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2AD = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1AD = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2AD = gcn(dims_en, gcn_depth, dropout, *list_weight, 'RNN')

        self.gz1_de = gcn(dims_de, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2_de = gcn(dims_de, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = gcn(dims_de, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2_de = gcn(dims_de, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = gcn(dims_de, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2_de = gcn(dims_de, gcn_depth, dropout, *list_weight, 'RNN')

        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth

    def preprocessing(self, adj, predefined_A):
        adj = adj + torch.eye(self.num_nodes).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def cal_wavelet1(self, input, level):
        x = input #64,207,12
        wavelet = 'db1'  # 使用Daubechies 1小波

        coeffs = pywt.wavedec(x.cpu(), wavelet, level=level, axis=-1)
        xA = torch.tensor(coeffs[0])
        low = coeffs[1:]
        xD = torch.tensor(low[0])
        #xAD = torch.tensor(low[1])

        return xD.to(self.device), xA.to(self.device) #(64,207,6)

    def cal_wavelet2(self, input, level):
        x = input #64,207,12
        wavelet = 'db1'  # 使用Daubechies 1小波
        output = []

        for i in range(level):
            coeffs = pywt.wavedec(x.cpu(), wavelet, level=1, axis=-1)
            x = torch.tensor(coeffs[0])
            xD = coeffs[1:]
            output.append(xD)
        output.append(x)
        assert len(output) == level + 1

        return [torch.tensor(element).squeeze().to(self.device) for element in output]

    def interleave_tensors(self, A, B):
        # A和B的shape都为(64, 207, 32)
        batch_size, seq_len, features = A.size()

        # 将A和B展开为(64 * 207, 32)
        A_flat = A.view(-1, features)
        B_flat = B.view(-1, features)

        # 创建一个形状为(64 * 207, 64)的空白张量
        C = torch.zeros(batch_size * seq_len, features * 2, device=A.device)

        # 将A的值放入拼接后的偶数列，B的值放入拼接后的奇数列
        C[:, ::2] = A_flat
        C[:, 1::2] = B_flat

        # 将拼接后的张量重塑回原始形状(64, 207, 64)
        C = C.view(batch_size, seq_len, features * 2)

        return C

    def idwtl_layer(self, hidden_state_list):
        a = torch.tensor(hidden_state_list[0]).squeeze()
        for i in range(1, len(hidden_state_list)):
            a = self.interleave_tensors(self.IDWT_L(a) + self.IDWT_L(torch.tensor(hidden_state_list[i]).squeeze()),
                                        self.IDWT_H(a) + self.IDWT_H(torch.tensor(hidden_state_list[i]).squeeze()))
        return a

    def step(self,
             input,
             Hidden_State,
             Cell_State,
             predefined_A,
             md,
             frequncy='D',
             type='encoder',
             idx=None,
             i=None):

        x = input

        x = x.transpose(1, 2).contiguous()

        adp = self.preprocessing(md, predefined_A[0])
        adpT = self.preprocessing(md.transpose(1, 2), predefined_A[1])

        if type == 'encoder':
            Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size_en)
            Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size_en)
            combined = torch.cat((x, Hidden_State), -1)

            if frequncy == 'D':
                z = F.sigmoid(self.gz1D(combined, adp) + self.gz2D(combined, adpT))
                r = F.sigmoid(self.gr1D(combined, adp) + self.gr2D(combined, adpT))

                temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
                Cell_State = F.tanh(self.gc1D(temp, adp) + self.gc2D(temp, adpT))

            if frequncy == 'AD':
                z = F.sigmoid(self.gz1AD(combined, adp) + self.gz2AD(combined, adpT))
                r = F.sigmoid(self.gr1AD(combined, adp) + self.gr2AD(combined, adpT))

                temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
                Cell_State = F.tanh(self.gc1AD(temp, adp) + self.gc2AD(temp, adpT))

        elif type == 'decoder':
            Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size_de)
            Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size_de)
            combined = torch.cat((x, Hidden_State), -1)

            z = F.sigmoid(
                self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = F.sigmoid(
                self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(
                self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(
            1 - z, Cell_State)

        return Hidden_State, Cell_State

    def forward(self,
                input,
                md1, md2,
                idx=None,
                ycl=None,
                batches_seen=None,
                task_level=12):

        predefined_A = self.predefined_A
        x = input
        batch_size = x.size(0)
        T = x[:, 1, :, :]
        T_ds = T[:, :, ::2] #2 = level+1
        f = x[:, 0, :, :]
        f_ds = f[:, :, ::2]

        level = 1
        xD, xAD = self.cal_wavelet2(f, level)
        Hidden_State = []
        Cell_State = []
        
        Hidden_State_1, Cell_State_1 = self.initHidden(batch_size * self.num_nodes, self.hidden_size_en)
        Hidden_State_2, Cell_State_2 = self.initHidden(batch_size * self.num_nodes, self.hidden_size_en)

        for i in range(xD.shape[-1]):#self.seq_length
            x1 = torch.squeeze(xD[..., i]).unsqueeze(1)
            t = torch.squeeze(T_ds[..., i]).unsqueeze(1)
            x1 = torch.cat((x1,t), dim=1)
            Hidden_State_1, Cell_State_1 = self.step(x1, Hidden_State_1, Cell_State_1,
                                                     predefined_A, md1, 'D', 'encoder', idx, i)
        Hidden_State.append(Hidden_State_1)
        Cell_State.append(Cell_State_1)

        for i in range(xAD.shape[-1]):#self.seq_length
            x2 = torch.squeeze(xAD[..., i]).unsqueeze(1)
            t = torch.squeeze(T_ds[..., i]).unsqueeze(1)
            x2 = torch.cat((x2,t), dim=1)
            Hidden_State_2, Cell_State_2 = self.step(x2, Hidden_State_2, Cell_State_2,
                                                     predefined_A, md2, 'AD', 'encoder', idx, i)
        Hidden_State.append(Hidden_State_2)
        Cell_State.append(Cell_State_2)
                

        Hidden_State = self.idwtl_layer(Hidden_State)
        Cell_State =  self.idwtl_layer(Cell_State)
        
        if task_level <= 6.0: 
            Hidden_State = 0.3 * self.camH(Hidden_State) + 0.7 * self.aggH(Hidden_State_1, Hidden_State_2)
            Cell_State = 0.3 * self.camC(Cell_State) + 0.7 * self.aggC(Cell_State_1, Cell_State_2)

        else: 
            Hidden_State = 0.7 * self.camH(Hidden_State) + 0.3 * self.aggH(Hidden_State_1, Hidden_State_2)
            Cell_State = 0.7 * self.camC(Cell_State) + 0.3 * self.aggC(Cell_State_1, Cell_State_2)


        go_symbol = torch.zeros((batch_size, self.output_dim, self.num_nodes), device=self.device)

        timeofday = ycl[:, 1:, :, :] #是什么

        decoder_input = go_symbol

        outputs_final = []

        for i in range(task_level):
            try:
                decoder_input = torch.cat([decoder_input, timeofday[..., i]], dim=1)
            except:
                print(decoder_input.shape, timeofday.shape)
                sys.exit(0)
            Hidden_State, Cell_State = self.step(decoder_input, Hidden_State, Cell_State,
                                                 predefined_A, predefined_A[0].unsqueeze(0).repeat(64, 1, 1), None, 'decoder', idx, None)

            Hidden_State, Cell_State = Hidden_State.view(-1, self.hidden_size_de), Cell_State.view(
                -1, self.hidden_size_de)

            decoder_output = self.fc_final(Hidden_State)

            decoder_input = decoder_output.view(batch_size, self.num_nodes,
                                                self.output_dim).transpose(
                                                    1, 2)
            outputs_final.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = ycl[:, :1, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)

        outputs_final = outputs_final.view(batch_size, self.num_nodes,
                                           task_level,
                                           self.output_dim).transpose(1, 2)
        #print("test output: ", Hidden_State_1[0,:,:].transpose(0, 1).clone().detach().cpu().numpy().sum())
        return outputs_final, md1, md2, Hidden_State_1[0,:,:].transpose(0, 1).clone().detach().cpu().numpy(), Hidden_State_2[0,:,:].transpose(0, 1).clone().detach().cpu().numpy()

    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self.device))

            nn.init.orthogonal(Hidden_State)
            nn.init.orthogonal(Cell_State)

            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

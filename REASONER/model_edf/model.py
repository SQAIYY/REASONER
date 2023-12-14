import torch.nn.functional as F
import torch
import torch.nn as nn
#from transformer import TransformerEncoder
import copy
from copy import deepcopy
import math
import numpy as np
from torch.autograd import Variable

class MultiEncoder(nn.Module):
    def __init__(self):
        super(MultiEncoder, self).__init__()
        # Define Conv, SepConv
        self.GELU = nn.GELU()
        self.features_EEG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=25, stride=3, bias=False, padding=12),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),
        )
        self.features_EOG = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.Conv1d(64, 64, kernel_size=25, stride=3, bias=False, padding=12),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),
            # nn.Dropout(0.1)
        )

    def forward(self, x_EEG, x_EOG):
        #########################################
        b, s, c, d = x_EEG.shape
        x_EEG = x_EEG.contiguous().view(-1, c, d)
        x_EOG = x_EOG.contiguous().view(-1, c, d)
        x_EEG = self.features_EEG(x_EEG)
        x_EOG = self.features_EOG(x_EOG)
        x_EEG_logits = x_EEG
        x_EOG_logits = x_EOG

        return x_EEG_logits, x_EOG_logits
class MultiDecoder(nn.Module):
    def __init__(self):
        super(MultiDecoder, self).__init__()
        # Define Conv, SepConv

        sepconvtrans_same = lambda in_f, out_f, kernel, s=None, pad=None: nn.Sequential(
            nn.ConvTranspose1d(in_f, out_f, kernel, s, pad), nn.BatchNorm1d(out_f), nn.ReLU())
        self.GELU = nn.GELU()

        self.sepconv_trans_15 = sepconvtrans_same(128, 128, 8, 4, 1)
        self.sepconv_trans_16 = sepconvtrans_same(128, 64, 8, 1, 4)
        self.sepconv_trans_17 = sepconvtrans_same(64, 64, 25, 6, 4)
        self.trans_18 = nn.ConvTranspose1d(64, 1, 50, stride=6, padding=13)
        self.sigmoid_18 = nn.Sigmoid()

        # EOG Decoder
        self.sepconv_trans_25 = sepconvtrans_same(128, 128, 8, 4, 1)
        self.sepconv_trans_26 = sepconvtrans_same(128, 64, 8, 1, 4)
        self.sepconv_trans_27 = sepconvtrans_same(64, 64, 25, 6, 4)
        self.trans_28 = nn.ConvTranspose1d(64, 1, 50, stride=6, padding=13)
        self.sigmoid_28 = nn.Sigmoid()

    def forward(self, x_EEG, x_EOG):
        ##################

        bs, c, d = x_EEG.shape

        x_EEG = self.sepconv_trans_15(x_EEG)

        x_EEG = self.sepconv_trans_16(x_EEG)

        x_EEG = self.sepconv_trans_17(x_EEG)

        x_EEG = self.trans_18(x_EEG)

        x_EEG = self.sigmoid_18(x_EEG)

        x_EOG = self.sepconv_trans_25(x_EOG)
        x_EOG = self.sepconv_trans_26(x_EOG)
        x_EOG = self.sepconv_trans_27(x_EOG)
        x_EOG = self.trans_28(x_EOG)
        x_EOG = self.sigmoid_28(x_EOG)
        bs, c1, d1 = x_EOG.shape
        x_EEG  = x_EEG .contiguous().view(-1, 25, c1, d1)
        x_EOG = x_EOG.contiguous().view(-1, 25, c1, d1)
        return x_EEG, x_EOG
class MI_Projector(nn.Module):
    def __init__(self, ecoder_dim1, ecoder_dim2):
        super(MI_Projector, self).__init__()

        self.fc = nn.Linear(ecoder_dim1, ecoder_dim2)
        self.BN = nn.BatchNorm1d(ecoder_dim2)
        self.Soft = nn.Softmax(dim=1)
    def forward(self, x):
        ##################
        bs, c, d = x.shape
        x = x.contiguous().view(-1,c*d)
        x = self.fc(x)
        x = self.BN(x)
        x = self.Soft(x)
        return x
class MKLD_Projector(nn.Module):
    def __init__(self, ecoder_dim1, ecoder_dim2):
        super(MKLD_Projector, self).__init__()

        self.fc = nn.Linear(ecoder_dim1, ecoder_dim2)
        self.BN = nn.BatchNorm1d(25)
        self.relu = nn.ReLU()
    def forward(self, x):
        ##################
        bs, c, d = x.shape
        x = x.contiguous().view(-1, 25,  c * d)
        x = self.fc(x)
        x = self.BN(x)
        x = self.relu(x)
        return x
##########################################################################################
class Prediction(nn.Module):
    def __init__(self, ecoder_dim1, ecoder_dim2):
        super(Prediction, self).__init__()

        self.conv_e = nn.Conv1d(ecoder_dim1, ecoder_dim2, kernel_size=3, stride=1,padding=1)
        self.BN_e = nn.BatchNorm1d(ecoder_dim2)
        self.relu_e = nn.GELU()
        self.conv_d = nn.ConvTranspose1d(ecoder_dim2, ecoder_dim1,kernel_size=3,stride=1,padding=1)
        self.BN_d = nn.BatchNorm1d(ecoder_dim1)
        self.relu_d = nn.GELU()
    def forward(self, x):
        ##################
        bs, c, d = x.shape
        #print(x.shape)
        #x = x.contiguous().view(-1, d)
        x = self.conv_e(x)
        x = self.BN_e(x)
        x = self.relu_e(x)

        x = self.conv_d(x)
        x = self.BN_d(x)
        x = self.relu_d(x)

        #x = x.contiguous().view(-1, s, d)
        return x

##########################################################################################

def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        #print(value.shape)
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Cross_TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer):
        super(Cross_TCE, self).__init__()
        self.layer = layer
        #self.norm = LayerNorm(layer.size)

    def forward(self, x_in, x_in_k, x_in_v):
        #for layer in self.layers:
        x = self.layer(x_in, x_in_k, x_in_v)
        return x
class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)
        self.alpha = nn.Parameter(torch.rand(1))


    def forward(self, x_in, x_in_k, x_in_v):
        "Transformer Encoder"
        bs, c, d = x_in.shape
        x_in = x_in.permute(0,2,1)
        x_in_k =  x_in_k.permute(0,2,1)
        x_in_v = x_in_v.permute(0,2,1)
        query = self.conv(x_in)
        x = self.self_attn(query, x_in_k, x_in_v)  # Encoder self-attention
        #print(self.alpha)

        return self.alpha * x_in_v + (1 - self.alpha) * x

##########################################################################################
class TCE(nn.Module):
    def __init__(self, dim1, dim2):
        super(TCE, self).__init__()
        self.biGRU = nn.GRU(dim1, dim2, num_layers=2, dropout=0.5, bidirectional=True)

        #self.norm = LayerNorm(layer.size)
        self.dropout_1 = nn.Dropout()

    def forward(self, x):
        bs, d, c = x.shape
        x = x.contiguous().view(-1, 25, c * d)
        x, _ = self.self.biGRU(x.transpose(0, 1))
        x = self.dropout_1(x.transpose(0, 1))
        return x

class CLS(nn.Module):
    def __init__(self, dim1, dim2, clas):
        super(CLS, self).__init__()
        self.fc = nn.Linear(dim1, dim2)
        self.relu = nn.ReLU()
        self.cls = nn.Linear(dim2, clas)


    def forward(self, x):
        b, s, d = x.shape
        x = x.contiguous().view(-1, d)
        x = self.fc(x)
        x = self.relu(x)
        x = self.cls(x)
        x = x.contiguous().view(b, s, -1)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.MultiEncoder = MultiEncoder()
        self.projection_MI1 = MI_Projector(128 * 20, 128)  # 互信息特征一致性
        self.projection_MI2 = MI_Projector(128 * 20, 128)  # 互信息特征一致性
        self.projection_MKLD1 = MKLD_Projector(128 * 20, 128)  # MKLD上下文一致性
        self.projection_MKLD2 = MKLD_Projector(128 * 20, 128)  # MKLD上下文一致性
        #self.Imputation = Imputation(128, 128)
        self.imputation1 = Prediction(128, 64)
        self.imputation2 = Prediction(128, 64)
        attn = MultiHeadedAttention(8, 128, 20)
        self.cross_tce1 = Cross_TCE(EncoderLayer(128, deepcopy(attn), 20, 0.5))
        self.cross_tce2 = Cross_TCE(EncoderLayer(128, deepcopy(attn), 20, 0.5))
        self.tce = TCE(256*20, 128*20)
        self.classfier = CLS(256*20, 256, 5)
        self.MultiDecoder = MultiDecoder()

if __name__ == '__main__':
    device = torch.device('cuda')
    model = MultiEncoder().to(device)
    projection1 = MI_Projector(128 * 20, 128).to(device)#互信息特征一致性
    projection2 = MI_Projector(128 * 20, 128).to(device)#互信息特征一致性

    projection3 = MKLD_Projector(128*20, 128).to(device)#MKLD上下文一致性
    projection4 = MKLD_Projector(128*20, 128).to(device)#MKLD上下文一致性

    imputation1 = Prediction(128, 64).to(device)
    imputation2 = Prediction(128, 64).to(device)
    attn = MultiHeadedAttention(8, 128*20, 25).to(device)
    tce1 = Cross_TCE(EncoderLayer(128*20, deepcopy(attn), 25, 0.5)).to(device)
    tce2 = Cross_TCE(EncoderLayer(128*20, deepcopy(attn), 25, 0.5)).to(device)
    #tce3 = TCE(256, 128).to(device)
    classfier = CLS(256*20, 256, 20).to(device)
    model1 = MultiDecoder().to(device)

    # model1 = MultiAutoencoder()
    x1 = torch.randn((2, 25, 1, 3000)).to(device)
    x2 = torch.randn((2, 25, 1, 3000)).to(device)
    x11 = torch.randn((2, 25, 128)).to(device)
    x22 = torch.randn((2, 25, 128)).to(device)
    mask_EEG = torch.zeros((50)).unsqueeze(-1).to(device)
    mask_EOG = torch.ones((50)).unsqueeze(-1).to(device)

    out1, out2 = model(x1, x2)
    out11 = projection1(out1)
    out22 = projection2(out2)
    out111 = projection3(out1)
    out222 = projection4(out2)
    out3 = imputation1(out1)
    out4 = imputation2(out2)
    out5 = tce1(out3, out4, out4)
    out6 = tce2(out4, out3, out3)
    out_cat = torch.cat([out5, out6], dim=2)
    # out_cat_T = tce3(out_cat)
    out_cla = classfier(out_cat)
    # #
    out7, out8 = model1(out3, out4)
    

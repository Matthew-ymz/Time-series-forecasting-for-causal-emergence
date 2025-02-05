import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_NN
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_NN()
        self.fc1 = nn.Linear(configs.c_in * configs.seq_len, configs.d_model)
        self.fc2 = nn.Linear(configs.d_model, configs.d_model)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        # Decoder
        if self.task_name == 'nn_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len * configs.c_out, bias=True)
        

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        B, T, N = x_enc.shape

        # Embedding
        enc_0 = self.enc_embedding(x_enc)
        enc_0 = self.fc1(enc_0)
        enc_out = self.dropout(enc_0)
        enc_out = self.relu(enc_out)
        enc_out = self.fc2(enc_out)
        enc_out = self.dropout(enc_out)
        enc_out = self.relu(enc_out) + enc_0
        dec_out = self.projection(enc_out) 
        dec_out = dec_out.reshape(B, self.pred_len, N)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out.reshape(B, self.pred_len, N)

    def forward(self, x_enc, mask=None):
        if self.task_name == 'nn_forecast':
            dec_out = self.forecast(x_enc)
            result = dec_out
        return result

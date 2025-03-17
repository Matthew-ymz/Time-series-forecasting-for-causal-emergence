import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_NN
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.cov_bool = configs.cov_bool
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_size = configs.pred_len * configs.c_out
        self.output_attention = configs.output_attention
        self.c_in = configs.c_in 
        self.seq_len = configs.seq_len
        # Embedding
        self.enc_embedding = DataEmbedding_NN()
        self.fc1 = nn.Linear(self.c_in * self.seq_len, configs.d_model)
        self.fc2 = nn.Linear(configs.d_model, configs.d_model)
        self.fc_L = nn.Linear(configs.d_model, self.output_size) 
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.func = lambda x: self.forecast(x)[0]
        # Decoder
        if self.task_name == 'nn_forecast':
            self.projection = nn.Linear(configs.d_model, self.output_size, bias=True)

    def cal_EI_1(self, x_enc, num_samples=1000, L=1):
        jac_in = L * (2 * torch.rand(num_samples, self.seq_len, self.c_in, dtype=x_enc.dtype, device=x_enc.device) - 1)
        jacobian_matrix = jacobian(self.func, jac_in)
        diag_matrices = jacobian_matrix.permute(0, 3, 1, 2, 4, 5).diagonal(dim1=0, dim2=1)
        diag_matrices = diag_matrices.permute(4, 0, 1, 2, 3).reshape(-1, self.seq_len * self.c_in, self.seq_len * self.c_in)
        diag_matrices = diag_matrices.cpu()
        det_list = torch.linalg.det(diag_matrices).to(device=x_enc.device)
        mask = det_list == 0
        count = mask.sum().item()
        det_list[mask] = 1  # 避免在 log 中计算 0
        avg_log_jacobian = torch.log(det_list.abs()).mean()
        # else:
        #     count = 0
        #     avg_log_jacobian = 0
        return count, avg_log_jacobian

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc = x_enc / stdev

        B, T, N = x_enc.shape

        # Embedding
        enc_0 = self.enc_embedding(x_enc)
        enc_0 = self.fc1(enc_0)
        enc_out = self.dropout(enc_0)
        enc_out = self.relu(enc_out)
        enc_out = self.fc2(enc_out)
        enc_out = self.dropout(enc_out)
        enc_out = self.relu(enc_out) 
        mu = self.projection(enc_out)
        if self.cov_bool:
            L_elements = self.fc_L(enc_out)  # Cholesky 分解的下三角部分         
            L = torch.diag_embed(L_elements).abs()
        else:
            L = 0
        dec_out = mu.reshape(B, self.pred_len, N)
        # print("!!!!!!!!!!")
        # print(dec_out)
        # print(L)
        return dec_out, L

    def forward(self, x_enc, EI_bool=False):
        if self.task_name == 'nn_forecast':
            dec_out, L = self.forecast(x_enc)
            result = dec_out

        if EI_bool:
            count, avg_log_jacobian = self.cal_EI_1(x_enc)
            h_t = x_enc.reshape(-1,x_enc.size(1)*x_enc.size(2))
            h_t1_hat = result.reshape(-1,result.size(1)*result.size(2))
            ei_items = {"L":L,
                    "h_t": h_t,
                    "h_t1_hat": h_t1_hat,
                    "avg_log_jacobian": avg_log_jacobian,
                    "count": count}
        else:
            ei_items = {"L":L}
        return result, ei_items

    

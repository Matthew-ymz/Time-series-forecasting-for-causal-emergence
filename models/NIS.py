import torch
from torch import nn
from torch import distributions
from torch.autograd.functional import jacobian
from layers.Embed import DataEmbedding_NN

class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask):
        super(InvertibleNN, self).__init__()
        
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = mask.size(0) // 2
        self.t = nn.ModuleList([nett() for _ in range(length)])
        self.s = nn.ModuleList([nets() for _ in range(length)])
        self.size = mask.size(1)
    
    def g(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0])
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    

class Model(nn.Module):
    def __init__(self, configs):

        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_size = configs.c_in * configs.seq_len
        self.hidden_units = configs.d_model
        self.B = configs.batch_size
        self.T = configs.seq_len
        self.N = configs.c_in
        self.dropout = nn.Dropout(p=0.1)
        self.enc_embedding = DataEmbedding_NN()
        if self.input_size % 2 != 0:
            self.input_size += 1
        
        self.latent_size = configs.latent_size
        self.output_size = configs.c_out
        self.pi = torch.tensor(torch.pi)
        self.func = lambda x: (self.dynamics(x) + x)
        self.enc_embedding = DataEmbedding_NN()
        self.gpu_type = configs.gpu_type

        nets = lambda: nn.Sequential(
            nn.Linear(self.input_size, self.hidden_units),
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.input_size),
            nn.Tanh()
        )
        
        nett = lambda: nn.Sequential(
            nn.Linear(self.input_size, self.hidden_units),
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.input_size)
        )
        
        self.dynamics = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.latent_size)
        )

        mask1 = torch.cat((torch.zeros(1, self.input_size // 2), torch.ones(1, self.input_size // 2)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        #mark0
        self.flow = InvertibleNN(nets, nett, masks)
        self.is_normalized = True
        
    def encoding(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        self.B, self.T, self.N = x_enc.shape
        x_enc = self.enc_embedding(x_enc)
        h, _ = self.flow.f(x_enc)
        return h[:, :self.latent_size], stdev, means
    
    def decoding(self, h_t1, stdev, means):
        sz = self.input_size - self.latent_size
        if sz > 0:
            if self.gpu_type == 'cuda':
                means_n = torch.zeros(sz, dtype=h_t1.dtype,device=h_t1.device)
                covs = torch.eye(sz, dtype=h_t1.dtype,device=h_t1.device)
                noise = distributions.MultivariateNormal(means_n, covs).sample((h_t1.size(0),))
            elif self.gpu_type == 'mps':
                means_n = torch.zeros(sz, dtype=h_t1.dtype)
                covs = torch.eye(sz, dtype=h_t1.dtype)
                noise = distributions.MultivariateNormal(means_n, covs).sample((h_t1.size(0),))
                noise = noise.to(device=h_t1.device)
            h_t1 = torch.cat((h_t1, noise), dim=1)
        x_t1_hat, _ = self.flow.g(h_t1)

        x_t1_hat = x_t1_hat.reshape(self.B, self.T, self.N)
        # De-Normalization from Non-stationary Transformer
        x_t1_hat = x_t1_hat * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.T, 1))
        x_t1_hat = x_t1_hat + (means[:, 0, :].unsqueeze(1).repeat(1, self.T, 1))
        return x_t1_hat.reshape(self.B, self.T, self.N)
    
    def cal_EI_1(self, x_enc, num_samples=1000, L=1):
        jac_in = L * (2 * torch.rand(num_samples, self.latent_size, dtype=x_enc.dtype, device=x_enc.device) - 1)
        jacobian_matrix = jacobian(self.func, jac_in)
        diag_matrices = jacobian_matrix.permute(0, 2, 1, 3).diagonal(dim1=0, dim2=1)
        diag_matrices = diag_matrices.permute(2, 0, 1)
        
        if self.gpu_type == 'cuda':
            det_list = torch.linalg.det(diag_matrices)
        elif self.gpu_type == 'mps':
            diag_matrices = diag_matrices.cpu()
            det_list = torch.linalg.det(diag_matrices).to(device=x_enc.device)
        mask = det_list == 0
        count = mask.sum().item()
        det_list[mask] = 1  # 避免在 log 中计算 0
        avg_log_jacobian = torch.log(det_list.abs()).mean()
        return count, avg_log_jacobian
    
    def forward(self, x_t, EI_bool=False, L=1, num_samples=1000):
        h_t, stdev, means = self.encoding(x_t)
       
        h_t1_hat = self.dynamics(h_t) + h_t
        
        if self.is_normalized:
            h_t1_hat = torch.tanh(h_t1_hat)
        
        x_t1_hat = self.decoding(h_t1_hat, stdev, means)

        if EI_bool:
            count, avg_log_jacobian = self.cal_EI_1(h_t, num_samples, L)
            ei_items = {"h_t": h_t,
                    "h_t1_hat": h_t1_hat,
                    "avg_log_jacobian": avg_log_jacobian,
                    "count": count}
        else:
            ei_items = {}
        
        return x_t1_hat, ei_items



import torch
from torch import nn
from models.NIS import Model


class Modelp(Model):
    def __init__(self, configs):
        super().__init__(configs)
        self.inv_dynamics = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units, self.latent_size)
        )
        
    def back_forward(self, x_t1):
        h_t1 = self.encoding(x_t1)
        h_t_hat = self.inv_dynamics(h_t1) + h_t1
        return h_t_hat



   
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


class RNISNet(Model):
    def __init__(self, 
                 input_size: int = 4, 
                 latent_size: int = 2, 
                 output_size: int = 4, 
                 hidden_units: int = 64, 
                 hidden_units_dyn: int = 64,
                 is_normalized: bool = True
                ) -> None:
        super().__init__(input_size, latent_size, output_size, hidden_units, is_normalized)

        self.func = lambda x: (self.dynamics.f(x)[0])


        nets_dyn = lambda: nn.Sequential(
            nn.Linear(latent_size, hidden_units_dyn),
            nn.LeakyReLU(), 
            nn.Linear(hidden_units_dyn, hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units_dyn, latent_size),
            nn.Tanh()
        )

        nett_dyn = lambda: nn.Sequential(
            nn.Linear(latent_size, hidden_units_dyn),
            nn.LeakyReLU(), 
            nn.Linear(hidden_units_dyn, hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units_dyn, latent_size)
        )

        mask1 = torch.cat((torch.zeros(1, latent_size // 2), torch.ones(1, latent_size // 2)), 1)
        mask2 = 1 - mask1
        masks_dyn = torch.cat((mask1, mask2, mask1, mask2), 0)
        self.dynamics = InvertibleNN(nets_dyn, nett_dyn, masks_dyn)
        
    def forward(self, x_t, x_t1, L=1, num_samples=1000):
        h_t = self.encoding(x_t)

        count, avg_log_jacobian = self.cal_EI_1(h_t, num_samples, L)
        
        h_t1 = self.encoding(x_t1)
        #mark1
        h_t1_hat, _ = self.dynamics.f(h_t)
        
        x_t1_hat = self.decoding(h_t1_hat)
        
        ei_items = {"h_t": h_t,
                    "h_t1": h_t1,
                    "h_t1_hat": h_t1_hat,
                    "avg_log_jacobian": avg_log_jacobian,
                    "count": count}
        
        return x_t1_hat, ei_items

    def back_forward(self, x_t1):
        h_t1 = self.encoding(x_t1)
        #mark2
        h_t_hat, _ = self.dynamics.g(h_t1)
        return h_t_hat
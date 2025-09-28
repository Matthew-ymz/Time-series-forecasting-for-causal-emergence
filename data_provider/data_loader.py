import os
import numpy as np
import pandas as pd
import scipy
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')


class SIRModel(Dataset):
    def __init__(self, path, data_path, size_list, beta, gamma, steps, dt, interval, sigma, rho, flag, use_cache=True):
        """
        Initialize the SIR model dataset.
        
        :param size_list: List of initial state sizes.
        :param beta: Infection rate.
        :param gamma: Recovery rate.
        :param steps: Number of steps to run (including the starting point).
        :param dt: Step size.
        :param interval: Sampling interval.
        :param sigma: Standard deviation of noise.
        :param rho: Correlation coefficient of noise.
        """
        if flag == "train" or flag == "testall":
            self.size_list = size_list
            self.steps = steps
        else:
            self.size_list = [1000]
            self.steps = 2
        self.beta, self.gamma = beta, gamma
        self.sigma, self.rho = sigma, rho
        self.path = path+flag+f"_{sum(self.size_list) * self.steps}_{self.sigma}"
        self.data_path = data_path
        self.dt = dt
        self.interval = interval
        self.scale = False
        self.init_total_number = np.sum(self.size_list)

        #self.data = self.simulate_multiseries(size_list)
        self.prior = multivariate_normal(mean=np.zeros(2), cov=np.array([[1, rho], [rho, 1]]))
        if 'macro' in self.data_path:
            loaded_data_dict = np.load(path+self.data_path, allow_pickle=True).item()
            self.input = loaded_data_dict['input']
            self.output = loaded_data_dict['output']

        elif use_cache and os.path.isfile(self.path):
            loaded_data_dict = np.load(self.path, allow_pickle=True).item()
            self.input = loaded_data_dict['input']
            self.output = loaded_data_dict['output']

        else:
            self.input, self.output = self._simulate_multiseries()
            data_dict = {
                'input': self.input,
                'output': self.output,
            }

            np.save(self.path, data_dict)

    def perturb(self, S, I):
        """
        Add observational noise to the macro states S and I.
        
        :param S: Susceptible population.
        :param I: Infected population.
        :return: Observed states with noise.
        """
        noise_S = self.prior.rvs(size=1) * self.sigma
        noise_I = self.prior.rvs(size=1) * self.sigma
        S_obs0 = np.expand_dims(S + noise_S[0], axis=0)
        S_obs1 = np.expand_dims(S + noise_S[1], axis=0)
        I_obs0 = np.expand_dims(I + noise_I[0], axis=0)
        I_obs1 = np.expand_dims(I + noise_I[1], axis=0)
        SI_obs = np.concatenate((S_obs0, I_obs0, S_obs1, I_obs1), 0)
        return SI_obs
    
    def simulate_oneserie(self, S, I):
        """
        Simulate a single time series from a specific starting point.
        
        :param S: Initial susceptible population (as a ratio).
        :param I: Initial infected population (as a ratio).
        :return: Time series data.
        """
        sir_data = []
        for k in range(self.steps):
            if k % self.interval == 0:
                SI_obs = self.perturb(S, I)
                sir_data.append(SI_obs)
                
            new_infected = self.beta * S * I 
            new_recovered = self.gamma * I
            S = S - new_infected * self.dt
            I = I + (new_infected - new_recovered) * self.dt
        return np.array(sir_data)

    def _simulate_multiseries(self):
        """
        Simulate multiple time series from various starting points to create the main dataset.
        
        :return: sir_input and sir_output arrays.
        """
        num_obs = int(self.steps / self.interval)
        sir_data_all = np.zeros([self.init_total_number, num_obs, 4])
        num_strip = len(self.size_list)
        frac = 1 / num_strip
        
        for strip in range(num_strip):
            sir_data_part = np.zeros([self.size_list[strip], num_obs, 4])
            boundary_left = strip * frac
            boundary_right = boundary_left + frac
            S_init = np.random.uniform(boundary_left, boundary_right, self.size_list[strip])
            I_init = []
            while len(I_init) < self.size_list[strip]:
                I = np.random.rand(1)[0]
                S = S_init[len(I_init)]
                if S + I <= 1:
                    sir_data_part[len(I_init),:,:] = self.simulate_oneserie(S, I)
                    I_init.append(I)
            size_list_cum = np.cumsum(self.size_list)
            size_list_cum = np.concatenate([[0], size_list_cum])
            sir_data_all[size_list_cum[strip]:size_list_cum[strip+1], :, :] = sir_data_part
        sir_input, sir_output = self.reshape(sir_data_all = sir_data_all)
        return sir_input, sir_output

    def reshape(self, sir_data_all):
        """
        Reshape the generated multi-time series into input and output arrays.
        
        :param sir_data_all: Array of all time series data.
        :return: sir_input and sir_output arrays.
        """
        sir_input = sir_data_all[:, :-1, :].reshape(-1, 1, 4)
        sir_output = sir_data_all[:, 1:, :].reshape(-1, 1, 4)
        return sir_input, sir_output

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.input)

    def __getitem__(self, idx):
        """
        Return an item from the dataset.
        
        :param idx: Index of the item.
        :return: A tuple of torch.Tensor representing the input and output.
        """
        return idx, torch.tensor(self.input[idx], dtype=torch.float), torch.tensor(self.output[idx], dtype=torch.float)


class Dataset_Ca2p(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Ca2p.csv', fold_loc='normal', data_partition = [0.7,0.1,0.2],
                 target='OT', scale=True, downsample=1, timeenc=0, freq='h', seasonal_patterns=None):

        assert size != None, "You must specify the size of the dataset"
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val', 'testall']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'testall': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.downsample = downsample
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.fold_loc = fold_loc
        self.data_partition = data_partition 
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        ind = cols[0]
        cols.remove(ind)
#         df_raw = df_raw[[ind] + cols]
        if self.target in cols:
            cols.remove(self.target)
        ds_len = (len(df_raw) // self.downsample) - 1
        num_train = int(ds_len * self.data_partition[0])
        num_test = int(ds_len * self.data_partition[1])
        num_vali = int(ds_len * self.data_partition[2])
        print(self.fold_loc)
        if self.fold_loc == 'normal':
            border1s = [0, num_train - self.seq_len, ds_len - num_test - self.seq_len, 0]
            border2s = [num_train, num_train + num_vali, ds_len, ds_len]
        elif self.fold_loc == 'vali_first':
            border1s = [num_vali - self.seq_len, 0,        ds_len - num_test - self.seq_len, 0]
            border2s = [num_vali + num_train,    num_vali, ds_len,                           ds_len]
        elif self.fold_loc == 'vali_test_first':
            border1s = [ds_len - num_train - self.seq_len, 0,        num_vali - self.seq_len, 0]
            border2s = [ds_len,                            num_vali, num_vali + num_test,     ds_len]
        else: 
            print("Error for train_vali_test")
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        data_x = []
        step = self.downsample
        for i in range(step):
            data_x.append(data[border1 * step + i : border2 * step + i : step])

        self.data_x = np.array(data_x)
        
    def __getitem__(self, index):
        offset = index % self.downsample
        s_begin = index // self.downsample
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[offset, s_begin:s_end]
        seq_y = self.data_x[offset, r_begin:r_end]

        return index, seq_x, seq_y

    def __len__(self):
        return (self.data_x.shape[1] - self.seq_len - self.pred_len + 1) * self.downsample

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_couzin(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='swarm_50_2000.csv', fold_loc='normal', data_partition = [0.7,0.1,0.2], scale=False, downsample=1):

        assert size != None, "You must specify the size of the dataset"
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val', 'testall']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'testall': 3}
        self.set_type = type_map[flag]
        self.scale = scale
        self.downsample = downsample

        self.root_path = root_path
        self.data_path = data_path
        self.fold_loc = fold_loc
        self.data_partition = data_partition 
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        ind = cols[0]
        cols.remove(ind)
        ds_len = (len(df_raw) // self.downsample) - 1
        num_train = int(ds_len * self.data_partition[0])
        num_test = int(ds_len * self.data_partition[1])
        num_vali = int(ds_len * self.data_partition[2])
        print(self.fold_loc)
        if self.fold_loc == 'normal':
            border1s = [0, num_train - self.seq_len, ds_len - num_test - self.seq_len, 0]
            border2s = [num_train, num_train + num_vali, ds_len, ds_len]
        elif self.fold_loc == 'vali_first':
            border1s = [num_vali - self.seq_len, 0,        ds_len - num_test - self.seq_len, 0]
            border2s = [num_vali + num_train,    num_vali, ds_len,                           ds_len]
        elif self.fold_loc == 'vali_test_first':
            border1s = [ds_len - num_train - self.seq_len, 0,        num_vali - self.seq_len, 0]
            border2s = [ds_len,                            num_vali, num_vali + num_test,     ds_len]
        else: 
            print("Error for train_vali_test")
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        print("scale:")
        print(self.scale)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        data_x = []
        step = self.downsample
        for i in range(step):
            data_x.append(data[border1 * step + i : border2 * step + i : step])

        self.data_x = np.array(data_x)
        
    def __getitem__(self, index):
        offset = index % self.downsample
        s_begin = index // self.downsample
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[offset, s_begin:s_end]
        seq_y = self.data_x[offset, r_begin:r_end]

        return index, seq_x, seq_y

    def __len__(self):
        return (self.data_x.shape[1] - self.seq_len - self.pred_len + 1) * self.downsample

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class KuramotoModel_Orderparameter(Dataset):
    def __init__(self, path, sz, groups, batch_size, time_steps, dt, sample_interval, 
                 coupling_strength, noise_level, flag, use_cache=True):
        """
        Initialize the Kuramoto model dataset.
        
        :param sz: Total number of oscillators
        :param groups: Number of groups/clusters
        :param batch_size: Number of trajectories to generate
        :param time_steps: Total simulation steps
        :param dt: Integration time step
        :param sample_interval: Sampling interval for storing data
        :param coupling_strength: Coupling strength parameter
        :param noise_level: Noise intensity
        :param flag: "train", "val", or "test" flag
        """
        self.path = path + flag + f"_sz{sz}_g{groups}_bs{batch_size}"
        self.sz = sz
        self.groups = groups
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dt = dt
        self.sample_interval = sample_interval
        self.coupling_strength = coupling_strength
        self.noise_level = noise_level
        self.flag = flag
        self.scale = False
        
        # Initialize Kuramoto model parameters
        self.obj_matrix, self.group_matrix, self.omegas = self.initialize_kuramoto(sz, groups)
        
        if use_cache and os.path.isfile(self.path):
            loaded_data_dict = np.load(self.path, allow_pickle=True).item()
            self.input = loaded_data_dict['input']
            self.output = loaded_data_dict['output']
        else:
            self.input, self.output = self._simulate_multiseries()
            data_dict = {
                'input': self.input,
                'output': self.output,
            }
            np.save(self.path, data_dict)

    def initialize_kuramoto(self, sz, groups):
        """Initialize Kuramoto model connectivity matrices."""
        obj_matrix = torch.zeros([sz, sz])
        group_matrix = torch.zeros([sz, groups])
        
        # Create block-diagonal connectivity (oscillators only connect within same group)
        group_size = sz // groups
        for k in range(groups):
            for i in range(group_size):
                for j in range(group_size):
                    if i != j:  # No self-connections
                        obj_matrix[i + k * group_size, j + k * group_size] = 1
        
        # Group membership matrix
        for k in range(groups):
            group_matrix[k * group_size:(k + 1) * group_size, k] = 1
        
        # Natural frequencies
        omegas = torch.randn(sz)
        
        return obj_matrix, group_matrix, omegas

    def one_step(self, thetas):
        """Perform one integration step of Kuramoto model."""
        ii = thetas.unsqueeze(0).repeat(thetas.size(0), 1)
        jj = ii.transpose(0, 1)
        dff = jj - ii
        sindiff = torch.sin(dff)
        mult = self.coupling_strength * self.obj_matrix @ sindiff
        dia = torch.diagonal(mult, 0)
        noise = torch.randn(self.sz) * self.noise_level
        thetas = self.dt * (self.omegas + dia + noise) + thetas
        return thetas

    def compute_order_parameters(self, thetas):
        """Compute order parameters (group-level observables)."""
        cos_ccs = (torch.cos(thetas) @ self.group_matrix) * self.groups / self.sz
        sin_ccs = (torch.sin(thetas) @ self.group_matrix) * self.groups / self.sz
        return torch.cat([cos_ccs, sin_ccs], dim=0)

    def simulate_oneserie(self):
        """Simulate one trajectory of the Kuramoto model."""
        thetas = torch.rand(self.sz) * 2 * np.pi
        kuramoto_data = []
        
        for t in range(self.time_steps):
            thetas = self.one_step(thetas)
            
            if t % self.sample_interval == 0:
                # Compute observables: sine of phases and order parameters
                phase_obs = torch.sin(thetas)
                order_params = self.compute_order_parameters(thetas)
                
                # Combine into observation vector (individual phases + group order parameters)
                obs = torch.cat([phase_obs, order_params])
                kuramoto_data.append(obs.numpy())
        
        return np.array(kuramoto_data)

    def _simulate_multiseries(self):
        """Simulate multiple trajectories."""
        num_obs = int(self.time_steps / self.sample_interval)
        # Output dimension: individual phases (sz) + order parameters (2*groups)
        output_dim = self.sz + 2 * self.groups
        
        kuramoto_data_all = np.zeros([self.batch_size, num_obs, output_dim])
        
        for i in range(self.batch_size):
            kuramoto_data_all[i, :, :] = self.simulate_oneserie()
        
        # Reshape into input-output pairs
        kuramoto_input = kuramoto_data_all[:, :-1, :].reshape(-1, 1, output_dim)
        kuramoto_output = kuramoto_data_all[:, 1:, :].reshape(-1, 1, output_dim)
        
        return kuramoto_input, kuramoto_output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return idx, torch.tensor(self.input[idx], dtype=torch.float), torch.tensor(self.output[idx], dtype=torch.float)

class KuramotoModel(Dataset):
    def __init__(self, path, sz, groups, batch_size, time_steps, dt, sample_interval, 
                 coupling_strength, noise_level, flag, use_cache=True):
        """
        Initialize the Kuramoto model dataset.
        
        :param sz: Total number of oscillators
        :param groups: Number of groups/clusters
        :param batch_size: Number of trajectories to generate
        :param time_steps: Total simulation steps
        :param dt: Integration time step
        :param sample_interval: Sampling interval for storing data
        :param coupling_strength: Coupling strength parameter
        :param noise_level: Noise intensity
        :param flag: "train", "val", or "test" flag
        """
        self.path = path + flag + f"_sz{sz}_g{groups}_bs{batch_size}"
        self.sz = sz
        self.groups = groups
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dt = dt
        self.sample_interval = sample_interval
        self.coupling_strength = coupling_strength
        self.noise_level = noise_level
        self.flag = flag
        self.scale = False
        
        # Initialize Kuramoto model parameters
        self.obj_matrix, self.group_matrix, self.omegas = self.initialize_kuramoto(sz, groups)
        
        if use_cache and os.path.isfile(self.path):
            loaded_data_dict = np.load(self.path, allow_pickle=True).item()
            self.input = loaded_data_dict['input']
            self.output = loaded_data_dict['output']
        else:
            self.input, self.output = self._simulate_multiseries()
            data_dict = {
                'input': self.input,
                'output': self.output,
            }
            np.save(self.path, data_dict)

    def initialize_kuramoto(self, sz, groups):
        """Initialize Kuramoto model connectivity matrices."""
        obj_matrix = torch.zeros([sz, sz])
        group_matrix = torch.zeros([sz, groups])
        
        # Create block-diagonal connectivity (oscillators only connect within same group)
        group_size = sz // groups
        for k in range(groups):
            for i in range(group_size):
                for j in range(group_size):
                    if i != j:  # No self-connections
                        obj_matrix[i + k * group_size, j + k * group_size] = 1
        
        # Group membership matrix
        for k in range(groups):
            group_matrix[k * group_size:(k + 1) * group_size, k] = 1
        
        # Natural frequencies
        omegas = torch.randn(sz)
        
        return obj_matrix, group_matrix, omegas

    def one_step(self, thetas):
        """Perform one integration step of Kuramoto model."""
        ii = thetas.unsqueeze(0).repeat(thetas.size(0), 1)
        jj = ii.transpose(0, 1)
        dff = jj - ii
        sindiff = torch.sin(dff)
        mult = self.coupling_strength * self.obj_matrix @ sindiff
        dia = torch.diagonal(mult, 0)
        noise = torch.randn(self.sz) * self.noise_level
        thetas = self.dt * (self.omegas + dia + noise) + thetas
        return thetas

    def compute_order_parameters(self, thetas):
        """Compute order parameters (group-level observables)."""
        cos_ccs = (torch.cos(thetas) @ self.group_matrix) * self.groups / self.sz
        sin_ccs = (torch.sin(thetas) @ self.group_matrix) * self.groups / self.sz
        return torch.cat([cos_ccs, sin_ccs], dim=0)

    def simulate_oneserie(self):
        """Simulate one trajectory of the Kuramoto model."""
        thetas = torch.rand(self.sz) * 2 * np.pi
        kuramoto_data = []
        
        for t in range(self.time_steps):
            thetas = self.one_step(thetas)
            
            if t % self.sample_interval == 0:
                # Compute observables: sine of phases and order parameters
                phase_obs = torch.sin(thetas)
                order_params = self.compute_order_parameters(thetas)
                
                # Combine into observation vector (individual phases + group order parameters)
                obs = torch.cat([phase_obs, order_params])
                kuramoto_data.append(phase_obs.numpy())
        
        return np.array(kuramoto_data)

    def _simulate_multiseries(self):
        """Simulate multiple trajectories."""
        num_obs = int(self.time_steps / self.sample_interval)
        # Output dimension: individual phases (sz) + order parameters (2*groups)
        output_dim = self.sz
        
        kuramoto_data_all = np.zeros([self.batch_size, num_obs, output_dim])
        
        for i in range(self.batch_size):
            kuramoto_data_all[i, :, :] = self.simulate_oneserie()
        
        # Reshape into input-output pairs
        kuramoto_input = kuramoto_data_all[:, :-1, :].reshape(-1, 1, output_dim)
        kuramoto_output = kuramoto_data_all[:, 1:, :].reshape(-1, 1, output_dim)
        
        return kuramoto_input, kuramoto_output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return idx, torch.tensor(self.input[idx], dtype=torch.float), torch.tensor(self.output[idx], dtype=torch.float)
    

class Dataset_Lorzen(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='data_1000.csv', fold_loc='normal', data_partition = [0.7,0.1,0.2], scale=False, downsample=1):

        assert size != None, "You must specify the size of the dataset"
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val', 'testall']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'testall': 3}
        self.set_type = type_map[flag]
        self.scale = scale
        self.downsample = downsample

        self.root_path = root_path
        self.data_path = data_path
        self.fold_loc = fold_loc
        self.data_partition = data_partition 
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        ind = cols[0]
        cols.remove(ind)
        ds_len = (len(df_raw) // self.downsample) - 1
        num_train = int(ds_len * self.data_partition[0])
        num_test = int(ds_len * self.data_partition[1])
        num_vali = int(ds_len * self.data_partition[2])
        if self.fold_loc == 'normal':
            border1s = [0, num_train - self.seq_len, ds_len - num_test - self.seq_len, 0]
            border2s = [num_train, num_train + num_vali, ds_len, ds_len]
        elif self.fold_loc == 'vali_first':
            border1s = [num_vali - self.seq_len, 0,        ds_len - num_test - self.seq_len, 0]
            border2s = [num_vali + num_train,    num_vali, ds_len,                           ds_len]
        elif self.fold_loc == 'vali_test_first':
            border1s = [ds_len - num_train - self.seq_len, 0,        num_vali - self.seq_len, 0]
            border2s = [ds_len,                            num_vali, num_vali + num_test,     ds_len]
        else: 
            print("Error for train_vali_test")
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # if self.scale:
        #     print(120*"-")
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
        data = df_data.values
        data_x = []
        step = self.downsample
        for i in range(step):
            data_x.append(data[border1 * step + i : border2 * step + i : step])

        self.data_x = np.array(data_x)

    def __getitem__(self, index):   
        offset = index % self.downsample
        s_begin = index // self.downsample
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[offset, s_begin:s_end]
        seq_y = self.data_x[offset, r_begin:r_end]        
        return index, seq_x, seq_y

    def __len__(self):
        return (self.data_x.shape[1] - self.seq_len - self.pred_len + 1) * self.downsample

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Micro_to_Macro(Dataset):
    def __init__(self, path, data, micro_dims, macro_dims, flag):
        self.path = path + data + f"_{micro_dims}_to_{macro_dims}.npy"
        loaded_data_dict = np.load(self.path, allow_pickle=True).item()
        self.input = loaded_data_dict['input']
        self.output = loaded_data_dict['output']
        self.scale = False

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return idx, torch.tensor(self.input[idx], dtype=torch.float), torch.tensor(self.output[idx], dtype=torch.float)

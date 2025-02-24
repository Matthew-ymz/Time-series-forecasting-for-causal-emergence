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
    def __init__(self, path, size_list, beta, gamma, steps, dt, interval, sigma, rho, flag, use_cache=True):
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
        if flag == "train":
            self.size_list = size_list
            self.steps = steps
        else:
            self.size_list = [1000]
            self.steps = 2
        self.path = path+flag+"_"+str(self.size_list)
        self.beta, self.gamma = beta, gamma
        self.sigma, self.rho = sigma, rho
        self.dt = dt
        self.interval = interval
        self.init_total_number = np.sum(self.size_list)

        #self.data = self.simulate_multiseries(size_list)
        self.prior = multivariate_normal(mean=np.zeros(2), cov=np.array([[1, rho], [rho, 1]]))
        #self.__read_data__()
        if use_cache and os.path.isfile(self.path):
            loaded_data_dict = np.load(self.path, allow_pickle=True).item()
            self.sir_input = loaded_data_dict['input']
            self.sir_output = loaded_data_dict['output']

        else:
            self.sir_input, self.sir_output = self._simulate_multiseries()
            data_dict = {
                'input': self.sir_input,
                'output': self.sir_output,
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
        return len(self.sir_input)

    def __getitem__(self, idx):
        """
        Return an item from the dataset.
        
        :param idx: Index of the item.
        :return: A tuple of torch.Tensor representing the input and output.
        """
        return idx, torch.tensor(self.sir_input[idx], dtype=torch.float), torch.tensor(self.sir_output[idx], dtype=torch.float)


class Dataset_Ca2p(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Ca2p.csv', fold_loc=0,
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
        num_train = int(ds_len * 0.7)
        num_test = int(ds_len * 0.2)
        num_vali = ds_len - num_train - num_test
        if self.fold_loc == 0:
            border1s = [0, num_train - self.seq_len, ds_len - num_test - self.seq_len, 0]
            border2s = [num_train, num_train + num_vali, ds_len, ds_len]
        else:
            border1s = [ds_len - num_test - self.seq_len, 0,  num_vali - self.seq_len, 0]
            border2s = [ds_len, num_vali, num_vali + num_test, ds_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            # cols_data = df_raw.columns[1:-1]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            assert self.target in cols
            df_data = df_raw[[self.target]]

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


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


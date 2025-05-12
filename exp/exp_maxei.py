from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.ei import EI
import torch
import torch.nn as nn
from torch import optim
from torch.autograd.functional import jacobian
from torch.func import jacfwd, jacrev
import os
import time
from datetime import datetime
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity

warnings.filterwarnings('ignore')
def kde_density(X):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy()) #bindwidth=0.02
    log_density = kde.score_samples(X.cpu().data.numpy())
    return log_density

class Exp_MaxEI(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_MaxEI, self).__init__(args)
        self.weights = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Modelp(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def update_weight(self, h_t, L=1):
        samples = h_t.size(0)
        scale = h_t.size(1)
        log_density = kde_density(h_t)
        log_rho = - scale * np.log(2.0 * L) 
        logp = log_rho - log_density
        soft = nn.Softmax(dim=0)
        weights = soft(torch.tensor(logp))
        weights = weights * samples
        self.weights = weights.float().to(self.device)

    def reweight(self, dataset):
        inputs = torch.from_numpy(dataset.input).float().to(device=self.device)
        h_t_all = self.model.encoding(inputs)
        self.update_weight(h_t_all)

    def model_step(self, idxs, batch_x, batch_y, criterion, stage_flag=1):
        if stage_flag == 1:
            outputs,_ = self.model(batch_x, EI_bool=False)
            f_dim = 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = criterion(outputs, batch_y)
        elif stage_flag == 2:
            if self.weights is None:
                w = torch.ones(len(idxs)).to(self.device)
            else:
                w = self.weights[idxs]
            outputs,_ = self.model(batch_x, EI_bool=False)
            h_t_hat = self.model.back_forward(batch_y)
            h_t = self.model.encoding(batch_x)
            loss1 = (criterion(outputs, batch_y).mean(dim=(1,2)) * w).mean() 
            loss2 = (criterion(h_t_hat, h_t).mean(dim=1) * w).mean()
            loss = self.args.lambdas * loss1 + loss2
        return loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion2 = nn.L1Loss(reduction='none')#nn.MSELoss(reduction='none')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        if self.args.EI:
            EI_list = []
        for epoch in range(self.args.train_epochs):
            if self.args.EI:
                self.EI = EI()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (idx, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if epoch < self.args.first_stage:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                           loss = self.model_step(idx, batch_x, batch_y, criterion, stage_flag=1)
                    else:
                        loss = self.model_step(idx, batch_x, batch_y, criterion, stage_flag=1)

                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            loss = self.model_step(idx, batch_x, batch_y, criterion2, stage_flag=2)

                    else:
                        loss = self.model_step(idx, batch_x, batch_y, criterion2, stage_flag=2)

                train_loss.append(loss.item())
                if (i + 1) % self.args.prints == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    outputs,ei_items = self.model(batch_x)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            vali_loss, d_EI = self.vali(vali_data, vali_loader, criterion)
            test_loss, d_EI = self.vali(test_data, test_loader, criterion)

            if epoch >= self.args.first_stage:
                self.reweight(train_data)

            if self.args.EI:
                EI_list.append(d_EI.cpu().item())
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} d_EI: {5:.4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, d_EI))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        if self.args.EI:
            folder_path = './results/outputs/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            path1 = folder_path + '/' + "EI.npy"
            np.save(path1, EI_list)
        return self.model

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

warnings.filterwarnings('ignore')

class Exp_Coarse_Graining(Exp_Basic):
    def __init__(self, args):
        super(Exp_Coarse_Graining, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):

        criterion0 = nn.MSELoss()
        
        return criterion0
    
    def model_step(self, idx, batch_x, batch_y, criterion):
        dec_inp = torch.zeros_like(batch_y).float().to(self.device)

        outputs, _ = self.model(batch_x, dec_inp)

        loss = criterion(outputs, batch_y)
        return loss
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (idx, batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.model_step(idx, batch_x, batch_y, criterion)      
                else:
                    loss = self.model_step(idx, batch_x, batch_y, criterion)

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def ig_cg(self, dec_inp, batch_x):
        micro_dims = self.args.c_in
        macro_dims = self.args.c_out
        fun = lambda x: self.model(x, dec_inp)[0]
        ig = IntegratedGradients(fun)
        attribution = np.zeros((micro_dims,macro_dims))
        if self.args.ig_baseline == "mean":
            baseline = batch_x.mean(0).unsqueeze(0) 
        else:
            baseline = torch.zeros_like(batch_x).float().to(self.device)

        for dim in range(macro_dims):
                attributions,_ = ig.attribute(batch_x,target=dim, baselines=baseline, method='gausslegendre', return_convergence_delta=True) 
                attributions = (attributions.abs().cpu().detach().numpy()).mean(0)
                attribution[:, dim] = attributions
        return attribution
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.es_delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (idx, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.model_step(idx, batch_x, batch_y, criterion)
                        train_loss.append(loss.item())
                else:
                    loss = self.model_step(idx, batch_x, batch_y, criterion)
                    train_loss.append(loss.item())

                if (i + 1) % self.args.prints == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def tensor_backward(self, target, source):
        if target.dim() == 0:
            # Base case: target is a scalar
            source.grad = None
            target.backward(retain_graph=True)
            return source.grad
        else:
            print(target.dim())
            # Recursive case: target is a higher-dimensional tensor
            return torch.stack([self.tensor_backward(subtarget, source) for subtarget in target])
        
    def test(self, setting, test=0):
        t0 = time.time()
        test_data, test_loader = self._get_data(flag='testall')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        ig_path = './results/ig_coarse_graining/' + setting + '/'
        if self.args.ig_output and (not os.path.exists(ig_path)):
            os.makedirs(ig_path)

        self.model.eval()

        for i, (idx, batch_x, batch_y) in enumerate(test_loader):
            self.model.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x.requires_grad_()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, :]).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, items = self.model(batch_x, dec_inp)

            else:
                outputs, items = self.model(batch_x, dec_inp)

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
    
            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            # #开始记录雅可比和协方差
            # if (i > self.args.jac_init) and (i <= self.args.jac_end):
            #     store_time = i
            #     batch_list.append(batch_x)
            #     if self.args.causal_net:
            #         batch_x_cat = torch.cat(batch_list, dim=0)
            #         ca_net = self.cal_causal_net(dec_inp, batch_x_cat)
            #         np.save(ig_path + f'ca_{store_time:04}.npy', ca_net)
            #         batch_list = []
                    
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/outputs/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_coarse_graining.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        if self.args.ig_output:
            if self.args.one_serie:
                micro_data = pd.read_csv(self.args.root_path+self.args.data_path)
                micro_data = torch.tensor(micro_data.iloc[:, 1:].values).float().to(self.device)
                macro_data, _ = self.model(micro_data, dec_inp)
                macro_data_np = macro_data.detach().cpu().numpy()
                df_to_save = pd.DataFrame(macro_data_np)
                df_to_save = df_to_save.reset_index()
                save_path = './dataset/' + self.args.data + f"/macro_{self.args.c_out}.csv"
                df_to_save.to_csv(save_path, index=False)
                attribution = self.ig_cg(dec_inp, micro_data)
            else:
                datas = np.load(self.args.root_path+self.args.data_path, allow_pickle=True)
                micro_data = datas.item()
                n_samp = micro_data['input'].shape[0]
                micro_data_input = torch.tensor(micro_data['input']).float().to(self.device)
                micro_data_output = torch.tensor(micro_data['output']).float().to(self.device)
                macro_data_input, _ = self.model(micro_data_input, dec_inp)
                macro_data_output, _ = self.model(micro_data_output, dec_inp)
                data_dict = {
                'input': macro_data_input.detach().cpu().numpy(),
                'output': macro_data_output.detach().cpu().numpy(),
                }
                save_path = './dataset/' + self.args.data + f"/macro_{self.args.c_out}"
                np.save(save_path, data_dict)
                attribution = self.ig_cg(dec_inp, micro_data_output.reshape(n_samp,-1))

            full_path = ig_path + "ig_attribution.png"
            plt.figure(dpi=100)
            sns.heatmap(attribution.T, xticklabels=range(1, self.args.c_in + 1), yticklabels=range(1, self.args.c_out + 1))
            plt.ylabel('macro dim')
            plt.xlabel('micro dim')
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            plt.close()
            
        return
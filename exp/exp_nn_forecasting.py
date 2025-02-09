from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
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

warnings.filterwarnings('ignore')


class Exp_NN_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_NN_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        #flag: train, vali, test
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        d_EI = 0
        with torch.no_grad():
            for i, (idx, batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,ei_items = self.model(batch_x)
                    
                else:
                    outputs,ei_items = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

            x = torch.from_numpy(vali_data.sir_input).float().to(device=self.device)
            y = torch.from_numpy(vali_data.sir_output).float().to(device=self.device)
            if self.args.EI:
                outputs,ei_items = self.model(x, self.args.EI)
                if self.args.model == "NN":
                    h_t1 = y.reshape(-1,y.size(1)*y.size(2))
                else:
                    h_t1 = self.model.encoding(y)
                ei_items['h_t1'] = h_t1
                d_EI, term1, term2 = self.EI(ei_items=ei_items)
                print("term1:",term1.item())
                print("term2:",term2.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, d_EI

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


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,_ = self.model(batch_x, EI_bool=False)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs,_ = self.model(batch_x, EI_bool=False)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 200 == 0:
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

    # Recursive function to iterate over the tensor
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
        folder_path = './results/images/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
#         attention_path = './results/attentions/' + setting + '/'
#         if not os.path.exists(attention_path):
#             os.makedirs(attention_path)
        jacobian_path = './results/jacobian/' + setting + '/'
        if self.args.jacobian and (not os.path.exists(jacobian_path)):
            os.makedirs(jacobian_path)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.eval()
        # with torch.no_grad():
        for i, (idx, batch_x, batch_y) in enumerate(test_loader):
            self.model.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x.requires_grad_()

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs,_ = self.model(batch_x)

            else:
                outputs,_ = self.model(batch_x)

            # s = torch.sum(outputs)
            # if self.args.use_amp:
            #     scaler.scale(s).backward()
            # else:
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
#             if test_data.scale and self.args.inverse:
#                 shape = outputs.shape
#                 outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
#                 batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
    
            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            if (i >= 600) and (i % 12 == 0):
                t = time.time()
                print(f'elapse: {t-t0:.2}s')
                t0 = t
                if self.args.jacobian:
                    fun = lambda x: self.model(x)[0]
                    jac = jacobian(fun, batch_x)
                    jac = jac.detach().cpu().numpy()[0,:,:,0,:,:].astype(np.float16)
                    np.save(jacobian_path + f'jac_{i:04}.npy', jac)
                    print(f'saving jacobian: jac_{i:04}.npy(size: {jac.dtype.itemsize * jac.size // 1024}KB); ')
                    mae, mse, rmse, mape, mspe, msed = metric(pred, true, cor=True)
    
                    np.save(jacobian_path + f'msed_{i:04}.npy', msed)

                if self.args.output_attention and attn is not None:
                    attn = attn.astype(np.float16)
                    np.save(attention_path + f'attn_{i:04}.npy', attn)
                    print(f'saving attention: attn_{i:04}.npy(size: {attn.dtype.itemsize * attn.size // 1024}KB); ')
                input = batch_x.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = input.shape
#                     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                   
                # selecting variable index to output images
                si = 1
                gt = np.concatenate((input[0, :, si], true[0, :, si]), axis=0)
                pd = np.concatenate((input[0, :, si], pred[0, :, si]), axis=0)
                visual(gt, pd, os.path.join(folder_path, f'{i:04}.pdf'))

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
        f = open("result_nn_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'msed.npy', msed)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

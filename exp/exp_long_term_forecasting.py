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
from scipy.linalg import fractional_matrix_power
import torch.distributions as dist 
import os
import time
from datetime import datetime
import warnings
import numpy as np
from captum.attr import IntegratedGradients

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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

    def _select_criterion(self,cov_b=False, lam=0):
        if self.args.freq_loss:
            def criterion0(outputs, batch_y):
                return (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()  
        else:
            criterion0 = nn.MSELoss()

        def nll_loss(mu, L, y):  
            loss1 = criterion0(mu, y)
            #L_diag = torch.exp(torch.diagonal(L, dim1=1, dim2=2))
            #print(L.size())
            # 计算多元高斯分布的负对数似然  
            mvn = dist.MultivariateNormal(loc=mu, scale_tril=L)  
            return (1-lam) * loss1 - lam * mvn.log_prob(y).mean() 
        
        # def nll_loss(mu, L, y):
        #     loss1 = criterion0(mu, y)
        #     diff = y - mu  # 形状: (batch_size, n)
        #     L_diag = torch.diagonal(L, dim1=-2, dim2=-1)  # 形状: (batch_size, n)
        #     z = diff / torch.exp(L_diag)  # 形状: (batch_size, n)
        #     mahalanobis = torch.sum(z**2, dim=-1)  # 形状: (batch_size,)
        #     log_det = 2 * torch.sum(L_diag, dim=-1)  # 形状: (batch_size,)
        #     n = L.shape[-1]  # 维度
        #     log_prob = -0.5 * (n * torch.log(2 * torch.tensor(torch.pi)) + log_det + mahalanobis)  # 形状: (batch_size,)
        #     #print("loss1:{0}  log_prob:{1}".format(loss1.item(),log_prob.mean().item()))
        #     return (1-lam) * loss1 - lam * log_prob.mean()

        if cov_b:
            criterion = nll_loss
        else:
            criterion = criterion0
        return criterion

    def model_step(self, idx, batch_x, batch_y, criterion):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:,:]).float().to(self.device)

        outputs, ei_items = self.model(batch_x, dec_inp)
        if self.args.features[0] == -1:
            outputs = outputs[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        else:
            outputs = outputs[:, -self.args.pred_len:, self.args.features]
            batch_y = batch_y[:, -self.args.pred_len:, self.args.features].to(self.device)
        outputs = outputs.reshape(-1, outputs.size(1)*outputs.size(2))
        batch_y = batch_y.reshape(-1, batch_y.size(1)*batch_y.size(2))
        if self.args.cov_bool:
            loss = criterion(outputs, ei_items["L"], batch_y)
        else:
            loss = criterion(outputs, batch_y)
        return loss
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        d_EI = 0
        EI_data_x = torch.tensor([]).float().to(device=self.device)
        EI_data_y = torch.tensor([]).float().to(device=self.device)
        with torch.no_grad():
            for i, (idx, batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.EI:
                    EI_data_x = torch.cat((EI_data_x, batch_x), dim=0)
                    EI_data_y = torch.cat((EI_data_y, batch_y), dim=0)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.model_step(idx, batch_x, batch_y, criterion)      
                else:
                    loss = self.model_step(idx, batch_x, batch_y, criterion)

                total_loss.append(loss.item())

            if self.args.EI:
                _,ei_items = self.model(EI_data_x, EI_bool=self.args.EI)
                if "NIS" in self.args.model:
                    h_t1 = self.model.encoding(EI_data_y)
                else:
                    h_t1 = EI_data_y.reshape(-1, EI_data_y.size(1)*EI_data_y.size(2))
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
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.es_delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(cov_b=self.args.cov_bool, lam=self.args.loss_lam)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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
            vali_loss = self.vali(vali_data, vali_loader, criterion)[0]
            test_loss = self.vali(test_data, test_loader, criterion)[0]

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

    def cal_jac(self,dec_inp, batch_x):
        if self.args.features[0] == -1:
            fun = lambda x: self.model(x, dec_inp)[0]
        else:
            fun = lambda x: self.model(x, dec_inp)[0][:, :, self.args.features]
        jac = jacobian(fun, batch_x)
        jac = jac.detach().cpu().numpy()[0,:,:,0,:,:].astype(np.float16)
        return jac.reshape(jac.shape[0]*jac.shape[1], -1).astype(float)
    
    def cal_causal_net(self, dec_inp, batch_x):
        step_number = batch_x.size(1)
        space_number = batch_x.size(2)
        if self.args.output_attention or self.args.cov_bool:
            fun = lambda x: self.model(x, dec_inp)[0]#.reshape(-1,step_number*space_number)
        else:
            fun = lambda x: self.model(x, dec_inp)#.reshape(-1,step_number*space_number)
        ig = IntegratedGradients(fun)
        temp_attribution = np.zeros((step_number, space_number,step_number, space_number))
        for tar_1 in range(step_number):
            for tar_2 in range(space_number):
                attributions,_ = ig.attribute(batch_x,target=(tar_1,tar_2), method='gausslegendre', return_convergence_delta=True) 
                attributions = (attributions.abs().cpu().detach().numpy()).mean(0)
                temp_attribution[:,:,tar_1,tar_2] = attributions
        return temp_attribution

    def MSED(self, pred, true):
        pred = pred.reshape(pred.shape[0],-1)
        true = true.reshape(pred.shape[0],-1)
        mse_per_dimension = np.mean((true - pred)**2, axis=0)
        return np.diag(mse_per_dimension)
    
    def cal_cov(self,batch_x,batch_y):
        if self.args.cov_bool:
            if "Transformer" in self.args.model:
                mu, attn, L = self.model.forecast(batch_x)
            else:
                mu, L = self.model.forecast(batch_x)
            L = torch.matmul(L, L.transpose(1, 2)) 
            L = L.cpu().detach().data.numpy()[0]
        else:
            if "Transformer" in self.args.model:
                mu, attn = self.model.forecast(batch_x)
            else:
                mu = self.model.forecast(batch_x)
            mu = mu.cpu().detach().data.numpy()
            L = self.MSED(mu, batch_y)
        return L

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
        attention_path = './results/attentions/' + setting + '/'
        if not os.path.exists(attention_path):
            os.makedirs(attention_path)
        jacobian_path = './results/jacobian/' + setting + '/'
        L_path = './results/cov_L/' + setting + '/'
        if self.args.jacobian and (not os.path.exists(jacobian_path)):
            os.makedirs(jacobian_path)
        if self.args.jacobian and (not os.path.exists(L_path)):
            os.makedirs(L_path)
        ca_path = './results/causal_net/' + setting + '/'
        if self.args.causal_net and (not os.path.exists(ca_path)):
            os.makedirs(ca_path)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.eval()
        idx, batch_x, batch_y = next(iter(test_loader))
        size = batch_y.size(1)*batch_y.size(2)
        jacs = np.zeros([size,size])
        Ls = np.zeros([size,size])
        if self.args.data == "QBO":
            jacs = np.eye(size)
            Ls = np.eye(size)
        nums = 0
        batch_list = []
        for i, (idx, batch_x, batch_y) in enumerate(test_loader):
            self.model.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x.requires_grad_()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, items = self.model(batch_x, dec_inp)
                    if self.args.output_attention:
                        attn = items['attn']
                    elif self.args.cov_bool:
                        L = items['L']

            else:
                outputs, items = self.model(batch_x, dec_inp)
                if self.args.output_attention:
                    attn = items['attn']
                elif self.args.cov_bool:
                    L = items['L']

            outputs = outputs[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
    
            if self.args.features[0] != -1:
                outputs = outputs[:, :, self.args.features]
                batch_y = batch_y[:, :, self.args.features]

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            if (i > self.args.jac_init) and (i <= self.args.jac_end):
                batch_list.append(batch_x)
                if self.args.cov_mean:
                    L = self.cal_cov(batch_x,batch_y)
                    Ls = Ls + L
                    nums = nums + 1
                if self.args.jac_mean and (i-self.args.jac_init) % self.args.jac_mean_interval == 0:
                    jac = self.cal_jac(dec_inp, batch_x)
                    if self.args.data == "QBO":
                        jacs = jacs @ jac
                    else:
                        jacs = jacs + jac
                    nums = nums + 1
                if (i-self.args.jac_init) % self.args.jac_interval == 0:
                    if self.args.data == "QBO":
                        store_time = i
                    else:
                        store_time = i - self.args.jac_interval + batch_x.size(1)
                    t = time.time()
                    print(f'elapse: {t-t0:.2}s')
                    t0 = t
                    if self.args.jacobian:
                        if self.args.jac_mean:
                            if self.args.data == "QBO":
                                jacs = fractional_matrix_power(jacs, 1/nums)
                            else:
                                jacs = jacs / nums
                            np.save(jacobian_path + f'jac_{store_time:04}.npy', jacs)
                            if self.args.data == "QBO":
                                jacs = np.eye(size)
                            else:
                                jacs = np.zeros([size,size])
                        else:
                            jac = self.cal_jac(dec_inp, batch_x)
                            np.save(jacobian_path + f'jac_{store_time:04}.npy', jac)
                        print(f'saving jacobian: jac_{store_time:04}.npy(size: {jac.dtype.itemsize * jac.size // 1024}KB); ')
                    if self.args.jacobian:
                        if self.args.cov_mean:
                            # import pdb; pdb.set_trace()
                            Ls = Ls / nums
                            np.save(L_path + f'L_{store_time:04}.npy', Ls)
                            Ls = np.zeros([size,size])
                            nums = 0
                        else:
                            L = self.cal_cov(batch_x, batch_y)
                            np.save(L_path + f'L_{store_time:04}.npy', L)
                    if self.args.causal_net:
                        batch_x_cat = torch.cat(batch_list, dim=0)
                        ca_net = self.cal_causal_net(dec_inp, batch_x_cat)
                        np.save(ca_path + f'ca_{store_time:04}.npy', ca_net)
                        batch_list = []

                    if self.args.output_attention and attn is not None:
                        attn = attn.astype(np.float16)
                        np.save(attention_path + f'attn_{store_time:04}.npy', attn)
                        print(f'saving attention: attn_{store_time:04}.npy(size: {attn.dtype.itemsize * attn.size // 1024}KB); ')
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    
                    # selecting variable index to output images
                    si = 0
                    gt = np.concatenate((input[0, :, si], true[0, :, si]), axis=0)
                    pd = np.concatenate((input[0, :, si], pred[0, :, si]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f'{store_time:04}.pdf'))
                    if self.args.output_attention and attn is not None:
                        print(f'saving fig: {store_time:04}.pdf')

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
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

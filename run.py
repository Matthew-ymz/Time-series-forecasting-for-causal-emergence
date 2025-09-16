import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_coarse_graining import Exp_Coarse_Graining
from exp.exp_maxei import Exp_MaxEI
from exp.exp_imputation import Exp_Imputation
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import pickle

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--prints', type=int, default=100, help='time steps for print during training')
    parser.add_argument('--seed', type=int, default=2025, help='seed for multi-experiments')
    parser.add_argument('--save_model', action='store_true', help='whether to save models param', default=False)

    # data loader
    parser.add_argument('--data', type=str, required=True, default='SIR', help='dataset type')
    parser.add_argument('--downsample', type=int, default=1, help='dataset downsampling interval')
    parser.add_argument('--use_cache', type=bool, default=True, help='dataset cache used status')
    parser.add_argument('--root_path', type=str, default='./dataset/SIR/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    # parser.add_argument('--features', type=str, default='M',
    #                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--features', type=int, nargs='+', default=[-1], help='list for predicted dims, -1 for all dims')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--fold_loc', type=str, required=False, default='normal', help='location of vali and test')
    parser.add_argument('--data_partition', type=float, nargs='+', default=[0.7, 0.1, 0.2], help='partition for train, vali and test')
    
    # data sir
    parser.add_argument('--size_list', type=int, nargs='+', default=9000, help='dataset size for sir. Its sum is the total number of the init points.')
    parser.add_argument('--beta', type=float, default=1, help='dynamic param of sir')
    parser.add_argument('--gamma', type=float, default=0.5, help='dynamic param of sir')
    parser.add_argument('--steps', type=int, default=7, help='dynamic steps of sir')
    parser.add_argument('--dt', type=float, default=0.01, help='dynamic dt for differential equations')
    parser.add_argument('--sigma', type=float,default=0.03, help='noise strength')
    parser.add_argument('--rho', type=float, default=-0.5, help='noise correlation param')

    # data kuramoto
    parser.add_argument('--sz_kuramoto', type=int, default=32, help='dataset size for kuramoto. Its sum is the total number of the init points.')
    parser.add_argument('--groups_kuramoto', type=int, default=2, help='groups dividing of kuramoto')
    parser.add_argument('--batch_size_kuramoto', type=int, default=1, help='samples of data')
    parser.add_argument('--time_steps_kuramoto', type=int, default=1000, help='dynamic steps of sir')
    parser.add_argument('--dt_kuramoto', type=float, default=0.01, help='dynamic dt for differential equations')
    parser.add_argument('--sample_interval_kuramoto', type=int,default=1, help='sample interval of kuramoto')
    parser.add_argument('--coupling_strength', type=float, default=2.0, help='coupling strength of kuramoto')
    parser.add_argument('--noise_level_kuramoto', type=float, default=10, help='noise of kuramoto')

    #max EI
    parser.add_argument('--first_stage', type=int, default=2, help='len(epoch) of first stage for maxmize EI')
    parser.add_argument('--lambdas', type=float, default=1, help='balance param for two losses in maxmizing EI')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=1, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
    
    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size for MLP')
    parser.add_argument('--c_in', type=int, default=7, help='input size for MLP')
    parser.add_argument('--latent_size', type=int, default=2, help='latent space size for NIS')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--MLP_layers', type=int, default=1, help='number of hidden layers of MLP')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--freq_loss', action='store_true', help='whether to use loss in frequence domain')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--jacobian', action='store_true', help='whether to output jacobian matrix')
    parser.add_argument('--jac_init', type=int, default=1000, help='start time of jacobian output')
    parser.add_argument('--jac_end', type=int, default=2000, help='end time of jacobian output')
    parser.add_argument('--jac_interval', type=int, default=30, help='interval time of jacobian output')
    parser.add_argument('--cov_mean_num', type=int, default=1, help='how many samples to mean cov matrix')
    parser.add_argument('--EI', action='store_true', help='whether to output EI')
    parser.add_argument('--causal_net', action='store_true', help='whether to output causal network by IG')
    parser.add_argument('--ig_output', action='store_true', help='whether to output coarse graining by IG')
    parser.add_argument('--one_serie', action='store_false', help='whether it has only one start points', default=True)
    parser.add_argument('--ig_baseline',type=str, default='zero', help='IG baseline: zero, mean...')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--es_delta', type=float, default=0, help='early stopping score count scope')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if (torch.cuda.is_available() or torch.backends.mps.is_available()) \
        and args.use_gpu else False
    if torch.cuda.is_available():
        args.gpu_type = 'cuda'
    elif torch.backends.mps.is_available():
        args.gpu_type = 'mps'

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'coarse_graining':
        Exp = Exp_Coarse_Graining
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'maxei':
        Exp = Exp_MaxEI
    else:
        Exp = Exp_Long_Term_Forecast

    def set_setting(args, ii):
        if args.task_name == "coarse_graining":
            setting = '{}_{}_{}_{}_{}_to_{}_dm{}_layer{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.c_in,
                args.c_out,
                args.d_model,
                args.MLP_layers, 
                ii
            )
        else:
            if args.data == "SIR":
                setting = '{}_{}_{}_{}_samp{}_sigma{}_rho{}_dm{}_layer{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                sum(args.size_list),
                args.sigma,
                args.rho,
                args.d_model,
                args.MLP_layers,
                ii)
            elif args.data == "Couzin":
                if args.model == "NN":
                    setting = '{}_{}_{}_{}_sl{}_pl{}_dm{}_layer{}_floc{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.MLP_layers,
                    args.fold_loc,
                    ii)
            else:
                setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_floc{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features[0],
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.fold_loc,
                    args.distil,
                    args.des, ii)
        return setting
    
    if args.is_training:
        for ii in range(args.itr):
            seed = args.seed + ii
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = set_setting(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if args.save_model:
                #args_dict = vars(args)
                file_path = './checkpoints/' + setting + "/args.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(args, f)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            seed = args.seed + ii
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = set_setting(args, ii)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
        # ii = 0
        # setting = set_setting(args, ii)

        # exp = Exp(args)  # set experiments
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        # torch.cuda.empty_cache()

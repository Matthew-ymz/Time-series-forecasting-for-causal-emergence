from data_provider.data_loader import PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, Dataset_Ca2p, SIRModel
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'Ca2p': Dataset_Ca2p,
    'QBO': Dataset_Ca2p,
    'custom': Dataset_Ca2p,
    'SIR': SIRModel,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'testall':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    
    if args.task_name == 'long_term_forecast' or 'nn_forecast':
        if args.data == 'm4':
            drop_last = False
        if args.data == 'Ca2p':
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.pred_len],
                features=args.features,
                target=args.target,
                downsample=args.downsample,
                timeenc=timeenc,
                freq=freq,
                fold_loc=args.fold_loc,
                seasonal_patterns=args.seasonal_patterns
            )
        if args.data == "SIR":
            data_set = Data(
                path=args.root_path,
                size_list=args.size_list,
                beta=args.beta,
                gamma=args.gamma,
                steps=args.steps,
                dt=args.dt,
                flag=flag,
                interval=args.downsample,
                sigma=args.sigma,
                rho=args.rho,
                use_cache=args.use_cache
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                fold_loc=args.fold_loc,
                scale=False,
                seasonal_patterns=args.seasonal_patterns
            )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

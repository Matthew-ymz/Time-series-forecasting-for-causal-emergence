from data_provider.data_loader import Dataset_Ca2p, SIRModel, Dataset_couzin, Dataset_Lorzen, KuramotoModel, Micro_to_Macro
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'Ca2p': Dataset_Ca2p,
    'QBO': Dataset_Ca2p,
    'custom': Dataset_Ca2p,
    'SIR': SIRModel,
    'Couzin': Dataset_couzin,
    'Spring': Dataset_couzin,
    'Kuramoto':KuramotoModel,
    'coarse_graining':Micro_to_Macro,
    'Lorzen':Dataset_Lorzen,
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

    
    if args.task_name == 'long_term_forecast':
        if args.data == 'm4':
            drop_last = False
        elif args.data == 'Ca2p':
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
        elif args.data == 'Couzin' or args.data == 'Spring':
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                data_partition=args.data_partition,
                flag=flag,
                scale=args.data_scale,
                size=[args.seq_len, args.pred_len],
                downsample=args.downsample,
                fold_loc=args.fold_loc,
            )
        
        elif args.data == "SIR":
            data_set = Data(
                path=args.root_path,
                data_path=args.data_path,
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
        elif args.data == "Kuramoto":
            data_set = Data(
                path=args.root_path, 
                sz = args.sz_kuramoto,
                groups = args.groups_kuramoto, 
                batch_size = args.batch_size_kuramoto, 
                time_steps = args.time_steps_kuramoto, 
                dt = args.dt_kuramoto, 
                sample_interval = args.sample_interval_kuramoto, 
                coupling_strength = args.coupling_strength, 
                noise_level = args.noise_level_kuramoto, 
                flag = flag, 
                use_cache = args.use_cache
            )
        elif args.data == 'Lorzen':
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                data_partition=args.data_partition,
                flag=flag,
                scale=args.data_scale,
                size=[args.seq_len, args.pred_len],
                downsample=args.downsample,
                fold_loc=args.fold_loc,
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.pred_len],
                features=args.features,
                data_partition=args.data_partition,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                fold_loc=args.fold_loc,
                scale=False,
                seasonal_patterns=args.seasonal_patterns
            )
    
    elif args.task_name == 'coarse_graining':
        
        Data = data_dict[args.task_name]
        data_set = Data(
                path=args.root_path,
                data=args.data, 
                micro_dims=args.c_in, 
                macro_dims=args.c_out,
                flag=flag
            )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader

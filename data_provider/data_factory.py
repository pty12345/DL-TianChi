from data_provider.data_loader import Dataset_Stock
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_Stock

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        freq=freq
    )
    
    print(flag, data_set.shape())
    if len(data_set) < batch_size: 
        batch_size = len(data_set)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader

import os
import torch
import random
import argparse

import numpy as np
import pandas as pd

from exp.exp_main import Exp_Main



if __name__ == "__main__":
    
    from exp.args import args
    
    # fix random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    Exp = Exp_Main
        
    exp = Exp(args)  # set experiments
    setting = ""
    
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    
    # 打印加载的数据集
    # print((data_frames['product1']['transaction_date']).tolist())
    
    # sets = []
    # for file_name, df in train_frames.items():
    #     print(f"Loaded {file_name} with shape {df.shape}")
        
    #     sets.extend(df['transaction_date'].tolist())
        
    # print(set(sets))
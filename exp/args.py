import torch
import random
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Final for Deep Learning')

# random seed
parser.add_argument('--random_seed', type=int, default=2023, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model', type=str, required=False, default='ConvTimeNet')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# dataset
parser.add_argument('--enc_in', type=int, default=16, help='encoder input size') 
parser.add_argument('--c_out', type=int, default=3, help='output size')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')

# forecasting task
parser.add_argument('--seq_len', type=int, default=30, help='input sequence length')
parser.add_argument('--label_len', type=int, default=5, help='start token length')
parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# Revin
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

# General config
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

# ConvTimeNet
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

parser.add_argument('--dw_ks', type=str, default='11,15,21,29,39,51', help="kernel size of the deep-wise. default:9")
parser.add_argument('--re_param', type=int, default=1, help='Reparam the DeepWise Conv when train')
parser.add_argument('--enable_res_param', type=int, default=1, help='Learnable residual')
parser.add_argument('--re_param_kernel', type=int, default=3)

parser.add_argument('--patch_ks', type=int, default=2, help="kernel size of the patch window. default:32")
parser.add_argument('--patch_sd', type=float, default=0.5, \
					help="stride of the patch window. default: 0.5. if < 1, then sd = patch_sd * patch_ks")

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

args = parser.parse_args()
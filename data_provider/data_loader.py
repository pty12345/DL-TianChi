import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

# 请你帮我用代码写一个类，要求实现的功能与StandardScaler一致
class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
        self.transform_columns = ['uv_fundown', 'uv_fundopt']
        # self.transform_columns = None

    def fit(self, data_frames):
        if self.transform_columns is None:
            return
        
        if self.mean_ is not None or self.scale_ is not None:
            return
        
        # transform columns: 'uv_fundown', 'uv_fundopt'
        
        X = []
        for product_id, df in data_frames.items():
            # just transform 'uv_fundown', 'uv_fundopt'
            X.append(df[self.transform_columns].values)
            
        X = np.concatenate(X, axis=0)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, data_frames):
        if self.transform_columns is None:
            data_frames
        
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("You must fit the scaler before transforming data.")
        
        new_data_frames = {}
        for product_id, df in data_frames.items():
            new_data_frames[product_id] = df.copy()
            
            for idx, column in enumerate(self.transform_columns):
                new_data_frames[product_id][column] = (df[column] - self.mean_[idx]) / self.scale_[idx]
        
        return new_data_frames

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def load_csv_files(dataset_dir):
    data_frames = {}
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                # remove 'is_trade', 'trade_day_rank'
                for drop_column in ['is_trade', 'trade_day_rank']:
                    if drop_column in df.columns:
                        df = df.drop(columns=[drop_column]) 
                        
                # just keep 'transaction_date', 'apply_amt', 'redeem_amt', 'net_in_amt'
                # if 'test' not in dataset_dir:
                #     df = df[['transaction_date', 'apply_amt', 'redeem_amt', 'net_in_amt', 'uv_fundown', 'uv_fundopt']]
                
                data_frames[file.split('.')[0]] = df
                
                
    # 按照key中蕴含的数字大小进行索引字典序，对data_frame进行重新排列
    data_frames = dict(sorted(data_frames.items(), key=lambda x: int(x[0].split('duct')[1])))
    
    return data_frames

class Dataset_Stock(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        self.output_columns = ['apply_amt', 'redeem_amt', 'net_in_amt']
        
        self.flag = flag
        self.__read_data__(flag)

    def __read_data__(self, flag):
        if flag in ['train', 'val']:
            data_frames = load_csv_files(dataset_dir=f'./{self.root_path}//{flag}')
            
            if self.scale:
                self.scaler = CustomStandardScaler() 
                data_frames = self.scaler.fit_transform(data_frames)   
            
            # pd.set_option('display.max_rows', None) #显示全部行
            # pd.set_option('display.max_columns', None) #显示全部列

            # # # print the first line of the first product
            # print(data_frames['product1'].head(1))
            # # print(data_frames['product1'].values[0])
            # exit(0)
                    
            datas_x, datas_y = [], []
            datas_x_mark, datas_y_mark = [], []
            for product_id, df in data_frames.items():
                data_stamp = time_features(pd.to_datetime(df['transaction_date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
                
                # remove 'transaction_date' from df
                df = df.drop(columns=['transaction_date'])
                
                self.input_columns = df.columns
                
                input_data = df.values
                output_data = df[self.output_columns].values
                
                for s_begin in range(len(input_data) - self.seq_len - self.pred_len + 1):
                    s_end = s_begin + self.seq_len
                    r_begin = s_end - self.label_len
                    r_end = r_begin + self.label_len + self.pred_len

                    seq_x = input_data[s_begin:s_end]
                    seq_y = output_data[r_begin:r_end]
                    seq_x_mark = data_stamp[s_begin:s_end]
                    seq_y_mark = data_stamp[r_begin:r_end]
                    
                    datas_x.append(seq_x)
                    datas_y.append(seq_y)
                    datas_x_mark.append(seq_x_mark)
                    datas_y_mark.append(seq_y_mark)
                    
            self.datas_x, self.datas_y = np.array(datas_x), np.array(datas_y)
            self.datas_x_mark, self.datas_y_mark = np.array(datas_x_mark), np.array(datas_y_mark)
            
        else:
            input_train_frames = load_csv_files(dataset_dir=f'./{self.root_path}//train')
            input_val_frames = load_csv_files(dataset_dir=f'./{self.root_path}//val')
            
            # concat train and val
            input_frames = {}
            for key, value in input_train_frames.items():
                input_frames[key] = pd.concat([value, input_val_frames[key]])
            
            output_frames = load_csv_files(dataset_dir=f'./{self.root_path}//test')

            datas_x, datas_y = [], []
            datas_x_mark, datas_y_mark = [], []
            product_ids = []
            for product_id, output_df in output_frames.items():
                input_df = input_frames[product_id]
                
                input_stamp = time_features(pd.to_datetime(input_df['transaction_date'].values), freq=self.freq)
                output_stamp = time_features(pd.to_datetime(output_df['transaction_date'].values), freq=self.freq)
                
                # remove 'transaction_date' from df
                input_df = input_df.drop(columns=['transaction_date'])
                output_df = output_df.drop(columns=['transaction_date'])
                
                self.input_columns = input_df.columns
                
                input_data = input_df.values
                output_data = output_df.values
                
                assert self.pred_len == output_data.shape[0]
                
                datas_x.append(input_data[-self.seq_len:])
                datas_y.append(output_data)
                datas_x_mark.append(input_stamp[:, -self.seq_len:])
                datas_y_mark.append(output_stamp)
                
                product_ids.append(product_id)

            self.product_ids = product_ids
            self.datas_x, self.datas_y = np.array(datas_x), np.array(datas_y)
            
            self.datas_x_mark, self.datas_y_mark = np.array(datas_x_mark), np.array(datas_y_mark)

    def __getitem__(self, index):
        seq_x, seq_y = self.datas_x[index], self.datas_y[index]
        seq_x_mark, seq_y_mark = self.datas_x_mark[index], self.datas_y_mark[index]

        if self.flag == 'test':
            product_id = self.product_ids[index]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, product_id
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.datas_x)
    
    def shape(self):
        return f"Input: {self.datas_x.shape}, Output: {self.datas_y.shape}"

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)


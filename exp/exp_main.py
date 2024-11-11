### Mo

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ConvTimeNet
from models import MLP
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class WMAPELoss(nn.Module):
    def __init__(self):
        super(WMAPELoss, self).__init__()

    def forward(self, pred, true):
        return torch.sum(torch.abs(true - pred)) / torch.sum(torch.abs(true))

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'ConvTimeNet': ConvTimeNet,
            'MLP': MLP
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 损失函数为WMAPE
        criterion = WMAPELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                    
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        # set revin columns
        self.model.revin_layer.set_column(train_data.input_columns, train_data.output_columns)

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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # Input: [B, L, C]
                
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
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

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # set revin columns
        self.model.revin_layer.set_column(test_data.input_columns, test_data.output_columns)
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        product_ids = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, product_id) in enumerate(test_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                product_ids.append(product_id)
            
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        inputx = np.concatenate(inputx)
        product_ids = np.concatenate(product_ids)
        
        
        from data_provider.data_loader import load_csv_files
        data_frames = load_csv_files(dataset_dir=f'./{self.args.root_path}//test')
        
        for idx, product_ids in enumerate(product_ids):
            # 将'apply_amt_pred', 'redeem_amt_pred', 'net_in_amt_pred'结果保存至对应的dataframe
            data_frames[product_ids]['apply_amt_pred'] = preds[idx, :, 0]
            data_frames[product_ids]['redeem_amt_pred'] = preds[idx, :, 1]
            data_frames[product_ids]['net_in_amt_pred'] = preds[idx, :, 2]
            
            # 给dataframe添加索引行product_id，默认初始值为product1
            data_frames[product_ids].insert(0, 'product_pid', product_ids)
            
        # 将data_frames里的多个dataframe合并为一个dataframe，，新增索引行product_id, 并保存至csv文件
        final_frame = pd.concat(data_frames)
        
        # 保存结果
        final_frame.to_csv(f'./{self.args.checkpoints}/result.csv', index=False)
        

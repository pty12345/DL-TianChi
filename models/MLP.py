import torch

import torch.nn as nn
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs, hidden_dim=16):
        super(Model, self).__init__()
        enc_in = configs.enc_in
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        
        c_out = configs.c_out
        
        n_layers = 2
        hidden_dim = [hidden_dim] * n_layers

        
        layers = nn.ModuleList([nn.Flatten()])
        
        for i in range(n_layers):
            input_dim = enc_in * seq_len if i == 0 else hidden_dim[i - 1]
            output_dim = hidden_dim[i] if i != n_layers - 1 else pred_len * c_out
                
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.Dropout(configs.dropout))
            
            
        layers.append(nn.Unflatten(1, (pred_len, c_out)))
            
        self.main_net = nn.Sequential(*layers)
        
        affine, subtract_last = configs.affine, configs.subtract_last
        self.revin_layer = RevIN(enc_in, affine=affine, subtract_last=subtract_last)

    def forward(self, x):
        x = self.revin_layer(x, 'norm')
        
        x = self.main_net(x)
        x = self.revin_layer(x, 'denorm')
        return x

# Example usage:
# input_shape = (B, L, C)
# output_shape = (B, L1, C1)
# model = MLP(input_shape=(32, 10, 64), output_shape=(32, 20, 128))
# x = torch.randn(32, 10, 64)
# output = model(x)
# print(output.shape)  # Should be torch.Size([32, 20, 128])
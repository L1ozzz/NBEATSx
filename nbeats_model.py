import math
import numpy as np
import torch as t
from typing import Tuple
from tcn import TemporalConvNet
from LSTM import *

import torch.nn.functional as F

def filter_input_vars(insample_y, insample_x_t, outsample_x_t, t_cols, include_var_dict):
    # This function is specific for the EPF task
    if t.cuda.is_available():
        device = insample_x_t.get_device()
    else:
        device = 'cpu'
    #print("device is:", device)
    outsample_y = t.zeros((insample_y.shape[0], 1, outsample_x_t.shape[2])).to(device)

    insample_y_aux = t.unsqueeze(insample_y,dim=1)

    insample_x_t_aux = t.cat([insample_y_aux, insample_x_t], dim=1)
    outsample_x_t_aux = t.cat([outsample_y, outsample_x_t], dim=1)
    x_t = t.cat([insample_x_t_aux, outsample_x_t_aux], dim=-1)
    batch_size, n_channels, input_size = x_t.shape

    assert input_size == 8, f'input_size {input_size} not 31'

    x_t = x_t.reshape(batch_size, n_channels, input_size, 1)

    input_vars = []
    for var in include_var_dict.keys():
        if len(include_var_dict[var])>0:
            t_col_idx    = t_cols.index(var)
            t_col_filter = include_var_dict[var]
            if var != 'week_day':
                input_vars  += [x_t[:, t_col_idx, t_col_filter, :]]
            else:
                assert t_col_filter == [-1], f'Day of week must be of outsample not {t_col_filter}'
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)

    x_t_filter = t.cat(input_vars, dim=1)
    x_t_filter = x_t_filter.view(batch_size,-1)

    if len(include_var_dict['week_day'])>0:
        x_t_filter = t.cat([x_t_filter, day_var], dim=1)

    return x_t_filter


class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class NBeatsBlock(nn.Module):
    def __init__(self, x_t_n_inputs: int, x_s_n_inputs: int, x_s_n_hidden: int, theta_n_dim: int, basis: nn.Module,
                 n_layers: int, theta_n_hidden: list, include_var_dict, t_cols, batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):

            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1]))
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i+1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)]
        layers = hidden_layers + output_layer

        # x_s_n_inputs is computed with data, x_s_n_hidden is provided by user, if 0 no statics are used
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=x_s_n_inputs, out_features=x_s_n_hidden)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        if self.include_var_dict is not None:
            insample_y = filter_input_vars(insample_y=insample_y, insample_x_t=insample_x_t, outsample_x_t=outsample_x_t,
                                           t_cols=self.t_cols, include_var_dict=self.include_var_dict)

        # Static exogenous
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

#7月19日，继续修改这里

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor, return_decomposition=False):

        residuals = insample_y.flip(dims=(-1,))

        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))
        '''
        target_size = 9
        pad_size = target_size - residuals.size(1)
        if pad_size > 0:
            residuals = F.pad(residuals, (0, pad_size), "constant", 0)
            #forecast = F.pad(forecast, (0, pad_size), "constant", 0)

        print(residuals.shape)
        '''
        forecast = insample_y[:, -1:]
        block_forecasts = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            #print("residual",residuals.shape)
            #print("backcast",backcast.shape)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            # 确保所有预测都扩展到同一尺寸
            if block_forecast.size(1) != 30:
                block_forecast = F.pad(block_forecast, (0, 30 - block_forecast.size(1)), "constant", 0)
            block_forecasts.append(block_forecast)

        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        # 选择第一个预测点
        #forecast = forecast[:, 0].unsqueeze(1)  # Reshape to [batch_size, 1]
        forecast = torch.mean(forecast, dim=1, keepdim=True)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast

        #print("yuce1",forecast)

    def decomposed_prediction(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                              outsample_x_t: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]
        forecast_components = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            forecast_components.append(block_forecast)
            #print("yuce2", forecast)
        return forecast, forecast_components

        #

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast

class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        #print('TrendBasis theta shape:', theta.shape)
        #print('TrendBasis cut_point:', cut_point)
        #print('TrendBasis backcast_basis shape:', self.backcast_basis.shape)
        #print('TrendBasis forecast_basis shape:', self.forecast_basis.shape)

        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)

        #print('TrendBasis backcast shape:', backcast.shape)
        #print('TrendBasis forecast shape:', forecast.shape)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(np.zeros(1, dtype=np.float32),
                              np.arange(harmonics, harmonics / 2 * forecast_size,
                                        dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

        backcast_cos_template = t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32)
        backcast_sin_template = t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32)
        backcast_template = t.cat([backcast_cos_template, backcast_sin_template], dim=0)

        forecast_cos_template = t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32)
        forecast_sin_template = t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32)
        forecast_template = t.cat([forecast_cos_template, forecast_sin_template], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ExogenousBasisWavenet(nn.Module):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(t.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        padding = (kernel_size - 1) * (2**0)
        input_layer = [nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=kernel_size, padding=padding, dilation=2**0),
                       Chomp1d(padding),
                       nn.ReLU(),
                       nn.Dropout(dropout_prob)]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(nn.Conv1d(in_channels=out_features, out_channels=out_features,
                                         padding=padding, kernel_size=3, dilation=dilation))
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.Sequential(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = x_t * self.weight  # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast






class ExogenousBasisTCN(nn.Module):
    def __init__(self, out_features, in_features, num_levels=8, kernel_size=4, dropout_prob=0.2):
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = TemporalConvNet(num_inputs=in_features, num_channels=n_channels, kernel_size=kernel_size, dropout=dropout_prob)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = self.tcn(x_t)[:]
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]
        #print('这里是！！！',backcast_basis)
        #print('这里是111',forecast_basis)

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        #print('这里是222', backcast)
        #print('这里是333', forecast)
        return backcast, forecast




'''
class ExogenousBasisLSTM(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=1, dropout_prob=0.2):
        super(ExogenousBasisLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=31, hidden_size=64, num_layers=1, batch_first=True, dropout=dropout_prob)
        self.hidden_size = hidden_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)
        lstm_out, _ = self.lstm(x_t)

        backcast_basis = lstm_out[:, :, :insample_x_t.shape[2]]
        forecast_basis = lstm_out[:, :, insample_x_t.shape[2]:]

        print("Initial Theta shape:", theta.shape)
        print("Backcast basis shape:", backcast_basis.shape)
        print("Forecast basis shape:", forecast_basis.shape)

        #cut_point = forecast_basis.shape[1]
        cut_point = min(forecast_basis.shape[1], theta.size(1) - 1)
        #cut_point = min(theta.size(1), forecast_basis.shape[1], backcast_basis.shape[1])

        print("Cut point:", cut_point)
        print("Theta shape after cut point (remaining part):", theta[:, cut_point:].shape)
        print("Theta shape after cut point (initial part):", theta[:, :cut_point].shape)
        print("Forecast basis shape for einsum:", forecast_basis.shape)
        print("Backcast basis shape for einsum:", backcast_basis.shape)
        print((lstm_out[:, :, :cut_point]).shape)
        print(theta[:, cut_point:].shape)

        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast
        
        '''

class ExogenousBasisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, theta_size):
        super(ExogenousBasisLSTM, self).__init__()
        self.theta_size = theta_size
        self.lstm_block = LSTMBlock(input_size=input_size, hidden_size=hidden_size,
                                    output_size=theta_size, num_layers=num_layers,
                                    dropout_prob=dropout_prob)
    '''
    @property
    def weight(self):
        # 返回所有内部需要正则化的权重
        return [self.lstm_block.lstm.weight_ih_l0,
                self.lstm_block.lstm.weight_hh_l0,
                self.lstm_block.fc.weight]
    '''

    @property
    def weight(self):
        weights = []
        lstm = self.lstm_block.lstm
        num_layers = lstm.num_layers
        for layer_idx in range(num_layers):
            weights.append(getattr(lstm, f'weight_ih_l{layer_idx}'))
            weights.append(getattr(lstm, f'weight_hh_l{layer_idx}'))
        weights.append(self.lstm_block.fc.weight)
        return weights

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor):
        x_t = torch.cat([insample_x_t, outsample_x_t], dim=2)  # x_t: [batch_size, input_size, seq_length]
        x_t = x_t.permute(0, 2, 1)  # 调整维度为 [batch_size, seq_length, input_size]
        lstm_out = self.lstm_block(x_t)  # lstm_out: [batch_size, seq_length, theta_size]

        # 将 LSTM 输出分为 backcast 和 forecast
        insample_length = insample_x_t.shape[2]
        backcast_basis = lstm_out[:, :insample_length, :]  # [batch_size, insample_length, theta_size]
        forecast_basis = lstm_out[:, insample_length:, :]  # [batch_size, outsample_length, theta_size]

        # 对 theta 进行分割
        theta_backcast = theta[:, :theta.shape[1] // 2]  # 假设 theta 的前一半用于 backcast
        theta_forecast = theta[:, theta.shape[1] // 2:]  # 后一半用于 forecast

        # 将 theta 扩展到时间维度
        theta_backcast = theta_backcast.unsqueeze(1)  # [batch_size, 1, theta_size]
        theta_forecast = theta_forecast.unsqueeze(1)  # [batch_size, 1, theta_size]

        # 计算 backcast 和 forecast
        # 使用逐元素乘法，然后在时间维度上求和
        backcast = (backcast_basis * theta_backcast).sum(dim=2)  # [batch_size, insample_length]
        #backcast = backcast.mean(dim=2)  # [batch_size]

        forecast = (forecast_basis * theta_forecast).sum(dim=2)  # [batch_size, outsample_length]
        #forecast = forecast.mean(dim=2)  # [batch_size]

        #print("LSTM backcast:",backcast.shape)
        #print("LSTM forecast:",forecast.shape)




        # 计算 backcast 和 forecast
        #backcast = torch.einsum('bts,bh->bt', backcast_basis, theta_backcast)
        #forecast = torch.einsum('bts,bh->bt', forecast_basis, theta_forecast)

        return backcast, forecast


'''
class ExogenousBasisLSTM(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=1, dropout_prob=0.2, theta_size=36):
        super(ExogenousBasisLSTM, self).__init__()
        output_size = theta_size // 2  # 设定输出大小为 theta 的一半
        self.lstm_block = LSTMBlock(input_size=31, hidden_size=64, output_size=output_size, num_layers=2,
                                    dropout_prob=0.2)

    @property
    def weight(self):
        # 返回所有内部需要正则化的权重
        return [self.lstm_block.lstm.weight_ih_l0,
                self.lstm_block.lstm.weight_hh_l0,
                self.lstm_block.fc.weight]

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        x_t = torch.cat([insample_x_t, outsample_x_t], dim=2)
        lstm_out = self.lstm_block(x_t)  # LSTM模块处理合并后的时间序列

        cut_point = theta.size(1) // 2
        # 计算 backcast 和 forecast
        backcast = torch.einsum('bp,bp->bp', theta[:, cut_point:], lstm_out[:, :cut_point])
        forecast = torch.einsum('bp,bp->bp', theta[:, :cut_point], lstm_out[:, cut_point:])

        # 填充 backcast 和 forecast 使其维度达到 [1024, 30]
        target_size = 30
        pad_size = target_size - backcast.size(1)
        if pad_size > 0:
            backcast = F.pad(backcast, (0, pad_size), "constant", 0)
            forecast = F.pad(forecast, (0, pad_size), "constant", 0)

        #print(backcast.shape)  # 应该显示 [1024, 30]
        #print(forecast.shape)  # 应该显示 [1024, 30]
        return backcast, forecast
    '''


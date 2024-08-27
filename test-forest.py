import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from epf import EPF
from ts_dataset import TimeSeriesDataset
from ts_loader import TimeSeriesLoader
from nbeats import Nbeats
from losses import MAELoss
from skopt import forest_minimize

# Load and preprocess data
directory = 'D:/nbeatsx3/NBEATSX/NBEATSx/data'
filename = 'updated_processed_weather_data.csv'
Y_df, X_df, S_df = EPF.load(directory, filename)

# Handling numerical and string data dynamically
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
string_cols = X_df.select_dtypes(include=[object, 'category']).columns

# Scaling numerical data
scaler = StandardScaler()
X_df[numerical_cols] = scaler.fit_transform(X_df[numerical_cols])

# Encoding categorical string data
label_encoders = {}
for col in string_cols:
    le = LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col].astype(str))
    label_encoders[col] = le

# Set up mask for training data
train_mask = np.ones(len(Y_df))
train_mask[-3587:] = 0  # Last week of data (168 hours)




# Creating the dataset object
ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)


# 定义搜索空间
space = [
    Real(1e-6, 1e-3, "log-uniform", name='learning_rate'),
    Real(0.0, 0.5, name='dropout_prob_theta'),
    Real(0.0, 0.5, name='dropout_prob_exogenous'),
    Real(0.0, 0.01, name='weight_decay'),
    Real(0.0, 0.9, name='lr_decay'),
    #Real(1.0, 10.0, name='n_lr_decay_steps')
]

from sklearn.metrics import mean_absolute_error

def mae_loss(target, forecast, weights):
    return mean_absolute_error(target, forecast, sample_weight=weights)

# 包含要包含的滞后变量的字典。
include_var_dict = { 'y': list(range(-7, -1)),  # 过去30天的土壤湿度
                    'GustDir': list(range(-7, 0)),
                    'GustSpd': list(range(-7, 0)),
                    'WindRun': list(range(-7, 0)),
                    'Rain': list(range(-7, 0)),
                    'Tmean': list(range(-7, 0)),
                    'Tmax': list(range(-7, 0)),
                    'Tmin': list(range(-7, 0)),
                    'Tgmin': list(range(-7, 0)),
                    'VapPress': list(range(-7, 0)),
                    'ET10': list(range(-7, 0)),
                    'Rad': list(range(-7, 0)),
                    'week_day': [-1]}  # Last day of the week
# 定义目标函数
@use_named_args(space)
def objective(**params):
    nbeats_model = Nbeats(
        input_size_multiplier=30, output_size=1, shared_weights=False,
        initialization='he_normal', activation='relu',
        stack_types=['seasonality']+['identity']+ ['exogenous']+ ['exogenous_lstm']+['trend'],
        n_blocks=[4, 4, 4, 4, 4], n_layers=[4, 4, 4, 8, 4],
        n_hidden=[[1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024]],
        #stack_types=['seasonality']+['identity']+ ['exogenous_lstm'],
        #n_blocks=[4, 4, 4], n_layers=[4, 4, 16],
        #n_hidden=[[1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024],  [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]],
        n_harmonics=0, n_polynomials=0, x_s_n_hidden=0,
        exogenous_n_channels=len(string_cols), include_var_dict=include_var_dict,
        t_cols=ts_dataset.t_cols, batch_normalization=True, dropout_prob_theta=params['dropout_prob_theta'],
        dropout_prob_exogenous=params['dropout_prob_exogenous'],
        learning_rate=params['learning_rate'], lr_decay=params['lr_decay'], n_lr_decay_steps=5,
        early_stopping=10, weight_decay=params['weight_decay'],
        l1_theta=0, n_iterations=5000, loss='MAE', loss_hypar=0.5,
        val_loss='MAE', seasonality=7, random_seed=1
    )
    train_loader = TimeSeriesLoader(
        model='nbeats', ts_dataset=ts_dataset, window_sampling_limit=8969,
        offset=0, input_size=30, output_size=1, idx_to_sample_freq=1,
        batch_size=1024, is_train_loader=True, shuffle=False
    )
    val_loader = TimeSeriesLoader(
        model='nbeats', ts_dataset=ts_dataset, window_sampling_limit=8969,
        offset=0, input_size=30, output_size=1, idx_to_sample_freq=1,
        batch_size=1024, is_train_loader=False, shuffle=False
    )
    nbeats_model.fit(train_loader, val_loader,eval_steps=50)
    val_loss = nbeats_model.evaluate_performance(val_loader,validation_loss_fn=mae_loss)
    return val_loss

# 执行贝叶斯优化
res = forest_minimize(objective,space,n_calls=50, random_state=0)
print("Best parameters: {}".format(res.x),"Best MAE: {}".format(res.fun))
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from epf import EPF
from ts_dataset import TimeSeriesDataset
from ts_loader import TimeSeriesLoader
from nbeats import Nbeats
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 加载本地数据
directory='./data'
filename='updated_processed_weather_data.csv'
Y_df, X_df, S_df = EPF.load(directory,filename)
print(S_df)
#print(Y_df.head)
# 初始历史数据，包含最后30天的数据用于第一次预测




'''
# 选择数值型列进行标准化
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_df[numerical_cols] = scaler.fit_transform(X_df[numerical_cols])

'''
# train_mask: 1 to keep, 0 to mask
train_mask = np.ones(len(Y_df))
train_mask[-2000:] = 0 # Last week of data (168 hours)

ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)


# Handling numerical and string data dynamically
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
string_cols = X_df.select_dtypes(include=[object, 'category']).columns
label_encoders = {}
for col in string_cols:
    le = LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col].astype(str))
    label_encoders[col] = le

'''
his_data_x = Y_df.iloc[-(8 + 7):-7]
new_ = his_data_x.iloc[-1].copy()  # 复制最后一行以保持结构
print(new_[0])
print(new_['ds'])
'''
#new_entry['ds'] = future_dates[i]  # 更新日期为未来日期
#new_entry['y'] = actuals[i]  # 使用实际值而非预测值
    # 转换 unique_id 为整数，递增后转换回字符串
#new_['unique_id'] = str(int(his_data_x['unique_id'].iloc[-1]) + 1)

# 打印 t_cols 确认
#print(ts_dataset.t_cols)
# 加载对象。采样数据集对象的窗口。
# 有关每个参数的更多信息，请参阅 Loader 对象上的注释。
train_loader = TimeSeriesLoader(model='nbeats',
                                ts_dataset=ts_dataset,
                                window_sampling_limit=365*20,  # 4 years of data
                                offset=0,
                                input_size=30,  # Last 7 days
                                output_size=1,  # Predict 1 day
                                idx_to_sample_freq=1,  # Sampling frequency of 1 day
                                batch_size=1024,
                                is_train_loader=True,
                                shuffle=True)

# 验证加载器（注意：在此示例中，我们还对预测期进行了验证）

val_loader = TimeSeriesLoader(model='nbeats',
                              ts_dataset=ts_dataset,
                              window_sampling_limit=365*20,  # 4 years of data
                              offset=0,
                              input_size=30,  # Last 7 days
                              output_size=1,  # Predict 1 day
                              idx_to_sample_freq=1,  # Sampling frequency of 1 day
                              batch_size=1024,
                              is_train_loader=False,
                              shuffle=False)

print(dir(val_loader))
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

model = Nbeats(input_size_multiplier=30,  # Last 7 days
               output_size=1,  # Predict 1 day
               shared_weights=False,
               initialization='he_normal',#'he_uniform'
               activation='relu',
               stack_types=['seasonality']+['identity']+ ['exogenous']+ ['exogenous_lstm']+['trend'] ,
               n_blocks=[4, 4, 4, 4, 4],
               n_layers=[4, 4, 4, 8, 4],
               n_hidden=[[1024, 1024,1024, 1024], [1024, 1024,1024, 1024],[1024, 1024,1024, 1024], [1024,1024,1024,1024,1024,1024,1024,1024], [1024,1024,1024,1024]],
               #stack_types=['seasonality']+['identity']+ ['exogenous_lstm'],
               #n_blocks=[4, 4, 4], n_layers=[4, 4, 16],
               #n_hidden=[[1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024],  [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]],
               n_harmonics=0,  # not used with exogenous_tcn
               n_polynomials=0,  # not used with exogenous_tcn
               x_s_n_hidden=0,
               exogenous_n_channels=len(string_cols),
               include_var_dict=include_var_dict,
               t_cols=ts_dataset.t_cols,
               batch_normalization=True,
               dropout_prob_theta=0.5,
               dropout_prob_exogenous=0,
               learning_rate=0.000343823,
               lr_decay=0.3667177,
               n_lr_decay_steps=3,
               early_stopping=10,
               weight_decay=0.01,
               l1_theta=0,
               n_iterations=5000,
               loss='MAE',
               loss_hypar=0.5,
               val_loss='MAE',
               seasonality=7,  # not used: only used with MASE loss
               random_seed=1)



model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=50)

# 保存模型的状态字典
model_dir = 'model'  # 指定保存模型的路径
model_id = 'nbeats_model-4-8-1024'  # 给模型命名

# 保存模型
model.save(model_dir, model_id)
print(val_loader)


y_true, y_hat, *_ = model.predict(ts_loader=val_loader, return_decomposition=False)
print("Y_df['y'].values[-336:]:", Y_df['y'].values[-336:].shape)
print("y_hat:", y_hat.shape)

# 设定历史数据和预测数据的长度
history_length = 30  # 用于绘制的历史天数
forecast_length = 7  # 预测天数

# 取历史数据的最后 history_length 天
history_y = Y_df['y'].values[-(history_length + forecast_length):-forecast_length]

# 取预测时间段内的实际数据
actual_y = Y_df['y'].values[-forecast_length:]

# 将预测结果展开并取最后 forecast_length 天的值
y_hat_flat = y_hat.flatten()
y_hat_flat = y_hat_flat[-forecast_length:]
# 确保预测结果和 x 轴长度一致
x_values = range(history_length, history_length + forecast_length)



# 绘制历史数据
plt.plot(range(history_length), history_y, label='Historical Soil Moisture')

# 绘制预测值
plt.plot(x_values, y_hat_flat, linestyle='dashed', label='Forecast',marker='o')

# 绘制预测时间段内的实际数据
plt.plot(x_values, actual_y, label='Actual Soil Moisture',marker='x')

# 添加分割线
plt.axvline(history_length, color='black')

# 添加图例、网格和标签
plt.legend()
plt.grid()
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# 计算 MSE, MAE 和 MAPE
mse = mean_squared_error(actual_y, y_hat_flat)
mae = mean_absolute_error(actual_y, y_hat_flat)
mape = np.mean(np.abs((actual_y - y_hat_flat) / actual_y)) * 100

print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')

import torch as t
from losses import *

# 转换 NumPy 数组到 Tensor
y_true = t.from_numpy(y_true)  # 假设原始的 y_true 是 NumPy 数组
y_hat = t.from_numpy(y_hat)    # 同样假设 y_hat 是 NumPy 数组
# 创建掩码
mask = t.ones_like(y_true)
#mask = t.ones_like(y_true)  # 创建一个与 y_true 形状相同的掩码

# 计算不同的损失
mape_loss = MAPELoss(y_true, y_hat, mask)
mse_loss = MSELoss(y_true, y_hat, mask)
smape_loss = SMAPELoss(y_true, y_hat, mask)
#mase_loss = MASELoss(y_true, y_hat, y_insample, seasonality=12, mask=mask)  # 假设季节性为12
mae_loss = MAELoss(y_true, y_hat, mask)
pinball_loss = PinballLoss(y_true, y_hat, mask, tau=0.5)  # 假设tau为0.5，可以根据需要调整

# 打印所有损失
print(f"MAPE Loss: {mape_loss.item():.4f}")
print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"SMAPE Loss: {smape_loss.item():.4f}")
#print(f"MASE Loss: {mase_loss.item():.4f}")
print(f"MAE Loss: {mae_loss.item():.4f}")
print(f"Pinball Loss: {pinball_loss.item():.4f}")

# 创建 DataFrame 存储实际值和预测值
results_df = pd.DataFrame({
    'Actual': actual_y,
    'Predicted': y_hat_flat
})

# 保存 DataFrame 到 CSV 文件
results_df.to_csv('prediction_results2.csv', index=False)

print("Results saved to prediction_results.csv")

from datetime import time

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

# 加载本地数据
directory='./data'
filename='updated_processed_weather_data.csv'
Y_df, X_df, S_df = EPF.load(directory,filename)
#print(S_df)
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
train_mask[-30:] = 0 # Last week of data (168 hours)

ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)
'''
his_data_x = Y_df.iloc[-(8 + 7):-7]
new_ = his_data_x.iloc[-1].copy()  # 复制最后一行以保持结构
print(new_[0])
print(new_['ds'])
'''


import numpy as np
import pandas as pd


history_length = 30  # 用于绘制的历史天数
forecast_length = 7  # 预测天数
#history_data_x = X_df.iloc[-(history_length + forecast_length):-forecast_length]
# 初始历史数据，包含最后30天的数据用于第一次预测
history_data_y = Y_df.iloc[-(history_length + forecast_length):-forecast_length]
# 数据集对象。将 DataFrame 预处理为 pytorch 张量和窗口。

test_mask = np.ones(len(history_data_y))
test_mask[-30:] = 0

def create_loader(current_history, model, input_size,history_data_x, batch_size=1):
    # 将current_history转换为DataFrame，因为TimeSeriesDataset可能需要DataFrame输入
    print("current_history: ", current_history)
    #test_mask = np.ones(len(current_history))
    #test_mask[-30:] = 0
    data_df = pd.DataFrame(current_history)
    datax_df = pd.DataFrame(history_data_x)
    print("data_df: ",data_df)
    #test_dataset = TimeSeriesDataset(Y_df=data_df, X_df=datax_df, S_df=S_df, ts_train_mask=test_mask)
    test_loader = 1
    return test_loader



# 设定历史数据和预测数据的长度

import numpy as np

# 初始化存储预测和实际结果的列表
predictions = []
actuals = Y_df['y'].values[-forecast_length:]  # 实际值
print(actuals)
last_known_date = history_data_y['ds'].iloc[-1]
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
print("日期们： ",future_dates)

# 进行递归预测
current_history = history_data_y.copy()
for i in range(forecast_length):
    print("this is iteration:",i)

    # 动态生成history_data_x
    start_idx = -(history_length + forecast_length) + i
    end_idx = -forecast_length + i if -forecast_length + i != 0 else None
    history_data_x = X_df.iloc[start_idx:end_idx]
    print(history_data_x)

    # 创建当前历史数据的加载器
    #current_loader = create_loader(current_history, 'nbeats', input_size=30,history_data_x=history_data_x)
    # 使用模型进行预测
    #y_hat_today = model.predict(ts_loader=current_loader, return_decomposition=False)[1] # 获取预测结果
    #print(y_hat_today)
    #y_hat_today = y_hat_today.flatten()[-1]  # 假设每次预测输出一个结果
    #print("迭代次数：",i, "预测结果：",y_hat_today)

    # 存储预测结果
    #predictions.append(y_hat_today)

    # 从current_history中删除最早的一行，并添加新的实际结果
    start_idx_1 = -(history_length + forecast_length) + i
    end_idx_1 = -forecast_length + i if -forecast_length + i != 0 else None
    current_history = Y_df.iloc[start_idx_1:end_idx_1]
    print(current_history)
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

print(type(history_data_y))
'''
import matplotlib.pyplot as plt

# 绘制历史数据
plt.plot(range(history_length), history_data_y, label='Historical Soil Moisture')

# 绘制预测值和实际值
x_values = range(history_length, history_length + forecast_length)
plt.plot(x_values, predictions, linestyle='dashed', label='Forecast', marker='o')
plt.plot(x_values, actuals, label='Actual Soil Moisture', marker='x')

# 添加图例和标签
plt.legend()
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.show()
# 计算误差指标
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - np.array(predictions)) / actuals)) * 100

print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')'''
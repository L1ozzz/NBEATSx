import torch

import torch

# 第一步：加载模型文件
checkpoint = torch.load('model/model_nbeats_model-4-8-1024.pth')

# 第二步：检查加载的内容
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    #print(state_dict)
else:
    print("The loaded checkpoint does not contain a 'model_state_dict' key.")


i = range(-7,0)
print(i)

'''
from nbeats import Nbeats


modeldir = 'model'
modelid = 'nbeats_model-4-8-1024'

model = Nbeats.load(modeldir,modelid)
print(model)
'''
'''
directory = './data'
filename = 'Processed_Data.csv'
Y_df, X_df, S_df = EPF.load(directory,filename)
train_mask = np.ones(len(Y_df))
train_mask[-30:] = 0 # Last week of data (168 hours)

#model = 'model/model_nbeats_model-4-8-1024.model'

ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)

test_dataset = torch.tensor(Y_df['y'].values.astype(np.float32))
# 假设 Nbeats 类和 TimeSeriesDataset 已经被正确导入
# 创建模型实例
model = Nbeats(input_size_multiplier=30,
               output_size=1,
               shared_weights=False,
               initialization='he_normal',
               activation='relu',
               stack_types=['seasonality', 'identity', 'exogenous_lstm', 'trend'],
               n_blocks=[3, 3, 4, 3],
               n_layers=[2, 2, 8, 2],
               n_hidden=[[1024, 1024], [1024, 1024], [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], [1024, 1024]],
               exogenous_n_channels=9,
               include_var_dict=include_var_dict,
               t_cols=ts_dataset.t_cols,  # 确保这个是从相应的 TimeSeriesDataset 实例中获取的
               batch_normalization=True,
               dropout_prob_theta=0.2,
               dropout_prob_exogenous=0.2,
               learning_rate=0.0001,
               lr_decay=0.6,
               n_lr_decay_steps=5,
               early_stopping=10,
               weight_decay=0,
               l1_theta=0,
               n_iterations=5000,
               loss='MAE',
               loss_hypar=0.5,
               val_loss='MAE',
               random_seed=1)

# 加载模型
model_dir = 'model'  # 模型保存的路径
model_id = 'nbeats_model-4-8-1024'  # 模型的ID

model.load(model_dir, model_id)

# 现在模型已经被加载，可以查看模型输入要求
# 这通常包括输入数据的维度和类型
print("模型输入大小:", model.input_size_multiplier)






# 假设数据已经按照需要格式处理，现在转换为 PyTorch Tensors
# 需要根据实际情况可能需要对特征和标签进一步处理，如转换数据类型和形状

'''
import pandas as pd

# 指定文件路径
filepath = 'data/titi.xlsx'

# 从Excel文件读取数据到DataFrame
df = pd.read_excel(filepath)

# 修改日期格式
df['Day(Local_Date)'] = pd.to_datetime(df['Day(Local_Date)'].str.split(':').str[0], format='%Y%m%d').dt.strftime('%Y/%m/%d')

# 提取前37天的数据
df_37_days = df.head(37)

# 将这些数据存储到一个新的CSV文件中
output_file_path = 'data/Processed_Data.csv'
df_37_days.to_csv(output_file_path, index=False)
'''

def predict_future_seven_days(model, dataset, initial_history_len=30, forecast_days=7):
    # 确保模型处于评估模式
    model.eval()

    # 用于存储预测和真实值
    predictions = []
    actuals = []

    # 初始历史数据（前30天）
    history_data = dataset[:initial_history_len].unsqueeze(0)  # 添加批处理维度

    # 逐天预测接下来的七天
    with torch.no_grad():
        for i in range(forecast_days):
            # 预测下一天
            prediction = model(history_data)
            predicted_value = prediction[:, -1, :].cpu().numpy()  # 获取预测值

            # 获取真实值（用于下一个输入）
            actual_value = dataset[initial_history_len + i].unsqueeze(0).unsqueeze(0)

            # 更新历史数据：去掉最早的一天，加入最新一天的真实数据
            history_data = torch.cat((history_data[:, 1:, :], actual_value), dim=1)

            # 保存预测结果和实际结果
            predictions.append(predicted_value.flatten())
            actuals.append(dataset[initial_history_len + i].numpy().flatten())

    # 计算MSE和MAE
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    return predictions, actuals, mse, mae

# 假定数据文件位于指定目录下


# 假设你有一个已经加载并处理好的测试集
# test_dataset 应该是一个 torch.Tensor 类型，包含了37天的数据
predictions, actuals, mse, mae = predict_future_seven_days(model, test_dataset)
print("MSE:", mse)
print("MAE:", mae)
'''
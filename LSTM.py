import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_length, hidden_size]
        # 通过线性层，将每个时间步的输出映射到 output_size
        lstm_out = self.fc(lstm_out)  # lstm_out: [batch_size, seq_length, output_size]
        return lstm_out



'''
class LSTMBlock(nn.Module):
    """A single LSTM block for processing sequences."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=8, dropout_prob=0.2):
        super(LSTMBlock, self).__init__()
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        # 线性层将 LSTM 输出维度转换为所需的输出维度
        self.fc = nn.Linear(hidden_size, output_size)
        # 激活函数
        #self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the LSTM block."""
        # 通过 LSTM 层
        lstm_out,_= self.lstm(x)
        # 取最后一个时间步的输出用于预测
        out = lstm_out[:, -1, :]
        # 通过线性层
        out = self.fc(out)
        # 应用激活函数
        #out = self.relu(out)
        return out
'''
'''
import torch
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(0)
torch.manual_seed(0)

# 生成示例数据
# 每个输入序列的长度为10，每个时间步的特征数为5
input_size = 5
sequence_length = 10
num_samples = 100

# 随机生成数据
data = np.random.randn(num_samples, sequence_length, input_size)
targets = np.random.randn(num_samples, 1)  # 每个序列对应一个输出

# 转换为torch tensors
data = torch.tensor(data, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
# 参数配置
hidden_size = 64
output_size = 1
num_layers = 2
dropout = 0.1

# 实例化模型
model = LSTMBlock(input_size, hidden_size, output_size, num_layers, dropout)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
model.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 关闭梯度计算
    predicted = model(data)
    print("Predicted:", predicted[:5])  # 打印前5个预测结果
    print("Actual:", targets[:5])
'''
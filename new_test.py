import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from epf import EPF
from ts_dataset import TimeSeriesDataset
from ts_loader import TimeSeriesLoader
from nbeats import Nbeats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV

# Load and preprocess data
directory = './data'
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
train_mask[-500:] = 0  # Last week of data (168 hours)

# Creating the dataset object
ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)


train_loader = TimeSeriesLoader(model='nbeats',
                                ts_dataset=ts_dataset,
                                window_sampling_limit=1000,  # 4 years of data
                                offset=0,
                                input_size=30,  # Last 7 days
                                output_size=1,  # Predict 1 day
                                idx_to_sample_freq=1,  # Sampling frequency of 1 day
                                batch_size=512,
                                is_train_loader=True,
                                shuffle=True)

# 验证加载器（注意：在此示例中，我们还对预测期进行了验证）

val_loader = TimeSeriesLoader(model='nbeats',
                              ts_dataset=ts_dataset,
                              window_sampling_limit=1000,  # 4 years of data
                              offset=0,
                              input_size=30,  # Last 7 days
                              output_size=1,  # Predict 1 day
                              idx_to_sample_freq=1,  # Sampling frequency of 1 day
                              batch_size=512,
                              is_train_loader=False,
                              shuffle=False)

# Model definition
def build_nbeats(input_size_multiplier=30, output_size=1):
    return Nbeats(
        input_size_multiplier=input_size_multiplier,
        output_size=output_size,
        shared_weights=False,
        initialization='he_normal',
        activation='relu',
        stack_types=['seasonality']+['identity']+ ['exogenous']+ ['exogenous_lstm']+['trend'],
        n_blocks=[4, 4, 4, 4, 4],
        n_layers=[4, 4, 4, 8, 4],
        n_hidden=[[1024, 1024,1024, 1024], [1024, 1024,1024, 1024],[1024, 1024,1024, 1024], [1024,1024,1024,1024,1024,1024,1024,1024], [1024,1024,1024,1024]],
        n_harmonics=0,
        n_polynomials=0,
        x_s_n_hidden=0,
        exogenous_n_channels=len(string_cols),
        include_var_dict=include_var_dict,
        t_cols=ts_dataset.t_cols,
        batch_normalization=True,
        dropout_prob_theta=0.2,
        dropout_prob_exogenous=0.2,
        learning_rate=0.00001,
        lr_decay=0.6,
        n_lr_decay_steps=5,
        early_stopping=10,
        weight_decay=0,
        l1_theta=0,
        n_iterations=100000,
        loss='MAE',
        loss_hypar=0.5,
        val_loss='MAE',
        seasonality=7,
        random_seed=1
    )

# Implementing Bayesian Optimization for hyperparameter tuning
search_spaces = {
    'learning_rate': (1e-6, 1e-3, 'log-uniform'),
    'dropout_prob_theta': (0.0, 0.5),
    'dropout_prob_exogenous': (0.0, 0.5),
    'weight_decay': (0.0, 1e-2),
}

opt = BayesSearchCV(
    estimator=build_nbeats(),
    search_spaces=search_spaces,
    n_iter=32,
    cv=3,
    random_state=0
)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X_df, Y_df, test_size=0.2, random_state=42)

# Fit model with hyperparameter optimization
opt.fit(X_train, y_train)

# Best model
best_model = opt.best_estimator_

# Cross-Validation and fitting
train_loader = TimeSeriesLoader(
    model='nbeats',
    ts_dataset=ts_dataset,
    window_sampling_limit=1000,
    offset=0,
    input_size=30,
    output_size=1,
    idx_to_sample_freq=1,
    batch_size=512,
    is_train_loader=True,
    shuffle=True
)

val_loader = TimeSeriesLoader(
    model='nbeats',
    ts_dataset=ts_dataset,
    window_sampling_limit=1000,
    offset=0,
    input_size=30,
    output_size=1,
    idx_to_sample_freq=1,
    batch_size=512,
    is_train_loader=False,
    shuffle=False
)

best_model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=1000)

# Reverse scaling of predictions and actuals
def reverse_scaling(predictions, actuals, scaler):
    return scaler.inverse_transform(predictions), scaler.inverse_transform(actuals)


def create_loader(current_history, model, input_size,history_data_x, batch_size=1):
    # 将current_history转换为DataFrame，因为TimeSeriesDataset可能需要DataFrame输入
    print("current_history: ", current_history)
    #test_mask = np.ones(len(current_history))
    #test_mask[-30:] = 0
    data_df = pd.DataFrame(current_history)
    datax_df = pd.DataFrame(history_data_x)
    print("data_df: ",data_df)
    test_dataset = TimeSeriesDataset(Y_df=data_df, X_df=datax_df, S_df=S_df, ts_train_mask=test_mask)
    test_loader = TimeSeriesLoader(
        model='nbeats',
        ts_dataset=test_dataset,
        window_sampling_limit=30,
        offset=0,
        input_size=30,
        output_size=1,
        idx_to_sample_freq=1,
        batch_size=1,
        is_train_loader=False,
        shuffle=False
        )
    return test_loader


# Forecasting and performance evaluation
history_length = 30
forecast_length = 7
current_history = Y_df.iloc[-(history_length + forecast_length):-forecast_length]
test_mask = np.ones(len(current_history))
test_mask[-30:] = 0
predictions = []
actuals = Y_df['y'].values[-forecast_length:]

for i in range(forecast_length):
    start_idx = -(history_length + forecast_length) + i
    end_idx = -forecast_length + i if -forecast_length + i != 0 else None
    history_data_x = X_df.iloc[start_idx:end_idx]

    current_loader = create_loader(current_history, best_model, 30, history_data_x)
    y_true, y_hat_today, *_ = best_model.predict(ts_loader=current_loader, return_decomposition=False)
    y_hat_today = y_hat_today.flatten()[-1]

    predictions.append(y_hat_today)

# Reverse scaling predictions and actuals
predictions, actuals = reverse_scaling(predictions, actuals, scaler)

# Calculate error metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - np.array(predictions)) / actuals)) * 100

print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')

# Plotting predictions
plt.plot(range(history_length), current_history['y'], label='Historical Soil Moisture')
plt.plot(range(history_length, history_length + forecast_length), predictions, linestyle='dashed', label='Forecast', marker='o')
plt.plot(range(history_length, history_length + forecast_length), actuals, label='Actual Soil Moisture', marker='x')

plt.legend()
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.show()

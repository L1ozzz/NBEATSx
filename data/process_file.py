import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # 必须导入以启用 IterativeImputer
from sklearn.impute import IterativeImputer

# 步骤1：读取数据
df = pd.read_excel('weather.xlsx')

# 步骤2：处理日期格式
def convert_date(date_str):
    date_part = date_str[:8]
    formatted_date = '{}-{}-{}'.format(date_part[:4], date_part[4:6], date_part[6:8])
    return formatted_date

df['Day(Local_Date)'] = df['Day(Local_Date)'].apply(convert_date)

# 步骤3：处理缺失值表示
df.replace('-', np.nan, inplace=True)

# 步骤4：将数据类型转换为数值型
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
print("非数值型列：", non_numeric_cols)

for col in df.columns:
    if col not in non_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 步骤5：选择数值型列
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 方法3：IterativeImputer
imputer_iter = IterativeImputer(random_state=0, max_iter=30, tol=1e-3)
df_iterative_imputed = df.copy()
df_iterative_imputed[numeric_cols] = imputer_iter.fit_transform(df_iterative_imputed[numeric_cols]).round(2)
df_iterative_imputed[non_numeric_cols] = df[non_numeric_cols]

'''
# 方法1：SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
df_simple_imputed = df.copy()
df_simple_imputed[numeric_cols] = imputer_mean.fit_transform(df_simple_imputed[numeric_cols]).round(2)
df_simple_imputed[non_numeric_cols] = df[non_numeric_cols]

# 方法2：KNNImputer
imputer_knn = KNNImputer(n_neighbors=5)
df_knn_imputed = df.copy()
df_knn_imputed[numeric_cols] = imputer_knn.fit_transform(df_knn_imputed[numeric_cols])
df_knn_imputed[non_numeric_cols] = df[non_numeric_cols]


'''

# 步骤7：根据日期生成季节（南半球）
def get_season(date_str):
    # 解析日期字符串为 datetime 对象
    try:
        date = pd.to_datetime(date_str)
        month = date.month
        if month in [12, 1, 2]:
            return 'Summer'   # 夏季
        elif month in [3, 4, 5]:
            return 'Autumn'  # 秋季
        elif month in [6, 7, 8]:
            return 'Winter'  # 冬季
        elif month in [9, 10, 11]:
            return 'Spring'  # 春季
    except:
        return np.nan  # 如果日期解析失败，返回 NaN

df_iterative_imputed['Season'] = df_iterative_imputed['Day(Local_Date)'].apply(get_season)

# 步骤6：保存插补后的数据（以 IterativeImputer 为例）
df_iterative_imputed.to_csv('imputed_data_ITer.csv', index=False)

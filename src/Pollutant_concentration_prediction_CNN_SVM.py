import math
import os.path
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Reshape, UpSampling1D, Conv1DTranspose, \
    Cropping1D

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 配置路径 -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
dataset_path = os.path.join(project_dir, 'dataset', 'GF5A_v3.xlsx')
output_imgs = os.path.join(project_dir, 'imgs', 'GF5A_v3')
os.makedirs(output_imgs, exist_ok=True)

# ----------------- 数据加载 -----------------
df = pd.read_excel(dataset_path, header=0)
target_columns = ['总氮浓度', '总磷浓度', '氨氮浓度']


# ----------------- 定义填充方法 -----------------
def apply_filling(data, method):
    """ 应用不同的缺失值填充方法 """
    if method == 'zero':
        return data.fillna(0)
    elif method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'knn':
        return pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data), columns=data.columns)
    return data


# ----------------- 模拟退火优化器 -----------------
class SimulatedAnnealing:
    def __init__(self, X, y, target_idx):
        self.X = X
        self.y = y[:, target_idx]
        self.best_score = -float('inf')
        self.best_params = None

    # SVM参数空间（对数尺度）
    param_ranges = {
        'C': (0.1, 100),
        'gamma': (1e-4, 10)
    }

    # 生成初始解（对数均匀采样）
    def generate_initial(self):
        params = {
            'C': 10 ** random.uniform(-1, 2),  # 0.1 ~ 100
            'gamma': 10 ** random.uniform(-4, 1)  # 0.0001 ~ 10
        }
        return params

    # 生成邻域解（高斯扰动）
    def generate_neighbor(self, current_params):
        new_params = current_params.copy()
        param_to_change = random.choice(list(self.param_ranges.keys()))

        # 在对数尺度上扰动
        current_val_log = np.log10(new_params[param_to_change])
        new_val_log = current_val_log + random.gauss(0, 0.3)  # 标准差0.3个对数单位
        new_val = 10 ** new_val_log

        # 边界处理
        low, high = self.param_ranges[param_to_change]
        new_val = max(min(new_val, high), low)
        new_params[param_to_change] = new_val
        return new_params

    # 目标函数：交叉验证R2均值
    def evaluate(self, params):
        model_best = SVR(
            kernel='rbf',
            C=params['C'],
            gamma=params['gamma'],
            epsilon=0.1
        )
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model_best, self.X, self.y, cv=kfold, scoring='r2')
        return np.mean(scores)

    # 退火过程
    def optimize(self, initial_temp=1000, cooling_rate=0.95, iterations=50):
        current_params = self.generate_initial()
        current_score = self.evaluate(current_params)
        temp = initial_temp

        for _ in range(iterations):
            neighbor_params = self.generate_neighbor(current_params)
            neighbor_score = self.evaluate(neighbor_params)

            # Metropolis准则
            if neighbor_score > current_score:
                accept_prob = 1.0
            else:
                accept_prob = math.exp((neighbor_score - current_score) / temp)

            if accept_prob > random.random():
                current_params = neighbor_params
                current_score = neighbor_score

            if current_score > self.best_score:
                self.best_score = current_score
                self.best_params = current_params

            temp *= cooling_rate

        return self.best_params, self.best_score


# ----------------- CNN特征提取模块 -----------------
def build_pretrained_feature_extractor():
    """ 构建自监督预训练的特征提取模型（修正维度版本） """
    # 编码器结构
    inputs = Input(shape=(294, 1))
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)  # 输出长度147
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 输出长度74
    x = Flatten()(x)
    encoded = Dense(128, name='bottleneck')(x)

    # 解码器结构
    x = Dense(74 * 128)(encoded)  # 匹配池化后维度
    x = Reshape((74, 128))(x)
    x = Conv1DTranspose(128, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)  # 长度148
    x = Conv1DTranspose(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)  # 长度296
    x = Cropping1D((0, 2))(x)  # 裁剪至294
    decoded = Conv1DTranspose(1, 3, activation='linear', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    return autoencoder, Model(inputs, encoded)


# ----------------- 模型训练与评估 -----------------
fill_methods = ['zero', 'mean', 'median', 'knn']
results = {target: {method: [] for method in fill_methods} for target in target_columns}
optimized_params = {target: {} for target in target_columns}

for method in fill_methods:
    # 数据预处理
    df_filled = apply_filling(df, method)
    X = df_filled.iloc[:, :294].values
    y = df_filled[target_columns].values

    # 数据标准化
    X_scaled = StandardScaler().fit_transform(X)

    # 重塑为CNN输入格式并提取特征
    X_reshaped = X_scaled.reshape(-1, 294, 1)
    autoencoder, feature_extractor = build_pretrained_feature_extractor()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_reshaped, X_reshaped, epochs=100, batch_size=32, verbose=0)
    cnn_features = feature_extractor.predict(X_reshaped)

    # 特征标准化
    feat_scaler = StandardScaler()
    X_enhanced = feat_scaler.fit_transform(cnn_features)

    for target_idx, target_name in enumerate(target_columns):
        # 执行模拟退火优化
        sa = SimulatedAnnealing(X_enhanced, y, target_idx)
        best_params, best_score = sa.optimize()
        optimized_params[target_name][method] = best_params
        print(f'{target_name} | {method} - 最优参数:{best_params}, R2:{best_score:.3f}')

        # 使用优化参数训练模型
        model_best = SVR(
            kernel='rbf',
            C=best_params['C'],
            gamma=best_params['gamma'],
            epsilon=0.1
        )

        # 五折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model_best,
            X_enhanced,
            y[:, target_idx],
            cv=kfold,
            scoring='r2',
            n_jobs=-1
        )

        results[target_name][method] = cv_scores

# ----------------- 可视化 -----------------
plt.figure(figsize=(15, 5))
colors = {'zero': '#1f77b4', 'mean': '#ff7f0e', 'median': '#2ca02c', 'knn': '#d62728'}
markers = {'zero': 'o', 'mean': 's', 'median': 'D', 'knn': '^'}

for idx, target in enumerate(target_columns):
    ax = plt.subplot(1, 3, idx + 1)

    for method in fill_methods:
        scores = results[target][method]
        x_pos = np.arange(1, 6) + 0.1 * (list(fill_methods).index(method) - 1.5)
        plt.plot(
            x_pos,
            scores,
            marker=markers[method],
            color=colors[method],
            linewidth=1,
            markersize=6,
            label=method
        )

    plt.title(f'{target}预测性能')
    plt.xlabel('Fold')
    plt.ylabel('R2 score')
    plt.xticks(range(1, 6))
    plt.ylim(-0.1, 1.0)
    plt.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_imgs, 'fill_method_comparison_SVM_SA_CNN.png'), dpi=300)
plt.show()

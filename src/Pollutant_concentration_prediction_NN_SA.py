import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

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
    if method == 'zero':
        return data.fillna(0)
    elif method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'knn':
        return pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(data), columns=data.columns)
    return data

# ----------------- 增强型神经网络模型 -----------------
class EnhancedRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        layers = []
        # 输入层
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        ]
        # 隐藏层
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout_rate)
            ]
        # 输出层
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.model(x)

# ----------------- 评估函数 -----------------
def evaluate(X, y, target_idx, hidden_size, learning_rate, num_layers, dropout_rate, weight_decay, n_epochs=200):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in kfold.split(X):
        # 数据标准化
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X[train_idx])
        X_val = scaler_x.transform(X[val_idx])

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y[train_idx, target_idx].reshape(-1, 1)).flatten()
        y_val = y[val_idx, target_idx]

        # 转换为Tensor
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)

        # 初始化模型
        model = EnhancedRegressionModel(
            input_size=X_train.shape[1],
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout_rate=dropout_rate
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.SmoothL1Loss()

        # 早停配置
        best_val_loss = float('inf')
        patience = 15
        trigger_times = 0

        # Mini-batch训练
        batch_size = 64

        for epoch in range(n_epochs):
            model.train()
            permutation = torch.randperm(X_train_t.size(0))

            for i in range(0, X_train_t.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = X_train_t[indices]
                batch_y = y_train_t[indices]
                # 扩展目标值维度
                batch_y = batch_y.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # 早停检查
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_preds_np = val_preds.cpu().numpy().reshape(-1, 1)
                val_preds = scaler_y.inverse_transform(np.asarray(val_preds_np))
                # 扩展目标值维度
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
                val_loss = criterion(torch.FloatTensor(val_preds), y_val_tensor)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    break

        # 最终评估
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).squeeze().cpu().numpy().reshape(-1, 1)
            val_preds = scaler_y.inverse_transform(np.asarray(val_preds))
            score = r2_score(y_val, val_preds)
            fold_scores.append(score)

    return np.mean(fold_scores)

# ----------------- 贝叶斯优化 -----------------
def optimize(X, y, target_idx):
    pbounds = {
        'hidden_size': (256, 512),
        'learning_rate': (1e-5, 1e-3),
        'num_layers': (2, 5),
        'dropout_rate': (0.0, 0.5),
        'weight_decay': (1e-6, 1e-4)
    }

    def black_box_function(hidden_size, learning_rate, num_layers, dropout_rate, weight_decay):
        return evaluate(X, y, target_idx, hidden_size, learning_rate, num_layers, dropout_rate, weight_decay)

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=20,
    )

    return optimizer.max['params'], optimizer.max['target']

# ----------------- 模型训练与评估 -----------------
fill_methods = ['zero', 'mean', 'median', 'knn']
results = {target: {method: [] for method in fill_methods} for target in target_columns}
optimized_params = {target: {} for target in target_columns}

for method in fill_methods:
    print(f"\n=== 当前填充方法: {method} ===")
    df_filled = apply_filling(df, method)
    X = df_filled.iloc[:, :294].values
    y = df_filled[target_columns].values

    for target_idx, target_name in enumerate(target_columns):
        print(f"\n** 目标变量: {target_name} **")
        best_params, best_score = optimize(X, y, target_idx)
        optimized_params[target_name][method] = best_params
        print(f"最优参数: {best_params}")
        print(f"验证集R²: {best_score:.4f}")

        # 交叉验证评估
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X[train_idx])
            X_val = scaler_x.transform(X[val_idx])

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y[train_idx, target_idx].reshape(-1, 1)).flatten()
            y_val = y[val_idx, target_idx]

            # 转换数据
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)

            model = EnhancedRegressionModel(
                input_size=X.shape[1],
                hidden_size=int(best_params['hidden_size']),
                num_layers=int(best_params['num_layers']),
                dropout_rate=best_params['dropout_rate']
            )

            optimizer = optim.AdamW(
                model.parameters(),
                lr=best_params['learning_rate'],
                weight_decay=best_params['weight_decay']
            )

            criterion = nn.SmoothL1Loss()

            # 训练配置
            best_val_loss = float('inf')
            patience = 15
            trigger_times = 0
            batch_size = 64

            for epoch in range(200):
                model.train()
                permutation = torch.randperm(X_train_t.size(0))

                # Mini-batch训练
                for i in range(0, X_train_t.size(0), batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x = X_train_t[indices]
                    batch_y = y_train_t[indices]
                    # 扩展目标值维度
                    batch_y = batch_y.unsqueeze(1)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # 早停检查
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_t)
                    val_preds_np = val_preds.cpu().numpy().reshape(-1, 1)
                    val_preds = scaler_y.inverse_transform(np.asarray(val_preds_np))
                    # 扩展目标值维度
                    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
                    val_loss = criterion(torch.FloatTensor(val_preds), y_val_tensor)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        break

            # 最终预测
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).squeeze().cpu().numpy().reshape(-1, 1)
                val_preds = scaler_y.inverse_transform(np.asarray(val_preds))
                score = r2_score(y_val, val_preds)
                fold_scores.append(score)
                print(f"Fold {fold + 1} R²: {score:.4f}")

        results[target_name][method] = fold_scores
        print(f"平均R²: {np.mean(fold_scores):.4f}")

# ----------------- 可视化 -----------------
plt.figure(figsize=(15, 5))
colors = {'zero': '#1f77b4', 'mean': '#ff7f0e', 'median': '#2ca02c', 'knn': '#d62728'}
markers = {'zero': 'o', 'mean': 's', 'median': 'D', 'knn': '^'}

for idx, target in enumerate(target_columns):
    ax = plt.subplot(1, 3, idx + 1)

    for method in fill_methods:
        scores = results[target][method]
        x_pos = np.arange(1, 6) + 0.1 * (list(fill_methods).index(method) - 1.5)
        plt.plot(x_pos, scores, marker=markers[method], color=colors[method],
                 linewidth=1, markersize=6, label=method)

    plt.title(f'{target}预测性能')
    plt.xlabel('Fold')
    plt.ylabel('R2 score')
    plt.xticks(range(1, 6))
    plt.ylim(-0.1, 1.0)
    plt.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_imgs, 'fill_method_comparison_NN_SA.png'), dpi=300)
plt.show()
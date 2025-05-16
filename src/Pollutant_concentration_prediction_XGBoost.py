import math
import os.path
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 配置路径 ----------------
# 获取当前脚本文件的绝对路径并提取所在目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 构建数据集所在路径
dataset_path = os.path.join(project_dir, 'dataset', 'GF5A_v3.xlsx')
# 构建输出路径
output_imgs = os.path.join(project_dir, 'imgs', 'GF5A_v3')
# 确保路径存在
os.makedirs(output_imgs, exist_ok=True)

# ---------------- 数据加载 ----------------
# 读取数据
df = pd.read_excel(dataset_path, header=0)
# 定义目标变量列名
target_columns = ['总氮浓度', '总磷浓度', '氨氮浓度']

# ---------------- 定义填充方法 ----------------


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


class SimulatedAnnealing:
    def __init__(self, X, y, target_idx):
        self.X = X
        self.y = y[:, target_idx]
        self.best_score = -float('inf')
        self.best_params = None

    # 定义参数空间
    param_ranges = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'n_estimators': (100, 300),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }

    # 生成初始解
    def generate_initial(self):
        params = {}
        for param, (low, high) in self.param_ranges.items():
            if param in ['max_depth', 'n_estimators']:
                params[param] = int(random.uniform(low, high))
            else:
                params[param] = round(random.uniform(low, high), 3)
        return params

    # 生成邻域解
    def generate_neighbor(self, current_params):
        new_params = current_params.copy()
        param_to_change = random.choice(list(self.param_ranges.keys()))
        low, high = self.param_ranges[param_to_change]

        if param_to_change in ['max_depth', 'n_estimators']:
            new_val = int(random.gauss(new_params[param_to_change], 1))
            new_val = max(min(new_val, high), low)
        else:
            new_val = new_params[param_to_change] + random.uniform(-0.1, 0.1)
            new_val = max(min(new_val, high), low)
            new_val = round(new_val, 3)

        new_params[param_to_change] = new_val
        return new_params

    # 目标函数：交叉验证R2均值
    def evaluate(self, params):
        model = xgboost.XGBRegressor(
            objective='reg:squarederror',
            **params,
            random_state=42
        )
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X, self.y, cv=kfold, scoring='r2')
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


# ---------------- 模型训练与评估 ----------------
fill_methods = ['zero', 'mean', 'median', 'knn']
results = {target: {method: [] for method in fill_methods} for target in target_columns}
optimized_params = {target: {} for target in target_columns}

for method in fill_methods:
    # 数据预处理
    df_filled = apply_filling(df, method)
    X = df_filled.iloc[:, :294].values  # 光谱特征
    y = df_filled[target_columns].values

    # 数据标准化
    X_scaled = StandardScaler().fit_transform(X)

    # 遍历每个目标变量
    for target_idx, target_name in enumerate(target_columns):
        # 执行模拟退火优化
        sa = SimulatedAnnealing(X_scaled, y, target_idx)
        best_params, best_score = sa.optimize(initial_temp=1000, cooling_rate=0.95)
        optimized_params[target_name][method] = best_params
        print(f'best_params:{best_params}; best_score:{best_score}')

        # 使用优化参数训练最终模型
        model = xgboost.XGBRegressor(
            objective='reg:squarederror',
            **best_params,
            random_state=42
        )

        # # 配置模型
        # model = xgboost.XGBRegressor(
        #     objective='reg:squarederror',  # 显式声明损失函数为平方误差（MSE）
        #     n_estimators=200,
        #     max_depth=5,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42
        # )

        # 五折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model,
            X_scaled,
            y[:, target_idx],
            cv=kfold,
            scoring='r2',
            n_jobs=-1
        )

        # 保存结果
        results[target_name][method] = cv_scores

# ---------------- 可视化 ----------------
plt.figure(figsize=(15, 5))
colors = {'zero': '#1f77b4', 'mean': '#ff7f0e', 'median': '#2ca02c', 'knn': '#d62728'}
markers = {'zero': 'o', 'mean': 's', 'median': 'D', 'knn': '^'}

for idx, target in enumerate(target_columns):
    ax = plt.subplot(1, 3, idx+1)

    # 绘制每个填充方法的结果
    for method in fill_methods:
        scores = results[target][method]
        x_pos = np.arange(1, 6) + 0.1 * (list(fill_methods).index(method)-1.5)
        plt.plot(
            x_pos,
            scores,
            marker=markers[method],
            color=colors[method],
            linewidth=1,
            markersize=6,
            label=method
        )

    # 图表格式
    plt.title(f'{target}预测性能')
    plt.xlabel('Fold')
    plt.ylabel('R2 score')
    plt.xticks(range(1, 6))
    plt.ylim(-0.1, 1.0)
    plt.grid(alpha=0.3)

    # 添加图例
    ax.legend(
        loc='upper right',  # 图例位置
        # fontsize=8,        # 字体大小
        framealpha=0.9,    # 背景透明度
        edgecolor='gray'   # 边框颜色
    )

    # 添加参考线
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_imgs, 'fill_method_comparison_SimulateAnnealing.png'), dpi=300)
plt.show()

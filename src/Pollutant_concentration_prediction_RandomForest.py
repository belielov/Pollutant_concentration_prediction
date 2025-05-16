import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

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

# ----------------- 填充方法 -----------------


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


# ----------------- 贝叶斯优化器配置 -----------------
rf_params = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'max_features': Real(0.5, 1.0, prior='uniform'),
    'bootstrap': Categorical([True, False])
}

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

    # 遍历每个目标变量
    for target_idx, target_name in enumerate(target_columns):
        # 贝叶斯优化
        opt = BayesSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            search_spaces=rf_params,
            n_iter=30,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        opt.fit(X_scaled, y[:, target_idx])

        # 保存最优参数
        optimized_params[target_name][method] = opt.best_params_
        print(f'Best params ({method}): {opt.best_params_}')
        print(f'Best R2 ({method}): {opt.best_score_:.3f}')

        # 使用最优参数进行交叉验证
        model = RandomForestRegressor(
            **opt.best_params_,
            random_state=42
        )

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model,
            X_scaled,
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
plt.savefig(os.path.join(output_imgs, 'fill_method_comparison_BayesRF_without_scaled.png'), dpi=300)
plt.show()

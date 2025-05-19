import os
import numpy as np
import pandas as pd
import xgboost as xgb
from minepy import MINE
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------ 数据导入 ------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
dataset_path = os.path.join(project_dir, 'dataset', 'GF5A.xlsx')
XY = pd.read_excel(dataset_path).values  # 将pandas的DataFrame转换为Numpy数组(不包括第一个样本点)
X = XY[:, :330].astype(float)  # 将数据强制转换为浮点类型
Y = XY[:, 330].astype(float)  # 总氮浓度

# ------------------ 特征筛选 ------------------
# 初始化MINE对象，用于计算最大信息系数（MIC）以衡量特征与目标变量的非线性相关性
mine = MINE()

valid_features_mic = []  # 存储通过 MIC 筛选的有效特征索引
valid_features_pearson = []  # 存储通过 Pearson 相关系数筛选的有效特征索引

# 遍历所有波段，筛选有效特征
for i in range(330):
    feature = X[:, i]

    # 跳过常数列（方差为零）
    if np.var(feature) == 0:
        continue  # 终止本次循环

    # 计算 MIC 值
    mine.compute_score(feature, Y)  # 计算特征与目标变量 Y 的关联性
    mic_value = mine.mic()  # 获取 MIC 值（最大信息系数）

    # 计算 Pearson 相关系数
    try:
        correlation, _ = pearsonr(feature, Y)
    except ValueError:
        continue  # 捕获输入数据不合法（如 NaN、无穷大值）的异常

    # MIC 筛选（阈值0.45）
    if mic_value > 0.45:
        valid_features_mic.append(i)

    # Pearson 筛选（阈值0.39）
    if abs(correlation) > 0.39:
        valid_features_pearson.append(i)

# 打印筛选后的特征数量
print(f'经过 MIC 筛选的有效特征数量：{len(valid_features_mic)}')
print(f'经过 Pearson 筛选的有效特征数量：{len(valid_features_pearson)}')

# ------------------ 特征组合（MIC筛选部分） ------------------
# 初始化 save 变量（默认使用 MIC 筛选特征）
save = np.array([])  # 初始化为空数组

if len(valid_features_mic) > 0:
    # 根据索引提取原始数据中的特征列
    save = X[:, valid_features_mic]  # 形状 (样本数, 有效特征数)

    # 仅在特征数小于40时生成组合特征（避免内存风险）
    if 1 < save.shape[1] < 40:
        num_original = save.shape[1]
        for i in range(num_original):
            for j in range(i + 1, num_original):
                bi = save[:, i].reshape(-1, 1)  # 提取第 i 列特征并转为列向量
                bj = save[:, j].reshape(-1, 1)  # 提取第 j 列特征并转为列向量
                # 安全除法（防止分母为零）
                eps = 1e-10
                save = np.hstack((  # 将新生成的组合特征按列追加到 save 中
                    save,
                    bi + bj,
                    bi - bj,
                    bi / (bj + eps),  # 添加极小值避免除零
                    bi * bj
                ))
        # 清除无效值
        save = np.nan_to_num(
            save,
            nan=0.0,  # 将 NaN（非数值）替换为 0
            posinf=1e5,  # 将正无穷大替换为 100000
            neginf=-1e5  # 将负无穷大替换为 -100000
        )

    # 打印组合后的特征数量
    print(f'MIC 筛选部分经过组合后的特征数量：{save.shape[1]}')

# ------------------ 使用Pearson筛选的特征 ------------------
# 若 MIC 特征为空，则使用 Pearson 特征
if save.size == 0:
    if len(valid_features_pearson) == 0:
        raise ValueError(f'未筛选到有效特征！请调整相关系数阈值')
    save = X[:, valid_features_pearson]
    print(f'使用 Pearson 筛选的特征进行训练')

# ------------------ 模型训练与评估 ------------------
x_train, x_test, y_train, y_test = train_test_split(
    save, Y, test_size=0.2, random_state=42
)

# 初始化模型（显式处理缺失值）
model = xgb.XGBRegressor(
    max_depth=5,
    learning_rate=0.33,
    n_estimators=400,
    missing=np.nan
)

model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)

print("测试集评估结果：")
print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"决定系数 (R²): {r2_score(y_test, y_pred_test):.4f}")

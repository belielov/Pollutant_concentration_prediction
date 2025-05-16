import os.path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# # 设置显示选项以完整输出
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)


# ---------------- 数据导入 ----------------

# 获取当前脚本文件的绝对路径并提取所在目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 构建数据集所在路径
dataset_path = os.path.join(project_dir, 'dataset', 'GF5A_v3.xlsx')
# 构建输出路径
output_imgs = os.path.join(project_dir, 'imgs', 'GF5A_v3')

# # 测试：打印路径
# print(f'''
# 当前脚本所在目录为：{current_dir}
# 项目路径为：{project_dir}
# 数据集所在路径为：{dataset_path}
# ''')

df = pd.read_excel(dataset_path, header=0)

# # 测试：打印读取的数据
# print("列名：", df.columns.tolist())  # 显示Excel第一行的内容
# print("前5行数据：\n", df.head())  # 数据从Excel第二行开始

# # 查看并统计缺失值
# print("缺失值统计（填充前）：\n", df.isnull().sum())
# print(f'数据集第334列：{df.iloc[:, 333].values}')  # iloc 通过位置访问数据，避免了依赖列名可能带来的问题
# print(f'数据集第326列：{df.iloc[:, 325].values}')
# print(f'数据集第330列：{df.iloc[:, 329].values}')
# sns.heatmap(df.isnull(), cbar=False)  # 用热力图展示缺失值位置
# plt.show()

# ---------------- 数据预处理 ----------------

# 提取前300列作为光谱数据，剩余其他列作为污染物浓度数据
spectral_data = df.iloc[:, :294]
pollutant_data = df.iloc[:, 294:]
feature_names = spectral_data.columns.tolist()

# # zero filling
# df.fillna(0, inplace=True)  # 直接修改原DataFrame，所有缺失值替换为0

# # mean filling
# df.fillna(df.mean(), inplace=True)


# # median filling
# df.fillna(df.median(), inplace=True)

# custom filling(KNN+PCA+RandomForest)
# KNN填充
imputer_spectral = KNNImputer(n_neighbors=5)
spectral_data_filled = pd.DataFrame(
    imputer_spectral.fit_transform(spectral_data),
    columns=spectral_data.columns,
    index=spectral_data.index
)

imputer_pollutant = KNNImputer(n_neighbors=5)
pollutant_data_filled = pd.DataFrame(
    imputer_pollutant.fit_transform(pollutant_data),
    columns=pollutant_data.columns,
    index=pollutant_data.index
)

# 合并填充后的数据
df = pd.concat([spectral_data_filled, pollutant_data_filled], axis=1)

# # 验证填充效果
# print("缺失值统计（填充后）：\n", df.isnull().sum())
# print(f'数据集第334列：{df.iloc[:, 333].values}')
# print(f'数据集第326列：{df.iloc[:, 325].values}')
# print(f'数据集第330列：{df.iloc[:, 329].values}')
# sns.heatmap(df.isnull(), cbar=False)  # 用热力图展示缺失值位置
# plt.show()

# ---------------- 可视化采样点不同波段的光谱值 ----------------

# 从填充后的总数据中提取光谱数据
spectral_data_filled = df.iloc[:, :294]

# 创建画布
plt.figure(figsize=(16, 8), dpi=100)

# 设置坐标轴
plt.title('Distribution of spectral values across different bands at sampling points', fontsize=14)
plt.xlabel('Band index', fontsize=12)
plt.ylabel('Spectral values', fontsize=12)
plt.grid(alpha=0.3)

# # 测试：打印光谱图像相关数据
# print(f'''
# 提取的光谱数据（前5行）：\n {spectral_data_filled.head()}
# len(spectral_data) = {len(spectral_data_filled)}
# spectral_data.columns：\n {spectral_data_filled.columns}
# ''')

# 绘制所有样本的光谱曲线
for idx in range(len(spectral_data_filled)):
    plt.plot(
        spectral_data_filled.columns,  # x轴：列名
        spectral_data_filled.iloc[idx, :],  # y轴：某采样点的光谱值
        linewidth=0.8,
        alpha=0.4,  # 设置透明度避免完全重叠
        color='steelblue'
    )

# 在 y=0 的位置添加一条水平参考线
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

# 优化显示
plt.xlim(spectral_data_filled.columns[0], spectral_data_filled.columns[-1])
plt.xticks(rotation=45, fontsize=8)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))  # 自动计算间隔，最多显示 20 个标签
plt.tight_layout()

# 保存或显示图像
plt.savefig(os.path.join(output_imgs, 'spectral_value', 'spectral_curves_KNNfilling.png'), bbox_inches='tight')
plt.show()
plt.close()

# # ---------------- 特征工程 ----------------
# # 定义目标变量列名
# target_columns = ['总氮浓度', '总磷浓度', '氨氮浓度']  # 根据实际列名修改
#
# # 提取特征和目标变量
# X = spectral_data_filled.values  # 光谱数据 (112 samples, 294 features)
# y = df[target_columns].values  # 多目标矩阵 (112 samples, 3 targets)
#
# # # 测试：打印提取的目标变量
# # print(f'提取的目标变量：\n{y}')
#
# # 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # # PCA降维（保留95%方差）
# # pca = PCA(n_components=0.95)
# # X_pca = pca.fit_transform(X_scaled)
# # print(f"原始特征数：{X.shape[1]}，PCA后特征数：{X_pca.shape[1]}")
#
# # # 创建画布
# # plt.figure(figsize=(10, 6))
# #
# # # 设置坐标轴
# # n_components = len(pca.explained_variance_ratio_)  # 生成横轴坐标（主成分数量从1开始）
# # x = range(1, n_components + 1)
# # plt.xticks(x)  # 强制刻度为整数
# # plt.xlabel('Number of Components')
# # plt.ylabel('Cumulative Explained Variance')
# # plt.title('PCA Explained Variance Ratio')
# #
# # # 绘制方差累积曲线
# # plt.plot(x, np.cumsum(pca.explained_variance_ratio_), marker='o')
#
# # # 保存或显示图像
# # plt.savefig(os.path.join(output_imgs, 'pca_variance.png'))
# # plt.show()
# # plt.close()
#
# # ---------------- XGBoost模型训练 ----------------
# results = {}
#
# # 定义超参数搜索空间
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
#
# # # 定义参数分布空间（比网格搜索更灵活）
# # param_dist = {
# #     'n_estimators': randint(100, 500),        # 整数均匀分布：100-500
# #     'max_depth': randint(3, 10),              # 整数均匀分布：3-10
# #     'learning_rate': uniform(0.01, 0.3),      # 连续均匀分布：0.01-0.3
# #     'subsample': uniform(0.6, 0.4),           # 连续均匀分布：0.6-1.0
# #     'colsample_bytree': uniform(0.6, 0.4),    # 连续均匀分布：0.6-1.0
# #     'gamma': uniform(0, 0.5),                 # 新增正则化参数
# #     'reg_alpha': uniform(0, 1),               # 新增L1正则化
# #     'reg_lambda': uniform(0, 1)               # 新增L2正则化
# # }
#
# # 创建预测结果散点图画布
# plt.figure(figsize=(15, 5))
#
# # 遍历每个目标变量
# for target_idx, target_name in enumerate(target_columns):
#     print(f'\n=== 正在训练模型：{target_name} ===')
#
#     # 提取目标变量
#     y_target = y[:, target_idx]
#
#     # # 测试：打印提取的目标变量
#     # print(f'提取的目标变量：\n{y_target}')
#
#     # 划分数据集
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y_target, test_size=0.2, random_state=42
#     )
#
# #     # 打印划分的数据集
# #     print(f'''
# #     X_train:{X_train}
# #     y_train:{y_train}
# #     X_test:{X_test}
# #     y_test:{y_test}
# # ''')
#
#     # 初始化模型
#     model = xgb.XGBRegressor(
#         objective='reg:squarederror',  # 显式声明损失函数为平方误差（MSE）
#         random_state=42,
#         n_jobs=-1  # 启用模型内置并行
#     )
#
#     # 初始化网格搜索
#     grid_search = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         scoring='r2',
#         cv=KFold(n_splits=3, shuffle=True, random_state=42),
#         verbose=1,  # 用于控制程序运行时信息输出详细程度的参数2
#         n_jobs=-1,  # 使用所有CPU核心
#     )
#
#     # 执行网格搜索
#     grid_search.fit(X_train, y_train)
#
#     # # 初始化随机搜索
#     # random_search = RandomizedSearchCV(
#     #     estimator=model,
#     #     param_distributions=param_dist,
#     #     n_iter=50,  # 随机采样次数（可根据计算资源调整）
#     #     scoring='r2',
#     #     cv=KFold(n_splits=3, shuffle=True, random_state=42),
#     #     verbose=1,
#     #     n_jobs=-1,  # 使用所有CPU核心
#     #     random_state=42
#     # )
#     #
#     # # 执行参数搜索
#     # random_search.fit(X_train, y_train)
#
#     # 获取最佳模型
#     best_model = grid_search.best_estimator_
#
#     # 打印最佳参数
#     print(f'\n最佳参数组合：{grid_search.best_params_}')
#     print(f'最佳验证R2：{grid_search.best_score_:.4f}')
#
#     # # 打印搜索报告
#     # print(f'\n=== {target_name} 最佳参数 ===')
#     # print(random_search.best_params_)
#     # print(f'最佳验证R2: {random_search.best_score_:.4f}')
#
#     # 预测结果
#     y_pred = best_model.predict(X_test)
#
#     # 评估指标
#     r2 = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#
#     # 交叉验证
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_scores = cross_val_score(
#         best_model,
#         X_scaled,
#         y_target,
#         cv=kfold,
#         scoring='r2'
#     )
#
#     # 保存结果
#     results[target_name] = {
#         'model': best_model,
#         'best_param': grid_search.best_params_,
#         'r2': r2,
#         'rmse': rmse,
#         'cv_mean': cv_scores.mean(),
#         'cv_std': cv_scores.std()
#     }
#
#     # 打印结果
#     print(f'''
#     {target_name} 模型评估：
#     - 测试集R²：{r2:.4f}
#     - 测试集RMSE：{rmse:.4f}
#     - 交叉验证R²：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}
# ''')
#
#     # 绘制预测结果散点图
#     plt.subplot(1, 3, target_idx + 1)
#     plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
#     plt.plot(
#         [min(y_test), max(y_test)],
#         [min(y_test), max(y_test)],
#         color='red',
#         linestyle='--'
#     )
#     plt.xlabel('真实值')
#     plt.ylabel('预测值')
#     plt.title(f'{target_name}预测结果  R2 = {results[target_name]['r2']:.3f}')
#
# plt.tight_layout()
# plt.savefig(os.path.join(output_imgs, 'grid_search', 'medianfilling', 'prediction_scatter.png'))
# plt.close()
#
# # ---------------- 保存最佳模型 ----------------
#
# # 创建模型保存目录
# model_dir = os.path.join(project_dir, 'models', 'GF5A_v2', 'grid_search', 'medianfilling')
# os.makedirs(model_dir, exist_ok=True)
#
# # 保存每个目标的模型
# for target_name in target_columns:
#     joblib.dump(
#         results[target_name]['model'],
#         os.path.join(model_dir, f'best_model_{target_name}.pkl')  # Python Pickle 序列化文件的标准后缀名
#     )
#
# # ---------------- 结果可视化 ----------------
#
# # 绘制特征重要性
# for target_name in target_columns:
#     fig, ax = plt.subplots(figsize=(12, 8))
#     xgb.plot_importance(results[target_name]['model'], ax=ax, max_num_features=20)
#     plt.title(f'{target_name}特征重要性 Top 20')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_imgs, 'grid_search', 'medianfilling', f'feature_importance_{target_name}.png'))
#     plt.close()
#
# # 绘制交叉验证结果
# plt.figure(figsize=(10, 6))
# for target_name in target_columns:
#     # # 打印目标列索引
#     # print(f'{target_name}的列索引 : {target_columns.index(target_name)}')
#
#     cv_scores = cross_val_score(
#         results[target_name]['model'],
#         X_scaled,
#         y[:, target_columns.index(target_name)],
#         cv=KFold(n_splits=5, shuffle=True, random_state=42),
#         scoring='r2'
#     )
#     plt.plot(range(1, 6), cv_scores, marker='o', label=target_name)
#
# # 显式设置横轴刻度
# plt.xticks(ticks=range(1, 6), labels=['1', '2', '3', '4', '5'])
#
# plt.xlabel('折叠次数')
# plt.ylabel('R2 Score')
# plt.title('交叉验证结果（5折）')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(output_imgs, 'grid_search', 'medianfilling', 'cross_validation.png'))
# plt.close()

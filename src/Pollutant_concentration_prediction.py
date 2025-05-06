import os.path

import matplotlib.pyplot as plt

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 数据导入 ----------------
# 获取当前脚本文件的绝对路径并提取所在目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 构建数据集所在路径
dataset_path = os.path.join(project_dir, 'dataset', 'GF5A.xlsx')

# # 测试：打印路径
# print(f'''
# 当前脚本所在目录为：{current_dir}
# 项目路径为：{project_dir}
# 数据集所在路径为：{dataset_path}
# ''')

# ---------------- 数据预处理 ----------------

# zero filling

# mean filling

# median filling

# custom filling(KNN+PCA+RandomForest)

# ---------------- XGBoost模型训练 ----------------
# 保存最佳模型

# ---------------- 交叉验证模型稳定性 ----------------

# ---------------- 结果可视化 ----------------
# 横轴为折叠次数，纵轴为 R²score

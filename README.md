# Pollutant_concentration_prediction
本项目使用的数据集为 GF5A_v2.xlsx ，共有330列，112行。每一行代表一个采样点，每一列代表一个波段。

dataset 文件夹下的 GF5A_origin.xlsx 文件，由于未知格式问题，无法将第一行作为列名；将第一行手动输入，其余数据复制后得到的 GF5A_v2.xlsx 文件可以将第一行作为列名。

源代码位于 src 文件夹，训练结果以图片的形式存储在 imgs 文件夹。

训练得到的模型位于 models 文件夹。

src 文件夹下的 delete_random_data.py 用于随机删除指定数量的光谱数据单元格并保存结果。
***
## XGBoost SimulateAnnealing HPO Results
```python
fill_methods = ['zero', 'mean', 'median', 'knn']  # 外循环
target_columns = ['总氮浓度', '总磷浓度', '氨氮浓度']  # 内循环
best_params:{'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 276, 'subsample': 0.664, 'colsample_bytree': 0.811}; best_score:0.0693
best_params:{'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 230, 'subsample': 0.83, 'colsample_bytree': 0.634}; best_score:-1.9026
best_params:{'learning_rate': 0.045, 'max_depth': 4, 'n_estimators': 157, 'subsample': 0.849, 'colsample_bytree': 0.6}; best_score:0.2679
best_params:{'learning_rate': 0.275, 'max_depth': 6, 'n_estimators': 124, 'subsample': 0.782, 'colsample_bytree': 0.684}; best_score:0.0303
best_params:{'learning_rate': 0.128, 'max_depth': 3, 'n_estimators': 289, 'subsample': 0.788, 'colsample_bytree': 1.0}; best_score:-0.8225
best_params:{'learning_rate': 0.076, 'max_depth': 3, 'n_estimators': 202, 'subsample': 0.764, 'colsample_bytree': 0.6}; best_score:0.3144
best_params:{'learning_rate': 0.012, 'max_depth': 8, 'n_estimators': 240, 'subsample': 0.91, 'colsample_bytree': 0.851}; best_score:0.0705
best_params:{'learning_rate': 0.298, 'max_depth': 3, 'n_estimators': 235, 'subsample': 0.728, 'colsample_bytree': 0.933}; best_score:-0.7827
best_params:{'learning_rate': 0.075, 'max_depth': 3, 'n_estimators': 166, 'subsample': 0.779, 'colsample_bytree': 0.748}; best_score:0.3003
best_params:{'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 116, 'subsample': 0.677, 'colsample_bytree': 0.884}; best_score:0.0542
best_params:{'learning_rate': 0.183, 'max_depth': 3, 'n_estimators': 182, 'subsample': 0.83, 'colsample_bytree': 0.976}; best_score:-0.9194
best_params:{'learning_rate': 0.155, 'max_depth': 3, 'n_estimators': 174, 'subsample': 0.6, 'colsample_bytree': 0.974}; best_score:0.2761
```
***
## RandomForest Bayes HPO Results
```python
Best params (zero): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200})
Best R2 (zero): -0.182
Best params (zero): OrderedDict({'bootstrap': True, 'max_depth': 5, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200})
Best R2 (zero): -0.165
Best params (zero): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50})
Best R2 (zero): -4.288
Best params (mean): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.6178643954980271, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 71})
Best R2 (mean): -0.142
Best params (mean): OrderedDict({'bootstrap': True, 'max_depth': 22, 'max_features': 0.7471364524106656, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200})
Best R2 (mean): -0.161
Best params (mean): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 50})
Best R2 (mean): -4.087
Best params (median): OrderedDict({'bootstrap': True, 'max_depth': 6, 'max_features': 0.6981000476866821, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 106})
Best R2 (median): -0.105
Best params (median): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 50})
Best R2 (median): -0.179
Best params (median): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50})
Best R2 (median): -4.095
Best params (knn): OrderedDict({'bootstrap': True, 'max_depth': 18, 'max_features': 0.9629793842913454, 'min_samples_leaf': 3, 'min_samples_split': 9, 'n_estimators': 103})
Best R2 (knn): -0.089
Best params (knn): OrderedDict({'bootstrap': True, 'max_depth': 5, 'max_features': 0.5, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200})
Best R2 (knn): -0.138
Best params (knn): OrderedDict({'bootstrap': True, 'max_depth': 30, 'max_features': 0.5679931616940275, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50})
Best R2 (knn): -4.110
```
***
## SVM SimulateAnnealing HPO Results
```python
总氮浓度 | zero - 最优参数:{'C': np.float64(0.2635988960444668), 'gamma': np.float64(0.014074296813429952)}, R2:-0.117
总磷浓度 | zero - 最优参数:{'C': np.float64(72.74701902324198), 'gamma': np.float64(0.039621336468497734)}, R2:-1.128
氨氮浓度 | zero - 最优参数:{'C': np.float64(43.93147321286425), 'gamma': np.float64(0.06365511538407016)}, R2:0.065
总氮浓度 | mean - 最优参数:{'C': np.float64(0.8675297704356162), 'gamma': np.float64(0.0015946752459717827)}, R2:0.169
总磷浓度 | mean - 最优参数:{'C': 0.1, 'gamma': np.float64(0.004320550759456012)}, R2:-0.275
氨氮浓度 | mean - 最优参数:{'C': np.float64(8.280050518059019), 'gamma': np.float64(0.04335564618363529)}, R2:0.325
总氮浓度 | median - 最优参数:{'C': np.float64(1.1710508996788833), 'gamma': np.float64(0.009448201272095516)}, R2:0.120
总磷浓度 | median - 最优参数:{'C': 0.3259810600405378, 'gamma': np.float64(0.0012375311297060556)}, R2:-0.268
氨氮浓度 | median - 最优参数:{'C': np.float64(5.475643755043997), 'gamma': np.float64(0.0014628489215848994)}, R2:0.265
总氮浓度 | knn - 最优参数:{'C': np.float64(0.7212049393273052), 'gamma': np.float64(0.0013971778755755156)}, R2:0.161
总磷浓度 | knn - 最优参数:{'C': np.float64(55.99460439868883), 'gamma': np.float64(0.4757360432697404)}, R2:-0.987
氨氮浓度 | knn - 最优参数:{'C': np.float64(30.148189052038298), 'gamma': np.float64(0.0002681486098742721)}, R2:0.258
```
***
## KNN(K Nearest Neighbors)填充
当数据集缺失数据时，基于K临近算法进行数据补充的一种方法。
使用数据集中与缺失样本最相似的K个样本的值进行填充。

KNN算法，也称为K最邻近算法，是**有监督学习**中的**分类算法**。它可以用于分类或回归问题，但通常用作分类算法。
### 主要步骤
1. **选择k值**：决定利用几个临近点来估算。K的取值可以依赖交叉验证或经验所得。
2. **计算距离**：计算缺失样本与其他样本的距离，通常是欧氏距离。
3. **选择最邻近的点**：选择最邻近的K个点。
4. **填充缺失值**：根据这K个点来估算缺失值。例如众数，中位数，平均数等。
### 核心思想
- 分类问题：根据它距离最近的K个样本点是什么类别来判断该新样本属于哪个类别（多数投票）
- 回归问题：寻找预测实例的k近邻，对这k个样本的目标值取**均值**即可作为新样本的预测值。

[详细说明](https://blog.csdn.net/m0_74405427/article/details/133714384?fromshare=blogdetail&sharetype=blogdetail&sharerId=133714384&sharerefer=PC&sharesource=bg_de_father&sharefrom=from_link)
***
## 运行结果的说明
### R² 的核心概念
#### 1. 定义
$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$
- $SS_{res}$ : 模型预测的残差平方和（预测值与真实值的差的平方和）。
- $SS_{tot}$ : 总平方和（真实值与其均值的差的平方和）。
#### 2. 理论范围
- **最佳情况**：$R^2 = 1$ （模型完美拟合数据）
- **基准情况**：$R^2 = 0$ （模型预测效果等于直接用均值预测）
- **可能负值**：当模型预测效果比均值预测更差时（$SS_{res} > SS_{tot}$）, $R^2$会出现负数。
### XGBoost 的特征重要性分数
XGBoost 的特征重要性分数是通过模型训练过程中特征的使用情况计算得出的相对重要性，其范围和计算方式如下：
***
1. 特征重要性的计算方式
XGBoost 支持三种计算方法，默认使用`weight`：
- `weight`（权重）：统计特征在所有树中被用作分裂节点的次数。次数越多，重要性越高。
  - 范围：取决于数据集和模型复杂度，可能从 0 到数百甚至更高。
  - 特点：简单直观，但可能偏向连续型或高基数特征。
- `gain`（增益）：统计特征在所有树中作为分裂节点时带来的平均损失函数减少量（如信息增益）。
  - 范围：通常为 0 到正数，具体值取决于模型对损失的优化程度。
  - 特点：更直接反映特征对模型性能的实际贡献。
- `cover`（覆盖）：统计特征在所有树中作为分裂节点时覆盖的样本量（即受该节点影响的样本数）。
  - 范围：取决于样本量和树结构，可能从 0 到数万。
  - 特点：反映特征对数据分布的全局影响。
2. 范围问题

## 为什么推荐使用 iloc 访问数据
### 1. 避免列名歧义
- `df[0]`的语法会被Pandas解释为“访问列名为`0`的列”，而非“第一列”。如果列名是字符串（如`"日期"`、`"浓度"`），`df[0]`会直接抛出`KeyError`，因为列名`0`不存在。
- `df.iloc[:, 0]`则明确表示“所有行（`:`）的第`0`列（第一列）”，完全基于位置索引，与列名无关，**避免列名冲突或误解**。
### 2. 处理动态数据更安全
- 如果数据预处理（如填充缺失值、特征选择、列重排序）导致列顺序或列名变化，`iloc`可以确保始终访问正确的列位置，而无需依赖列名。
- 例如，假设原始数据第一列是`"PM2.5"`，但经过处理后列顺序变为`["温度", "PM2.5"]`，此时`df.iloc[:, 0]`仍指向新的第一列（`"温度"`），而`df["PM2.5"]`需要手动调整列名引用。
### 3. 代码通用性更强
- 当处理多个不同结构的数据集时，列名可能不一致（例如有的数据集第一列叫`"Date"`，有的叫`"时间"`），使用`iloc`可以统一用位置索引访问数据，**减少代码适配成本**。
### 4. 避免隐式依赖
- `df[0]`依赖列名是否为整数，这种隐式依赖容易导致代码脆弱性。例如，若某次数据导入时列名被自动转换为字符串（如`"0"`），`df[0]`会失败，而`df.iloc[:, 0]`始终有效。
***
## 横轴标签过密
在 matplotlib 中可以通过以下两种方式解决横轴标签过密问题：
```python
# 优化显示
# 强制设置 x 轴的显示范围
# 取 DataFrame 列名的第一个值作为起点
# 取列名的最后一个值作为终点
plt.xlim(spectral_data.columns[0], spectral_data.columns[-1])  

# 方法 1：设置自定义刻度间隔（推荐）
step = 50  # 每隔 50 个波段显示一个标签
plt.xticks(
    ticks=spectral_data.columns[::step],  # 指定显示位置
    labels=spectral_data.columns[::step],  # 指定显示文本
    rotation=45,  # 标签旋转 45 度
    fontsize=8    # 缩小字体
)

# 方法 2：自动间隔 + 旋转（通用型）
plt.xticks(rotation=45, fontsize=8)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))  # 自动计算间隔，最多显示 20 个标签

plt.tight_layout()
```
***
## MIC值与Pearson相关系数的范围及意义
***
### Pearson相关系数
- 范围：`-1` 到 `1`
  - 1：完全正线性相关
  - -1：完全负线性相关
  - 0：无线性相关性
- 数值大小含义：
  - |r| > 0.7：强线性相关
  - 0.3 < |r| < 0.7：中等线性相关
  - |r| < 0.3：弱或无线性相关
- 特点：
  - 仅衡量线性关系，对非线性关系不敏感。
  - 对异常值敏感，可能因个别极端值导致误导性结果。
  - 要求数据服从正态分布（严格假设）。
- 示例：若特征与目标变量的Pearson系数为`0.8`，说明两者存在强正线性相关，适合用于线性模型。
***
### MIC（最大信息系数）
- 范围：`0` 到 `1`
  - 1：变量间存在完全的非线性或规律性关系（如二次函数、正弦曲线）。
  - 0：变量间完全独立，无任何关联。
- 数值大小含义：
  - MIC > 0.7 : 强非线性相关。
  - 0.3 < MIC < 0.7 : 中等非线性相关。
  - MIC < 0.3 : 弱或无相关性。
- 特点：
  - 捕捉**任意形式的关联性**（线性、非线性、周期性等）。
  - 对噪声和样本量有一定鲁棒性。
  - 计算复杂度较高，适合特征数较少时的筛选。
- 示例：若特征与目标变量的MIC值为`0.6`，说明两者存在中等程度的非线性相关，可能对树模型（如XGBoost）有重要贡献。
***
## 代码语法说明
### 1. `df.head()`方法
`df.head()` 是 Pandas 库中 **DataFrame 对象的一个方法**，用于快速查看数据集的前几行数据。
***
#### 语法
``` python
df.head(n=5)
```
- **参数** `n`：可选参数，指定要显示的行数。默认 `n=5`（显示前 5 行）。
- **返回值**：返回一个新的 DataFrame，包含前 `n` 行数据，**不修改原始 DataFrame**。
***
#### 作用
- **快速预览数据**：默认显示 DataFrame 的前 5 行数据（不含列名），方便快速了解数据的结构、内容和格式。
- **避免加载全部数据**：当数据集非常大时，直接打印全部数据会占用大量内存或导致界面卡顿，`head()` 仅展示前几行，高效且实用。
### 2. `df.isnull()`方法
`df.isnull()` 是 Pandas 库中 **DataFrame 或 Series 对象的方法**，用于检测数据中的缺失值（如 `NaN`、`None`）。
***
#### 语法
```python
df.isnull()
```
- **输入**：无需参数。
- **输出**：返回一个与原 DataFrame/Series **形状相同**的布尔型 DataFrame/Series，其中：
  - `True`：表示该位置的值为缺失值。
  - `False`：表示该位置的值非缺失。
***
#### 作用
- **定位缺失值**：快速识别数据集中哪些位置存在缺失值。
- **统计缺失值数量**：结合 `.sum()` 方法，统计每列（或每行）的缺失值数量。
- **条件筛选**：配合布尔索引，筛选出包含（或排除）缺失值的行或列。
***
#### 常见用法
- **全局缺失值统计**
```python
df.isnull().sum().sum()  # 统计整个DataFrame的缺失值总数
```
- **按列或行统计缺失值**
```python
df.isnull().sum()  # 每列的缺失值数量
df.isnull().sum(axis=1)  # 每行的缺失值数量
```
- **可视化缺失值分布**
```python
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False)  # 用热力图展示缺失值位置
```
### 3. `sns.heatmap()`方法
`seaborn.heatmap()` 是用于绘制热力图的函数，常用于可视化二维数据的矩阵形式（如相关性矩阵、缺失值分布或数值密度）。
***
#### 基本语法
```python
sns.heatmap(
    data,                  # 必需参数：输入数据（二维数组、DataFrame或类似结构）
    vmin=None,             # 颜色映射的最小值
    vmax=None,             # 颜色映射的最大值
    cmap=None,             # 颜色方案（如 'viridis', 'coolwarm', 'YlGnBu'）
    center=None,           # 颜色映射的中心值（用于发散型数据）
    annot=None,            # 是否在单元格中显示数值（True/False 或与data同形的数组）
    fmt='.2g',             # 数值显示的格式（如 '.2f' 表示保留两位小数）
    linewidths=0,          # 单元格边框宽度
    linecolor='white',     # 单元格边框颜色
    cbar=True,             # 是否显示颜色条
    cbar_kws=None,         # 颜色条参数（如 {'label': 'Title'}）
    mask=None,             # 掩盖部分数据（True的位置不显示）
    square=False,          # 单元格是否为正方形
    ax=None                # 指定绘图的Axes对象
)
```
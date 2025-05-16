import os
import random

import numpy as np
import pandas as pd


def delete_spectral_cells(dataset_path, output_excel, num_cells_to_delete):
    """
    随机删除光谱数据单元格
    :param dataset_path: 输入excel文件路径
    :param output_excel: 输出excel文件路径
    :param num_cells_to_delete: 要删除的单元格数量
    :return:
    """
    # 读取excel文件（第一行为标题）
    df = pd.read_excel(dataset_path, header=0)

    # 验证输入参数
    total_spectral_cells = df.shape[0] * 294  # 总数据行数 x 光谱列数
    if num_cells_to_delete > total_spectral_cells:
        raise ValueError(f'要删除的单元格数量不能超过光谱区域总单元格数{total_spectral_cells}')

    # 生成所有可能的光谱数据位置（行索引，列索引）
    all_positions = [(r, c) for r in range(df.shape[0]) for c in range(294)]

    # 随机选择要删除的单元格位置
    selected_positions = random.sample(all_positions, num_cells_to_delete)

    # 将选中位置设为NaN
    for row, col in selected_positions:
        df.iloc[row, col] = np.nan

    # 保存处理后的数据
    df.to_excel(output_excel, index=False)
    print(f'已成功删除{num_cells_to_delete}个单元格，结果保存至：{output_excel}')


# 主程序
if __name__ == "__main__":
    # 输入参数设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(project_dir, 'dataset', 'GF5A_delete_empty_columns.xlsx')
    output_excel = os.path.join(project_dir, 'dataset', 'GF5A_v3.xlsx')
    num_cells_to_delete = 1000

    # 执行处理
    delete_spectral_cells(dataset_path, output_excel, num_cells_to_delete)

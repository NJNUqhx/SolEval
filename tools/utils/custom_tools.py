import os
import json
import re
import pickle

def fix_missing_brackets(code: str) -> str:
    """
    检查代码中的花括号是否匹配。
    如果左括号多于右括号，则在末尾补齐缺失的右括号。
    """
    # 统计左右括号的数量
    left_count = code.count('{')
    right_count = code.count('}')
    
    # 如果左括号更多，计算差值并补齐
    if left_count > right_count:
        missing_count = left_count - right_count
        # 在末尾添加缺失的右括号，每个括号占一行以保证格式整齐
        code = code.rstrip() + '\n' + ('}' * missing_count)
        
    return code

def build_pattern_mapping(args):
    """
    构建 pattern 到 file_path 的映射字典。
    
    参数:
    - data: 原始数据集字典 {file_path: [methods...]}
    - args: 包含模型和实验配置的参数对象 (model, shot, context, testcase)
    - real_path_cargo: 路径映射字典，用于提取文件名
    
    返回:
    - pattern_to_path: 映射字典 {pattern: file_path}
    """
    
    with open('/root/contract2solidity/SolEval/data/dataset.json', 'r') as file:
        data = json.load(file)
    
    real_path_cargo = pickle.load(open("/root/contract2solidity/SolEval/prebuilt/real_path_cargo.pkl", "rb"))
    
    pattern_to_path = {}

    context_or_not = args.context
    
    if context_or_not == "y":
        context = "context_True_testcase_False"
    elif context_or_not == "n":
        context = "context_False_testcase_False"
    elif context_or_not == "c":
        context = "context_False_testcase_True"
    elif context_or_not == "h":
        context = "context_True_testcase_True"
    else:
        raise NotImplementedError("Invalid input for context_or_not!!!")
    
    # 1. 遍历数据构建映射
    for file_path, file_content in data.items():
        if file_path not in real_path_cargo:
            continue
            
        # 提取文件名 (例如: Contract.sol)
        file_name = real_path_cargo[file_path].split('/')[-1]
        
        for method in file_content:
            identifier = method.get('identifier', 'unknown')
            
            # 构建 pattern 字符串
            pattern = (
                f"patch/rag/{args.model}_shot_{args.shot}_"
                f"{context}/"
                f"patch_{file_name}_function_{identifier}_*"
            )
            
            pattern_to_path[pattern] = file_path
              
    return pattern_to_path

def get_original_path_by_patch(current_patch_path, mapping_dict):
    """
    通过当前补丁路径反向查找原始文件路径。
    
    Args:
        current_patch_path (str): 实际生成的补丁文件路径。
        mapping_dict (dict): 预先构建的映射字典。
    """
    for pattern, original_file_path in mapping_dict.items():
        # 去掉通配符，进行前缀匹配
        pattern_prefix = pattern.rstrip('*')
        if current_patch_path.startswith(pattern_prefix):
            return original_file_path
    return None

import pandas as pd

def export_execution_metrics(project_status, output_file="project_metrics_report.csv"):
    """
    将项目统计字典转换为 DataFrame 并导出，保留原始计数。
    
    Args:
        project_status (dict): 格式为 {path: {"compile": c, "pass": p, "total": t}}
        output_file (str): 导出的文件名
    """
    # 1. 从字典加载，orient='index' 保证 path 是行索引
    df_metrics = pd.DataFrame.from_dict(project_status, orient='index')

    # 2. 将索引（文件路径）转换为普通列，并命名
    df_metrics.index.name = 'file_path'
    df_metrics = df_metrics.reset_index()

    # 3. 在保留原始 count 的基础上，添加计算列
    # 使用填充 0 处理 total 为 0 的异常情况
    df_metrics['compile_rate'] = (df_metrics['compile'] / df_metrics['total']).fillna(0)
    df_metrics['pass_rate'] = (df_metrics['pass'] / df_metrics['total']).fillna(0)

    # 4. 导出为 CSV (建议使用 utf-8-sig 以便在 Excel 中直接打开不乱码)
    df_metrics.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"数据已导出至: \n{output_file}")
    return df_metrics
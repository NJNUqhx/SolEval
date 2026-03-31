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